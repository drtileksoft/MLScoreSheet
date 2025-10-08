using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace MLScoreSheet.Core;

public sealed class OnnxYoloFidDetector : IDisposable
{
        public readonly InferenceSession _session;
        private readonly int _imgsz;
        private readonly float _confThr;
        private readonly float _iouThr;
        private readonly int _maxDet;

        public OnnxYoloFidDetector(
            IResourceProvider resourceProvider,
            string logicalModelName = "best.onnx",
            int imgsz = 1024,
            float confThr = 0.05f,
            float iouThr = 0.60f,
            int maxDet = 200)
        {
            using var s = resourceProvider.OpenReadAsync(logicalModelName).GetAwaiter().GetResult();
            using var ms = new MemoryStream();
            s.CopyTo(ms);
            _session = new InferenceSession(ms.ToArray());
            _imgsz = imgsz;
            _confThr = confThr;
            _iouThr = iouThr;
            _maxDet = maxDet;
        }

        public void Dispose() => _session.Dispose();

        public struct Det { public float Cx, Cy, W, H, Conf; }

        /// <summary>Vrátí (cx,cy,w,h,conf) v souřadnicích původní fotky.</summary>
        public List<Det> Detect(SKBitmap photoBgr)
        {
            // 1) Letterbox → čtverec, normalizace, CHW
            var (inputBitmap, gain, padX, padY) = LetterboxToSquare(photoBgr, _imgsz, 114);
            using var input = inputBitmap;
            var tensor = BitmapToTensorCHW01(input);

            // 2) Inference
            string inputName = null!;
            foreach (var k in _session.InputMetadata.Keys) { inputName = k; break; }
            var feeds = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };
            using var results = _session.Run(feeds);

            // 3A) Single-tensor NMS: [1, N, 6]  =>  x1 y1 x2 y2 score class
            if (TryMapSingleNms(results, out var single))
            {
                int n = single.Dimensions[1];
                var dets = new List<Det>(n);
                for (int i = 0; i < n; i++)
                {
                    float x1 = single[0, i, 0];
                    float y1 = single[0, i, 1];
                    float x2 = single[0, i, 2];
                    float y2 = single[0, i, 3];
                    float conf = single[0, i, 4];
                    if (conf < _confThr) continue;

                    float cxL = 0.5f * (x1 + x2);
                    float cyL = 0.5f * (y1 + y2);
                    float wL = MathF.Max(0f, x2 - x1);
                    float hL = MathF.Max(0f, y2 - y1);

                    float cx = (cxL - padX) / gain;
                    float cy = (cyL - padY) / gain;
                    float w = wL / gain;
                    float h = hL / gain;

                    dets.Add(new Det { Cx = cx, Cy = cy, W = w, H = h, Conf = conf });
                }
                // export s NMS už je odduplikovaný, ale pro jistotu lehké NMS ještě jednou:
                return Nms(dets, _iouThr, _maxDet);
            }

            // 3B) Multi-tensor NMS: boxes[1,N,4] + scores[1,N] (+num_dets[1])
            if (TryMapNmsOutputs(results, out var boxes, out var scores, out int nNms))
            {
                int maxDet = Convert.ToInt32((boxes.Dimensions.Length >= 2) ? boxes.Dimensions[1] : scores.Length);
                int n = Math.Clamp(nNms >= 0 ? nNms : maxDet, 0, maxDet);
                var dets = new List<Det>(n);
                for (int i = 0; i < n; i++)
                {
                    float x1 = boxes[0, i, 0];
                    float y1 = boxes[0, i, 1];
                    float x2 = boxes[0, i, 2];
                    float y2 = boxes[0, i, 3];
                    float conf = scores[0, i];
                    if (conf < _confThr) continue;

                    float cxL = 0.5f * (x1 + x2);
                    float cyL = 0.5f * (y1 + y2);
                    float wL = MathF.Max(0f, x2 - x1);
                    float hL = MathF.Max(0f, y2 - y1);

                    float cx = (cxL - padX) / gain;
                    float cy = (cyL - padY) / gain;
                    float w = wL / gain;
                    float h = hL / gain;

                    dets.Add(new Det { Cx = cx, Cy = cy, W = w, H = h, Conf = conf });
                }
                return Nms(dets, _iouThr, _maxDet);
            }

            // 3C) RAW YOLO výstup (bez NMS): dekóduj + NMS
            if (TryMapRawOutputs(results, out var raw, out bool chw))
                return DecodeRawAndNms(raw, chw, gain, padX, padY);

            throw new InvalidOperationException("ONNX výstupy neodpovídají žádnému podporovanému tvaru. Vypiš názvy/rozměry přes DebugListOutputs(session).");
        }

        // ---------- Mapovače výstupů ----------
        private static bool TryMapSingleNms(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
            out DenseTensor<float> single)
        {
            single = null!;
            foreach (var r in results)
            {
                if (r.Value is DenseTensor<float> tf)
                {
                    var d = tf.Dimensions; // čekáme [1, N, 6]
                    if (d.Length == 3 && d[0] == 1 && d[2] == 6)
                    {
                        single = tf; return true;
                    }
                }
            }
            return false;
        }

        private static bool TryMapNmsOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
            out DenseTensor<float> boxes, out DenseTensor<float> scores, out int num)
        {
            boxes = null!; scores = null!; num = -1;

            foreach (var r in results)
            {
                if (r.Value is DenseTensor<float> tf)
                {
                    var d = tf.Dimensions;
                    if (d.Length == 3 && d[0] == 1 && d[2] == 4) boxes ??= tf;          // [1,N,4]
                    else if (d.Length == 2 && d[0] == 1) scores ??= tf;         // [1,N]
                    else if (d.Length == 1 && d[0] == 1) num = (int)Math.Round(tf[0]); // [1]
                }
                else if (r.Value is DenseTensor<long> tl)
                {
                    var d = tl.Dimensions;
                    if (d.Length == 1 && d[0] == 1) num = (int)tl[0];
                }
            }
            return boxes != null && scores != null;
        }

        private static bool TryMapRawOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
            out DenseTensor<float> raw, out bool chw /* true => [1,K,N], false => [1,N,K] */)
        {
            raw = null!; chw = true;
            foreach (var r in results)
            {
                if (r.Value is DenseTensor<float> tf)
                {
                    var d = tf.Dimensions;
                    if (d.Length != 3 || d[0] != 1) continue;
                    if (d[1] >= 6 && d[1] <= 200 && d[2] >= 512) { raw = tf; chw = true; return true; } // [1,K,N]
                    if (d[2] >= 6 && d[2] <= 200 && d[1] >= 512) { raw = tf; chw = false; return true; } // [1,N,K]
                }
            }
            return false;
        }

        // ---------- Dekódování RAW + NMS ----------
        private List<Det> DecodeRawAndNms(DenseTensor<float> raw, bool chw, float gain, float padX, float padY)
        {
            int K = chw ? raw.Dimensions[1] : raw.Dimensions[2];
            int N = chw ? raw.Dimensions[2] : raw.Dimensions[1];
            int ncls = Math.Max(1, K - 5);

            int baseCnt = Math.Max(1, (int)Math.Round(N / 21f));
            int n0 = Math.Min(N, 16 * baseCnt);
            int n1 = Math.Min(N - n0, 4 * baseCnt);
            int n2 = Math.Max(0, N - n0 - n1);

            int[] strides = { 8, 16, 32 };
            int[] counts = { n0, n1, n2 };
            int[] offsets = { 0, n0, n0 + n1 };

            var cand = new List<Det>(_maxDet * 2);

            foreach (bool needDecode in new[] { true, false })
            {
                cand.Clear();
                for (int lvl = 0; lvl < 3; lvl++)
                {
                    int stride = strides[lvl];
                    int gCount = counts[lvl];
                    if (gCount <= 0) continue;

                    // aproximační grid – není kritický, používá se jen u needDecode
                    int g = Math.Max(1, (int)Math.Round((float)_imgsz / stride));

                    int off = offsets[lvl];
                    for (int idx = 0; idx < gCount; idx++)
                    {
                        int gi = idx % g;
                        int gj = idx / g;
                        int nIdx = off + idx;

                        float px = chw ? raw[0, 0, nIdx] : raw[0, nIdx, 0];
                        float py = chw ? raw[0, 1, nIdx] : raw[0, nIdx, 1];
                        float pw = chw ? raw[0, 2, nIdx] : raw[0, nIdx, 2];
                        float ph = chw ? raw[0, 3, nIdx] : raw[0, nIdx, 3];
                        float po = chw ? raw[0, 4, nIdx] : raw[0, nIdx, 4];

                        float clsMax = 0f;
                        for (int c = 0; c < ncls; c++)
                        {
                            float pc = chw ? raw[0, 5 + c, nIdx] : raw[0, nIdx, 5 + c];
                            clsMax = MathF.Max(clsMax, Prob(pc));
                        }
                        float conf = Prob(po) * clsMax;
                        if (conf < _confThr) continue;

                        float cx, cy, w, h;
                        if (needDecode)
                        {
                            cx = ((Sigmoid(px) * 2f - 0.5f) + gi) * stride;
                            cy = ((Sigmoid(py) * 2f - 0.5f) + gj) * stride;
                            w = MathF.Pow(Sigmoid(pw) * 2f, 2f) * stride;
                            h = MathF.Pow(Sigmoid(ph) * 2f, 2f) * stride;
                        }
                        else
                        {
                            cx = px; cy = py; w = pw; h = ph;
                        }

                        if (!IsFinite(cx) || !IsFinite(cy) || !IsFinite(w) || !IsFinite(h)) continue;
                        if (w <= 1 || h <= 1) continue;

                        float cx0 = (cx - padX) / gain;
                        float cy0 = (cy - padY) / gain;
                        float w0 = w / gain;
                        float h0 = h / gain;

                        cand.Add(new Det { Cx = cx0, Cy = cy0, W = w0, H = h0, Conf = conf });
                        if (cand.Count >= _maxDet * 2) break;
                    }
                    if (cand.Count >= _maxDet * 2) break;
                }
                var dets = Nms(cand, _iouThr, _maxDet);
                if (dets.Count >= 3) return dets;
            }
            return Nms(cand, _iouThr, _maxDet);
        }

        // ---------- Letterbox & tensor ----------
        private static (SKBitmap, float gain, float padX, float padY) LetterboxToSquare(SKBitmap src, int size, byte pad = 114)
        {
            int iw = src.Width, ih = src.Height;
            float scale = Math.Min((float)size / iw, (float)size / ih);
            int nw = (int)Math.Round(iw * scale);
            int nh = (int)Math.Round(ih * scale);
            int padX = (size - nw) / 2;
            int padY = (size - nh) / 2;

            var dst = new SKBitmap(size, size, SKColorType.Bgra8888, SKAlphaType.Premul);
            using var c = new SKCanvas(dst);
            c.Clear(new SKColor(pad, pad, pad));
            c.DrawBitmap(src, new SKRect(0, 0, iw, ih), new SKRect(padX, padY, padX + nw, padY + nh));
            c.Flush();
            return (dst, scale, padX, padY);
        }

        private static DenseTensor<float> BitmapToTensorCHW01(SKBitmap b)
        {
            var t = new DenseTensor<float>(new[] { 1, 3, b.Height, b.Width });
            unsafe
            {
                for (int y = 0; y < b.Height; y++)
                {
                    var p = (uint*)b.GetPixels() + y * b.Width;
                    for (int x = 0; x < b.Width; x++)
                    {
                        uint c = p[x];
                        float B = (byte)(c & 0xFF) / 255f;
                        float G = (byte)((c >> 8) & 0xFF) / 255f;
                        float R = (byte)((c >> 16) & 0xFF) / 255f;
                        t[0, 0, y, x] = R;
                        t[0, 1, y, x] = G;
                        t[0, 2, y, x] = B;
                    }
                }
            }
            return t;
        }

        // ---------- NMS + IoU ----------
        private static List<Det> Nms(List<Det> dets, float iouThr, int maxDet)
        {
            // setřídíme ručně (bez LINQ) podle conf
            for (int i = 0; i < dets.Count - 1; i++)
            {
                int best = i;
                for (int j = i + 1; j < dets.Count; j++)
                    if (dets[j].Conf > dets[best].Conf) best = j;
                if (best != i) (dets[i], dets[best]) = (dets[best], dets[i]);
            }

            var keep = new List<Det>(Math.Min(dets.Count, maxDet));
            foreach (var d in dets)
            {
                bool suppr = false;
                foreach (var k in keep)
                {
                    if (IoUxywh(d, k) > iouThr) { suppr = true; break; }
                }
                if (!suppr)
                {
                    keep.Add(d);
                    if (keep.Count >= maxDet) break;
                }
            }
            return keep;
        }

        public static float IoUxywh(Det a, Det b)
        {
            float ax1 = a.Cx - a.W * 0.5f, ay1 = a.Cy - a.H * 0.5f;
            float ax2 = a.Cx + a.W * 0.5f, ay2 = a.Cy + a.H * 0.5f;
            float bx1 = b.Cx - b.W * 0.5f, by1 = b.Cy - b.H * 0.5f;
            float bx2 = b.Cx + b.W * 0.5f, by2 = b.Cy + b.H * 0.5f;

            float ix1 = MathF.Max(ax1, bx1), iy1 = MathF.Max(ay1, by1);
            float ix2 = MathF.Min(ax2, bx2), iy2 = MathF.Min(ay2, by2);
            float iw = MathF.Max(0, ix2 - ix1), ih = MathF.Max(0, iy2 - iy1);
            float inter = iw * ih;

            float areaA = MathF.Max(0, ax2 - ax1) * MathF.Max(0, ay2 - ay1);
            float areaB = MathF.Max(0, bx2 - bx1) * MathF.Max(0, by2 - by1);
            float union = areaA + areaB - inter;
            return union <= 0 ? 0f : inter / union;
        }

        // ---------- Matematické pomocníky ----------
        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
        private static float Prob(float v) => (v < 0f || v > 1f) ? Sigmoid(v) : Math.Clamp(v, 0f, 1f);
        private static bool IsFinite(float x) => !float.IsNaN(x) && !float.IsInfinity(x);

        // ---------- Debug výpis výstupů ----------
        public static string DebugListOutputs(InferenceSession session)
        {
            var metadata = session.OutputMetadata.Select(kv => new KeyValuePair<string, IReadOnlyList<int>>(kv.Key, kv.Value.Dimensions));
            return DebugListOutputs(metadata);
        }

        public static string DebugListOutputs(IEnumerable<KeyValuePair<string, IReadOnlyList<int>>> metadata)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var kv in metadata)
            {
                sb.Append(kv.Key).Append(": [");
                var dims = kv.Value;
                if (dims != null)
                {
                    int count = dims.Count;
                    for (int i = 0; i < count; i++)
                    {
                        if (i > 0) sb.Append(",");
                        var dim = dims[i];
                        sb.Append(dim);
                    }
                }
                sb.AppendLine("]");
            }
            return sb.ToString();
        }
    }



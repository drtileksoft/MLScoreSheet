using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Maui.Storage;
using MLScoreSheetCounter;
using SkiaSharp;

namespace YourApp.Services;

public static class SheetScoreEngine
{
    public sealed class ScoreOverlayResult : IDisposable
    {
        public int Total { get; init; }
        public float ThresholdUsed { get; init; }
        public SKBitmap Overlay { get; init; } = default!;
        public void Dispose() => Overlay?.Dispose();
    }

    // --------------------------- Public API ---------------------------

    public static async Task<int> ComputeTotalScoreAsync(
        Stream photoStream,
        string yoloOnnxLogical = "yolo_fids.onnx",
        string rectsJsonLogical = "boxes_rects.json",
        string fidPngLogical = "ocrscoresheetfiducials.png",
        float fixedThreshold = 0.35f,
        bool autoThreshold = false,
        float autoMin = 0.25f,
        float autoMax = 0.65f,
        float padFrac = 0.08f,
        float openFrac = 0.03f)
    {
        //using var photo = SKBitmap.Decode(photoStream) ?? throw new InvalidOperationException("Nelze dekódovat foto.");
        using var photo = DecodeLandscapePhoto(photoStream);
        var tpl = await LoadRectsJsonAsync(rectsJsonLogical);

        // --- Zdrojové fidy z fotky přes ONNX YOLO ---
        var src = DetectSrcFidsWithOnnx(photo, yoloOnnxLogical);

        // --- Cílové fidy v šabloně (z PNG, pokud je; jinak rohy size) ---
        var dst = await TryDetectDstFidsOrCornersAsync(fidPngLogical, tpl.SizeW, tpl.SizeH);

        // --- Homografie + warp ---
        var H = ComputeHomography(src, dst);
        using var warped = WarpToTemplate(photo, H, tpl.SizeW, tpl.SizeH);

        // --- % černé v ROI ---
        var pList = new float[tpl.Rects.Count];
        for (int i = 0; i < tpl.Rects.Count; i++)
            pList[i] = FillRatioFromRoi(warped, tpl.Rects[i], padFrac, openFrac);

        // --- práh ---
        float thr = autoThreshold ? AutoThresholdKMeans(pList, autoMin, autoMax, fixedThreshold, 0.12f)
                                  : fixedThreshold;

        // --- 3×2 skupiny a součet ---
        return TotalScoreFromItems(tpl.Rects, pList, thr);
    }

    public static async Task<ScoreOverlayResult> ComputeTotalScoreWithOverlayAsync(
        Stream photoStream,
        string yoloOnnxLogical = "yolo_fids.onnx",
        string rectsJsonLogical = "boxes_rects.json",
        string fidPngLogical = "ocrscoresheetfiducials.png",
        float fixedThreshold = 0.35f,
        bool autoThreshold = false,
        float autoMin = 0.25f,
        float autoMax = 0.65f,
        float padFrac = 0.08f,
        float openFrac = 0.03f)
    {
        //using var photo = SKBitmap.Decode(photoStream) ?? throw new InvalidOperationException("Nelze dekódovat foto.");
        using var photo = DecodeLandscapePhoto(photoStream);
        var tpl = await LoadRectsJsonAsync(rectsJsonLogical);

        var src = DetectSrcFidsWithOnnx(photo, yoloOnnxLogical);
        var dst = await TryDetectDstFidsOrCornersAsync(fidPngLogical, tpl.SizeW, tpl.SizeH);

        var H = ComputeHomography(src, dst);
        var warped = WarpToTemplate(photo, H, tpl.SizeW, tpl.SizeH); // vracíme v overlay

        var pList = new float[tpl.Rects.Count];

        for (int i = 0; i < tpl.Rects.Count; i++)
            pList[i] = (float)(GetFillPercentLocalContrast(warped, tpl.Rects[i]) / 100.0);

        float thr = autoThreshold ? AutoThresholdKMeans(pList, autoMin, autoMax, fixedThreshold, 0.12f)
                                  : fixedThreshold;

        var res = ScoreSelector3x2.SumWinnerTakesAll(tpl.Rects, pList, thr);

        // groups pro orámování a sloty 0..5
        var groups = BuildGroupsGrid3x2(tpl.Rects, pList); // interní helper v tomto souboru :contentReference[oaicite:2]{index=2}

        // overlaye (warped + originál) s novou logikou barvení + popisky
        var overlayWarped = MakeWarpedOverlay(
            warped,
            tpl.Rects,
            pList,
            res.WinnerIndices,
            groups
        );

        // ... návratová hodnota:
        return new ScoreOverlayResult
        {
            Total = res.Total,           // <<<< beru číslo z SumWinnerTakesAll
            ThresholdUsed = thr,
            Overlay = overlayWarped
        };

    }

    // % černé s lokálním kontrastem (stínuvzdorné)
    static double GetFillPercentLocalContrast(SKBitmap src, SKRectI rect)
    {
        // trochu rozšíříme výřez, aby histogram zachytil i okolní čáry/okraje
        const int margin = 3;
        var expanded = new SKRectI(
            Math.Max(0, rect.Left - margin),
            Math.Max(0, rect.Top - margin),
            Math.Min(src.Width, rect.Right + margin),
            Math.Min(src.Height, rect.Bottom + margin)
        );
        if (expanded.Width <= 0 || expanded.Height <= 0)
            return 0;
        rect = expanded;

        // 0) ignoruj rámeček – zarovnej 12 % z každé strany (klidně zvedni na 0.18)
        int ix = Math.Max(1, (int)Math.Round(rect.Width * 0.12));
        int iy = Math.Max(1, (int)Math.Round(rect.Height * 0.12));
        var r = new SKRectI(rect.Left + ix, rect.Top + iy, rect.Right - ix, rect.Bottom - iy);
        if (r.Width <= 2 || r.Height <= 2) return 0;

        using var roi = new SKBitmap(r.Width, r.Height, src.ColorType, src.AlphaType);
        using (var cc = new SKCanvas(roi)) cc.DrawBitmap(src, r, new SKRect(0, 0, r.Width, r.Height));
        using var pm = new SKPixmap(roi.Info, roi.GetPixels(out _));

        // 1) histogram TEMNOSTI (dark = 255 - luminance)
        Span<int> hist = stackalloc int[256]; hist.Clear();
        int total = roi.Width * roi.Height;
        unsafe
        {
            byte* p = (byte*)pm.GetPixels();
            int bpp = pm.Info.BytesPerPixel; // RGBA8888 => 4
            for (int y = 0; y < roi.Height; y++)
            {
                byte* row = p + y * pm.RowBytes;
                for (int x = 0; x < roi.Width; x++)
                {
                    byte B = row[x * bpp + 0], G = row[x * bpp + 1], R = row[x * bpp + 2];
                    int y8 = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    int d = 255 - y8;           // temnost
                    hist[d]++;
                }
            }
        }

        // 2) robustní min/max z percentilů (např. 5. a 95.)
        int pLo = PercentileFromHist(hist, total, 0.05);
        int pHi = PercentileFromHist(hist, total, 0.95);
        if (pHi <= pLo) { pLo = Math.Max(0, pLo - 1); pHi = Math.Min(255, pLo + 1); }
        double range = pHi - pLo;

        // 3) druhý průchod: normalizace na 0..255 a histogram pro Otsu
        Span<int> histN = stackalloc int[256]; histN.Clear();
        int blacks = 0; // vyplníme až po určení prahu
        unsafe
        {
            using var pm2 = new SKPixmap(roi.Info, roi.GetPixels(out _));
            byte* p = (byte*)pm2.GetPixels();
            int bpp = pm2.Info.BytesPerPixel;
            for (int y = 0; y < roi.Height; y++)
            {
                byte* row = p + y * pm2.RowBytes;
                for (int x = 0; x < roi.Width; x++)
                {
                    byte B = row[x * bpp + 0], G = row[x * bpp + 1], R = row[x * bpp + 2];
                    int y8 = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    int d = 255 - y8;
                    // kontrastní roztažení podle lokálního min/max
                    double dn = (d - pLo) / range;
                    if (dn < 0) dn = 0; else if (dn > 1) dn = 1;
                    int q = (int)Math.Round(dn * 255.0);
                    histN[q]++;
                }
            }
        }

        // 4) Otsu na normalizovaném histogramu → práh qThr (0..255)
        int qThr = OtsuThreshold(histN, total);

        // 5) finální spočtení podílu "černých" (>= práh) v jedné smyčce
        unsafe
        {
            using var pm3 = new SKPixmap(roi.Info, roi.GetPixels(out _));
            byte* p = (byte*)pm3.GetPixels();
            int bpp = pm3.Info.BytesPerPixel;
            for (int y = 0; y < roi.Height; y++)
            {
                byte* row = p + y * pm3.RowBytes;
                for (int x = 0; x < roi.Width; x++)
                {
                    byte B = row[x * bpp + 0], G = row[x * bpp + 1], R = row[x * bpp + 2];
                    int y8 = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    int d = 255 - y8;
                    double dn = (d - pLo) / range;
                    if (dn < 0) dn = 0; else if (dn > 1) dn = 1;
                    int q = (int)Math.Round(dn * 255.0);
                    if (q >= qThr) blacks++;
                }
            }
        }

        double pct = 100.0 * blacks / Math.Max(1, total);
        if (double.IsNaN(pct)) pct = 0;
        if (pct < 0) pct = 0; else if (pct > 100) pct = 100;
        return pct;

        // ---- lokální pomocné funkce ----
        static int PercentileFromHist(Span<int> h, int tot, double p)
        {
            int target = (int)Math.Round(p * tot);
            int cum = 0;
            for (int i = 0; i < 256; i++) { cum += h[i]; if (cum >= target) return i; }
            return 255;
        }
        static int OtsuThreshold(Span<int> h, int tot)
        {
            double sum = 0; for (int t = 0; t < 256; t++) sum += t * h[t];
            double sumB = 0; int wB = 0; double varMax = -1; int thr = 128;
            for (int t = 0; t < 256; t++)
            {
                wB += h[t]; if (wB == 0) continue;
                int wF = tot - wB; if (wF == 0) break;
                sumB += t * h[t];
                double mB = sumB / wB, mF = (sum - sumB) / wF;
                double varBetween = wB * wF * (mB - mF) * (mB - mF);
                if (varBetween > varMax) { varMax = varBetween; thr = t; }
            }
            return thr;
        }
    }

    static SKBitmap DecodeLandscapePhoto(Stream photoStream)
    {
        var decoded = SKBitmap.Decode(photoStream) ?? throw new InvalidOperationException("Nelze dekódovat foto.");

        if (decoded.Width >= decoded.Height)
            return decoded;

        var rotated = new SKBitmap(decoded.Height, decoded.Width, decoded.ColorType, decoded.AlphaType);
        using (var canvas = new SKCanvas(rotated))
        {
            canvas.Translate(0, rotated.Height);
            canvas.RotateDegrees(-90);
            canvas.DrawBitmap(decoded, 0, 0);
        }

        decoded.Dispose();
        return rotated;
    }


    // --- Robustní % černé pro jeden box ---
    static double GetFillPercent(SKBitmap src, SKRectI rect)
    {
        // 1) Ořez okrajů (rámeček/linky) ~12 % z každé strany
        int insetX = Math.Max(1, (int)Math.Round(rect.Width * 0.12));
        int insetY = Math.Max(1, (int)Math.Round(rect.Height * 0.12));
        var r = new SKRectI(
            rect.Left + insetX, rect.Top + insetY,
            rect.Right - insetX, rect.Bottom - insetY
        );
        if (r.Width <= 2 || r.Height <= 2) return 0;

        using var sub = new SKBitmap(r.Width, r.Height, src.ColorType, src.AlphaType);
        using (var c = new SKCanvas(sub))
            c.DrawBitmap(src, r, new SKRect(0, 0, r.Width, r.Height));

        // 2) Grayscale (luminance) a TEMNOST (255 - Y)
        using var pm = new SKPixmap(sub.Info, sub.GetPixels(out _));
        Span<byte> dark = pm.GetPixelSpan().Length == 0
            ? new byte[0]
            : new byte[sub.Width * sub.Height];

        int idx = 0;
        unsafe
        {
            byte* p = (byte*)pm.GetPixels();
            int bpp = pm.Info.BytesPerPixel; // RGBA8888 => 4
            for (int y = 0; y < sub.Height; y++)
            {
                byte* row = p + y * pm.RowBytes;
                for (int x = 0; x < sub.Width; x++)
                {
                    byte B = row[x * bpp + 0];
                    byte G = row[x * bpp + 1];
                    byte R = row[x * bpp + 2];
                    // Rec. 709 luminance ≈ 0.2126 R + 0.7152 G + 0.0722 B
                    int y8 = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    byte d = (byte)(255 - y8); // TEMNOST = čím větší, tím černější
                    dark[idx++] = d;
                }
            }
        }

        if (dark.Length == 0) return 0;

        // 3) Otsu práh na TEMNOSTI (robustní vůči stínu)
        Span<int> hist = stackalloc int[256];
        for (int i = 0; i < dark.Length; i++) hist[dark[i]]++;

        int total = dark.Length;
        double sum = 0;
        for (int t = 0; t < 256; t++) sum += t * hist[t];

        double sumB = 0;
        int wB = 0;
        double varMax = -1;
        int thr = 128;

        for (int t = 0; t < 256; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            int wF = total - wB;
            if (wF == 0) break;

            sumB += t * hist[t];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double varBetween = wB * wF * (mB - mF) * (mB - mF);
            if (varBetween > varMax)
            {
                varMax = varBetween;
                thr = t;
            }
        }

        // 4) Podíl "černých" pixelů (temnost >= práh)
        int black = 0;
        for (int i = 0; i < dark.Length; i++)
            if (dark[i] >= thr) black++;

        double pct = 100.0 * black / total;

        // 5) Bezpečné ořezání intervalu
        if (double.IsNaN(pct) || pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        return pct;
    }


    private static string SavePng(SKBitmap bmp, string name)
    {
        var path = Path.Combine(FileSystem.CacheDirectory, $"{name}.png");
        using var img = SKImage.FromBitmap(bmp);
        using var data = img.Encode(SKEncodedImageFormat.Png, 95);
        using var fs = File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        data.SaveTo(fs);
        return path;
    }

    // ------------------------ YOLO ONNX fiducials (src) ------------------------
    private static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL)
        DetectSrcFidsWithOnnx(SKBitmap photo, string yoloOnnxLogical)
    {
        using var det = new OnnxYoloFidDetector(yoloOnnxLogical, imgsz: 1024);
        System.Diagnostics.Debug.WriteLine(OnnxYoloFidDetector.DebugListOutputs(det._session)); // pokud si zpřístupníš session
        var raw = det.Detect(photo); // (cx,cy,w,h,conf)

        if (raw.Count < 3)
            throw new InvalidOperationException($"Nedostatek fiducialů na fotce: {raw.Count}");

        // dedup podle IoU (xywh), necháme vyšší conf
        var dets = DedupBoxes(raw, iouThr: 0.6f);

        // vyber 4 a přiřaď podle geometrie (stejně jako v Pythonu)
        var pts = dets.Select(d => (d.Cx, d.Cy, d.Conf)).ToList();
        if (pts.Count >= 4)
        {
            var ordered = AssignByGeometry(pts, photo.Width, photo.Height); // TL..BL
            return (ordered.TL, ordered.TR, ordered.BR, ordered.BL);
        }
        else
        {
            // fallback: dopočti chybějící
            var approx = OrderByCorners(pts.Select(p => new SKPoint((float)p.Cx, (float)p.Cy)).ToList(), photo.Width, photo.Height);
            var known = new Dictionary<string, (float x, float y, float conf)>();
            string[] NAMES = { "TL", "TR", "BR", "BL" };
            for (int i = 0; i < pts.Count && i < 4; i++)
            {
                known[NAMES[i]] = ((float)approx[i].X, (float)approx[i].Y, (float)pts[i].Conf);
            }
            var miss = CloseParallelogram(known);
            if (miss.name != null) known[miss.name] = (miss.x, miss.y, 0f);

            var TL = new SKPoint(known["TL"].x, known["TL"].y);
            var TR = new SKPoint(known["TR"].x, known["TR"].y);
            var BR = new SKPoint(known["BR"].x, known["BR"].y);
            var BL = new SKPoint(known["BL"].x, known["BL"].y);
            return (TL, TR, BR, BL);
        }
    }

    private static List<OnnxYoloFidDetector.Det> DedupBoxes(List<OnnxYoloFidDetector.Det> dets, float iouThr)
    {
        var sorted = dets.OrderByDescending(d => d.Conf).ToList();
        var keep = new List<OnnxYoloFidDetector.Det>();
        foreach (var d in sorted)
        {
            bool dup = keep.Any(k => IoUxywh(d, k) > iouThr);
            if (!dup) keep.Add(d);
        }
        return keep;
    }
    private static float IoUxywh(OnnxYoloFidDetector.Det a, OnnxYoloFidDetector.Det b)
    {
        (float ax1, float ay1, float ax2, float ay2) = (a.Cx - a.W / 2, a.Cy - a.H / 2, a.Cx + a.W / 2, a.Cy + a.H / 2);
        (float bx1, float by1, float bx2, float by2) = (b.Cx - b.W / 2, b.Cy - b.H / 2, b.Cx + b.W / 2, b.Cy + b.H / 2);
        float ix1 = Math.Max(ax1, bx1), iy1 = Math.Max(ay1, by1);
        float ix2 = Math.Min(ax2, bx2), iy2 = Math.Min(ay2, by2);
        float iw = Math.Max(0, ix2 - ix1), ih = Math.Max(0, iy2 - iy1);
        float inter = iw * ih;
        float ua = (ax2 - ax1) * (ay1 - ay2) + (bx2 - bx1) * (by1 - by2) - inter; // pozor na znaménka
        ua = Math.Abs(ua); // jistota
        return ua <= 0 ? 0 : inter / ua;
    }

    private static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL)
        AssignByGeometry(List<(float x, float y, float conf)> pts, int W, int H)
    {
        // Minimalizace vzdálenosti k rohům (TL, TR, BR, BL)
        var targets = new[] { new SKPoint(0, 0), new SKPoint(W, 0), new SKPoint(W, H), new SKPoint(0, H) };
        double diag = Math.Sqrt((double)W * W + (double)H * H);
        var cand = pts.Take(Math.Min(10, pts.Count)).ToList();

        double bestCost = 1e18;
        (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) best = default;

        // všechny kombinace 4 z N + permutace 4!
        var idxs = Enumerable.Range(0, cand.Count).ToArray();
        foreach (var combo in Combinations(idxs, 4))
        {
            var sub = combo.Select(i => cand[i]).ToArray();
            for (int p0 = 0; p0 < 4; p0++)
                for (int p1 = 0; p1 < 4; p1++) if (p1 != p0)
                        for (int p2 = 0; p2 < 4; p2++) if (p2 != p0 && p2 != p1)
                            {
                                int p3 = 6 - p0 - p1 - p2; // 0+1+2+3=6
                                double cost = 0, confSum = 0;
                                var s0 = new SKPoint(sub[p0].x, sub[p0].y);
                                var s1 = new SKPoint(sub[p1].x, sub[p1].y);
                                var s2 = new SKPoint(sub[p2].x, sub[p2].y);
                                var s3 = new SKPoint(sub[p3].x, sub[p3].y);
                                var S = new[] { s0, s1, s2, s3 };
                                for (int k = 0; k < 4; k++)
                                {
                                    cost += Distance(S[k], targets[k]) / diag;
                                    confSum += sub[(k == 0 ? p0 : k == 1 ? p1 : k == 2 ? p2 : p3)].conf;
                                }
                                cost -= 0.05 * (confSum / 4.0);
                                if (cost < bestCost)
                                {
                                    bestCost = cost; best = (s0, s1, s2, s3);
                                }
                            }
        }
        if (bestCost == 1e18) throw new InvalidOperationException("Nelze přiřadit 4 body.");
        return best;

        static IEnumerable<int[]> Combinations(int[] arr, int choose)
        {
            int n = arr.Length;
            var idx = Enumerable.Range(0, choose).ToArray();
            while (true)
            {
                yield return idx.Select(i => arr[i]).ToArray();
                int i;
                for (i = choose - 1; i >= 0; i--)
                {
                    if (idx[i] != i + n - choose) break;
                }
                if (i < 0) yield break;
                idx[i]++;
                for (int j = i + 1; j < choose; j++) idx[j] = idx[j - 1] + 1;
            }
        }
        static double Distance(SKPoint a, SKPoint b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));
    }

    private static (string name, float x, float y) CloseParallelogram(Dictionary<string, (float x, float y, float conf)> known)
    {
        var names = new[] { "TL", "TR", "BR", "BL" };
        var miss = names.FirstOrDefault(n => !known.ContainsKey(n));
        if (miss == null) return (null!, 0, 0);
        var v = known.ToDictionary(kv => kv.Key, kv => new SKPoint(kv.Value.x, kv.Value.y));
        SKPoint p;
        switch (miss)
        {
            case "TL":
                if (v.ContainsKey("TR") && v.ContainsKey("BR") && v.ContainsKey("BL")) p = new SKPoint(v["TR"].X + (v["BL"].X - v["BR"].X), v["TR"].Y + (v["BL"].Y - v["BR"].Y));
                else p = Avg(v.Values);
                break;
            case "TR":
                if (v.ContainsKey("TL") && v.ContainsKey("BR") && v.ContainsKey("BL")) p = new SKPoint(v["TL"].X + (v["BR"].X - v["BL"].X), v["TL"].Y + (v["BR"].Y - v["BL"].Y));
                else p = Avg(v.Values);
                break;
            case "BR":
                if (v.ContainsKey("TL") && v.ContainsKey("TR") && v.ContainsKey("BL")) p = new SKPoint(v["BL"].X + (v["TR"].X - v["TL"].X), v["BL"].Y + (v["TR"].Y - v["TL"].Y));
                else p = Avg(v.Values);
                break;
            default: // BL
                if (v.ContainsKey("TL") && v.ContainsKey("TR") && v.ContainsKey("BR")) p = new SKPoint(v["TL"].X + (v["BR"].X - v["TR"].X), v["TL"].Y + (v["BR"].Y - v["TR"].Y));
                else p = Avg(v.Values);
                break;
        }
        return (miss, p.X, p.Y);

        static SKPoint Avg(IEnumerable<SKPoint> P) { float sx = 0, sy = 0; int c = 0; foreach (var q in P) { sx += q.X; sy += q.Y; c++; } return new SKPoint(sx / c, sy / c); }
    }

    // ------------------------ Template rects JSON ------------------------
    sealed class TemplateData
    {
        public int SizeW { get; init; }
        public int SizeH { get; init; }
        public List<SKRectI> Rects { get; init; } = new();
    }
    static async Task<TemplateData> LoadRectsJsonAsync(string logicalName)
    {
        using var s = await FileSystem.OpenAppPackageFileAsync(logicalName);
        using var ms = new MemoryStream(); await s.CopyToAsync(ms); ms.Position = 0;
        using var doc = await System.Text.Json.JsonDocument.ParseAsync(ms);
        var root = doc.RootElement;

        var size = root.GetProperty("size").EnumerateArray().ToArray();
        int Wt = size[0].GetInt32(), Ht = size[1].GetInt32();

        var rects = new List<SKRectI>();
        foreach (var r in root.GetProperty("rects").EnumerateArray())
        {
            var a = r.EnumerateArray().ToArray();
            int x = a[0].GetInt32(), y = a[1].GetInt32(), w = a[2].GetInt32(), h = a[3].GetInt32();
            rects.Add(new SKRectI(x, y, x + w, y + h));
        }
        return new TemplateData { SizeW = Wt, SizeH = Ht, Rects = rects };
    }

    // ------------------------ Dst fiducials (šablona PNG nebo rohy) ------------------------
    static async Task<(SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL)>
        TryDetectDstFidsOrCornersAsync(string fidPngLogicalName, int Wt, int Ht)
    {
        try
        {
            using var fid = await FileSystem.OpenAppPackageFileAsync(fidPngLogicalName);
            using var bmp = SKBitmap.Decode(fid);
            if (bmp == null) throw new FileNotFoundException(fidPngLogicalName);
            return DetectFiducialsCentersInTemplate(bmp);
        }
        catch
        {
            return (new SKPoint(0, 0), new SKPoint(Wt - 1, 0), new SKPoint(Wt - 1, Ht - 1), new SKPoint(0, Ht - 1));
        }
    }

    static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) DetectFiducialsCentersInTemplate(SKBitmap src)
    {
        using var gray = ToGray(src);
        var (bin, _) = BinarizeObjectsWhite(gray);
        var comps = ConnectedComponents(bin);
        var top = comps.OrderByDescending(c => c.Count).Take(4).ToList();
        if (top.Count < 4) throw new InvalidOperationException("Ve fiducial šabloně nejsou 4 markery.");
        var pts = top.Select(c => new SKPoint((float)c.Cx, (float)c.Cy)).ToList();
        var ordered = OrderByCorners(pts, src.Width, src.Height);
        return (ordered[0], ordered[1], ordered[2], ordered[3]); // TL,TR,BR,BL
    }

    // ------------------------ Homography & Warp (Skia) ------------------------
    static float[] ComputeHomography((SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) src,
                                     (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) dst)
    {
        var s = new[] { src.TL, src.TR, src.BR, src.BL };
        var d = new[] { dst.TL, dst.TR, dst.BR, dst.BL };

        double[,] A = new double[8, 8];
        double[] b = new double[8];

        for (int i = 0; i < 4; i++)
        {
            double x = s[i].X, y = s[i].Y;
            double X = d[i].X, Y = d[i].Y;

            int r1 = i * 2, r2 = i * 2 + 1;
            A[r1, 0] = x; A[r1, 1] = y; A[r1, 2] = 1; A[r1, 3] = 0; A[r1, 4] = 0; A[r1, 5] = 0; A[r1, 6] = -x * X; A[r1, 7] = -y * X;
            A[r2, 0] = 0; A[r2, 1] = 0; A[r2, 2] = 0; A[r2, 3] = x; A[r2, 4] = y; A[r2, 5] = 1; A[r2, 6] = -x * Y; A[r2, 7] = -y * Y;
            b[r1] = X; b[r2] = Y;
        }

        var h = Solve8x8(A, b);
        return new float[] {
            (float)h[0], (float)h[1], (float)h[2],
            (float)h[3], (float)h[4], (float)h[5],
            (float)h[6], (float)h[7], 1f
        };
    }
    static double[] Solve8x8(double[,] A, double[] b)
    {
        int n = 8;
        double[,] M = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) M[i, j] = A[i, j];
            M[i, n] = b[i];
        }
        for (int i = 0; i < n; i++)
        {
            int piv = i;
            for (int r = i + 1; r < n; r++)
                if (Math.Abs(M[r, i]) > Math.Abs(M[piv, i])) piv = r;
            if (piv != i)
                for (int c = i; c <= n; c++) (M[i, c], M[piv, c]) = (M[piv, c], M[i, c]);

            double div = M[i, i];
            if (Math.Abs(div) < 1e-12) continue;
            for (int c = i; c <= n; c++) M[i, c] /= div;

            for (int r = 0; r < n; r++)
            {
                if (r == i) continue;
                double mul = M[r, i];
                for (int c = i; c <= n; c++) M[r, c] -= mul * M[i, c];
            }
        }
        var x = new double[n];
        for (int i = 0; i < n; i++) x[i] = M[i, n];
        return x;
    }

    static SKBitmap WarpToTemplate(SKBitmap src, float[] H, int Wt, int Ht)
    {
        var dst = new SKBitmap(Wt, Ht, SKColorType.Bgra8888, SKAlphaType.Premul);
        using var canvas = new SKCanvas(dst);
        canvas.Clear(SKColors.Black);

        // Skia matice odpovídá:
        // x' = (a*x + b*y + c) / (g*x + h*y + 1)
        // y' = (d*x + e*y + f) / (g*x + h*y + 1)
        var m = new SKMatrix
        {
            ScaleX = H[0],
            SkewX = H[1],
            TransX = H[2],
            SkewY = H[3],
            ScaleY = H[4],
            TransY = H[5],
            Persp0 = H[6],
            Persp1 = H[7],
            Persp2 = 1f
        };
        canvas.SetMatrix(m);
        canvas.DrawBitmap(src, 0, 0);
        canvas.Flush();
        return dst;
    }

    // ------------------------ Grayscale / Otsu / CCL ------------------------
    static SKBitmap ToGray(SKBitmap src)
    {
        var gray = new SKBitmap(src.Width, src.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            for (int y = 0; y < src.Height; y++)
            {
                var sp = (uint*)src.GetPixels() + y * src.Width; // BGRA premul
                var dp = (byte*)gray.GetPixels() + y * gray.RowBytes;
                for (int x = 0; x < src.Width; x++)
                {
                    uint c = sp[x];
                    byte b = (byte)(c & 0xFF);
                    byte g = (byte)((c >> 8) & 0xFF);
                    byte r = (byte)((c >> 16) & 0xFF);
                    byte a = (byte)((c >> 24) & 0xFF);
                    int add = 255 - a; // kompozice na bílou
                    int rOut = r + add, gOut = g + add, bOut = b + add;
                    int v = (int)Math.Round(0.299 * rOut + 0.587 * gOut + 0.114 * bOut);
                    dp[x] = (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
        return gray;
    }

    static byte Otsu(byte[] hist, int total)
    {
        long sum = 0; for (int t = 0; t < 256; t++) sum += t * (long)hist[t];
        long sumB = 0; int wB = 0; int wF = 0;
        double maxVar = -1; int threshold = 0;
        for (int t = 0; t < 256; t++)
        {
            wB += hist[t]; if (wB == 0) continue;
            wF = total - wB; if (wF == 0) break;
            sumB += t * (long)hist[t];
            double mB = (double)sumB / wB;
            double mF = (double)(sum - sumB) / wF;
            double varBetween = wB * wF * (mB - mF) * (mB - mF);
            if (varBetween > maxVar) { maxVar = varBetween; threshold = t; }
        }
        return (byte)threshold;
    }

    struct Comp { public int MinX, MinY, MaxX, MaxY, Count; public double Cx, Cy; }
    static List<Comp> ConnectedComponents(SKBitmap bin)
    {
        int w = bin.Width, h = bin.Height;
        var visited = new bool[w * h];
        var comps = new List<Comp>(256);
        var q = new Queue<(int x, int y)>();

        unsafe
        {
            byte* basePtr = (byte*)bin.GetPixels();
            int stride = bin.RowBytes;

            for (int y = 0; y < h; y++)
            {
                byte* row = basePtr + y * stride;
                for (int x = 0; x < w; x++)
                {
                    if (row[x] == 0) continue;
                    int id = y * w + x;
                    if (visited[id]) continue;

                    visited[id] = true;
                    q.Enqueue((x, y));

                    int minx = x, miny = y, maxx = x, maxy = y, cnt = 0;
                    long sumx = 0, sumy = 0;

                    while (q.Count > 0)
                    {
                        var (qx, qy) = q.Dequeue();
                        cnt++; sumx += qx; sumy += qy;
                        if (qx < minx) minx = qx; if (qy < miny) miny = qy;
                        if (qx > maxx) maxx = qx; if (qy > maxy) maxy = qy;

                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (dx == 0 && dy == 0) continue;
                                int nx = qx + dx, ny = qy + dy;
                                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                                int nid = ny * w + nx;
                                if (visited[nid]) continue;
                                byte* r = basePtr + ny * stride;
                                if (r[nx] == 0) continue;
                                visited[nid] = true; q.Enqueue((nx, ny));
                            }
                    }

                    comps.Add(new Comp
                    {
                        MinX = minx,
                        MinY = miny,
                        MaxX = maxx,
                        MaxY = maxy,
                        Count = cnt,
                        Cx = (double)sumx / cnt,
                        Cy = (double)sumy / cnt
                    });
                }
            }
        }
        return comps;
    }

    static (SKBitmap bin, bool inverted) BinarizeObjectsWhite(SKBitmap gray)
    {
        var hist = new byte[256];
        int total = gray.Width * gray.Height;
        unsafe
        {
            for (int y = 0; y < gray.Height; y++)
            {
                var p = (byte*)gray.GetPixels() + y * gray.RowBytes;
                for (int x = 0; x < gray.Width; x++) hist[p[x]]++;
            }
        }
        byte thr = Otsu(hist, total);

        var a = new SKBitmap(gray.Width, gray.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        var b = new SKBitmap(gray.Width, gray.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            for (int y = 0; y < gray.Height; y++)
            {
                var gp = (byte*)gray.GetPixels() + y * gray.RowBytes;
                var ap = (byte*)a.GetPixels() + y * a.RowBytes;
                var bp = (byte*)b.GetPixels() + y * b.RowBytes;
                for (int x = 0; x < gray.Width; x++)
                {
                    byte v = gp[x];
                    ap[x] = (byte)(v > thr ? 255 : 0);
                    bp[x] = (byte)(v > thr ? 0 : 255);
                }
            }
        }
        double ma = MeanWhite(a), mb = MeanWhite(b);
        bool inverted = ma < mb;
        return (inverted ? a : b, inverted);
    }
    static double MeanWhite(SKBitmap bin)
    {
        long sum = 0;
        unsafe
        {
            for (int y = 0; y < bin.Height; y++)
            {
                var p = (byte*)bin.GetPixels() + y * bin.RowBytes;
                for (int x = 0; x < bin.Width; x++) sum += p[x];
            }
        }
        return (sum / 255.0) / (bin.Width * (double)bin.Height);
    }

    static List<SKPoint> OrderByCorners(List<SKPoint> pts, int W, int H)
    {
        var corners = new[] { new SKPoint(0, 0), new SKPoint(W - 1, 0), new SKPoint(W - 1, H - 1), new SKPoint(0, H - 1) };
        var outPts = new List<SKPoint>(4);
        var P = new List<SKPoint>(pts);
        for (int i = 0; i < 4 && P.Count > 0; i++)
        {
            int bi = 0; double bd2 = double.MaxValue;
            for (int j = 0; j < P.Count; j++)
            {
                var d2 = (P[j].X - corners[i].X) * (P[j].X - corners[i].X) + (P[j].Y - corners[i].Y) * (P[j].Y - corners[i].Y);
                if (d2 < bd2) { bd2 = d2; bi = j; }
            }
            outPts.Add(P[bi]); P.RemoveAt(bi);
        }
        return outPts;
    }

    // ------------------------ Fill ratio & threshold & groups ------------------------
    static SKRect ToRect(SKRectI r) => new SKRect(r.Left, r.Top, r.Right, r.Bottom);

    static float FillRatioFromRoi(SKBitmap warpedBgr, SKRectI r, float padFrac, float openFrac)
    {
        int x = r.Left, y = r.Top, w = r.Width, h = r.Height;
        if (w <= 0 || h <= 0) return 0f;

        using var roi = new SKBitmap(w, h, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            for (int j = 0; j < h; j++)
            {
                var sp = (uint*)warpedBgr.GetPixels() + (y + j) * warpedBgr.Width + x;
                var dp = (byte*)roi.GetPixels() + j * roi.RowBytes;
                for (int i = 0; i < w; i++)
                {
                    uint c = sp[i];
                    byte bb = (byte)(c & 0xFF);
                    byte gg = (byte)((c >> 8) & 0xFF);
                    byte rr = (byte)((c >> 16) & 0xFF);
                    dp[i] = (byte)Math.Clamp((0.299 * rr + 0.587 * gg + 0.114 * bb), 0, 255);
                }
            }
        }

        int pad = Math.Max(1, (int)Math.Round(Math.Min(w, h) * padFrac));
        SKRectI inner = (w - 2 * pad >= 5 && h - 2 * pad >= 5)
            ? new SKRectI(pad, pad, w - pad, h - pad)
            : new SKRectI(0, 0, w, h);

        var hist = new byte[256];
        int total = inner.Width * inner.Height;
        unsafe
        {
            for (int j = inner.Top; j < inner.Bottom; j++)
            {
                var p = (byte*)roi.GetPixels() + j * roi.RowBytes;
                for (int i = inner.Left; i < inner.Right; i++) hist[p[i]]++;
            }
        }
        byte thr = Otsu(hist, total);

        int k = Math.Max(1, (int)Math.Round(Math.Min(inner.Width, inner.Height) * openFrac));
        if (k % 2 == 0) k++;

        int ones = 0;
        unsafe
        {
            var tmp = new byte[inner.Width * inner.Height];
            int idx = 0;
            for (int j = inner.Top; j < inner.Bottom; j++)
            {
                var p = (byte*)roi.GetPixels() + j * roi.RowBytes;
                for (int i = inner.Left; i < inner.Right; i++)
                    tmp[idx++] = (byte)(p[i] <= thr ? 1 : 0);
            }
            var er = MorphErode(tmp, inner.Width, inner.Height, k);
            var op = MorphDilate(er, inner.Width, inner.Height, k);
            for (int t = 0; t < op.Length; t++) ones += op[t];
        }
        return (float)ones / (inner.Width * inner.Height);
    }

    static byte[] MorphErode(byte[] src, int w, int h, int k)
    {
        int r = k / 2;
        var dst = new byte[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                bool ok = true;
                for (int dy = -r; dy <= r && ok; dy++)
                {
                    int yy = y + dy; if (yy < 0 || yy >= h) { ok = false; break; }
                    int row = yy * w;
                    for (int dx = -r; dx <= r; dx++)
                    {
                        int xx = x + dx; if (xx < 0 || xx >= w) { ok = false; break; }
                        if (src[row + xx] == 0) { ok = false; break; }
                    }
                }
                dst[y * w + x] = (byte)(ok ? 1 : 0);
            }
        return dst;
    }
    static byte[] MorphDilate(byte[] src, int w, int h, int k)
    {
        int r = k / 2;
        var dst = new byte[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                bool any = false;
                for (int dy = -r; dy <= r && !any; dy++)
                {
                    int yy = y + dy; if (yy < 0 || yy >= h) continue;
                    int row = yy * w;
                    for (int dx = -r; dx <= r; dx++)
                    {
                        int xx = x + dx; if (xx < 0 || xx >= w) continue;
                        if (src[row + xx] != 0) { any = true; break; }
                    }
                }
                dst[y * w + x] = (byte)(any ? 1 : 0);
            }
        return dst;
    }

    static float AutoThresholdKMeans(float[] values, float tmin, float tmax, float fallback, float minGap, int iters = 20)
    {
        if (values.Length == 0) return fallback;
        var v = values.Select(x => Math.Clamp(x, 0f, 1f)).OrderBy(x => x).ToArray();
        float c0 = Percentile(v, 20f), c1 = Percentile(v, 80f);
        for (int i = 0; i < iters; i++)
        {
            float mid = 0.5f * (c0 + c1);
            var m0 = v.Where(x => x < mid).ToArray();
            var m1 = v.Where(x => x >= mid).ToArray();
            if (m0.Length == 0 || m1.Length == 0) break;
            float c0n = (float)m0.Average();
            float c1n = (float)m1.Average();
            if (Math.Abs(c0n - c0) < 1e-4 && Math.Abs(c1n - c1) < 1e-4) { c0 = c0n; c1 = c1n; break; }
            c0 = c0n; c1 = c1n;
        }
        if (c0 > c1) (c0, c1) = (c1, c0);
        float thr = (c0 + c1) * 0.5f;
        if ((c1 - c0) < minGap) thr = fallback;
        return Math.Clamp(thr, tmin, tmax);
    }
    static float Percentile(float[] a, float p)
    {
        if (a.Length == 0) return 0f;
        float pos = (p / 100f) * (a.Length - 1);
        int lo = (int)Math.Floor(pos), hi = (int)Math.Ceiling(pos);
        if (lo == hi) return a[lo];
        return a[lo] + (a[hi] - a[lo]) * (pos - lo);
    }

    sealed class Group
    {
        public int[] Indices = new int[6];
        public int ChosenSlot = -1;
        public int ValueOf(int slot) => slot;
        public int ScoreContribution(float thr, float[] pList)
        {
            if (ChosenSlot < 0) return 0;
            int idx = Indices[ChosenSlot];
            return pList[idx] >= thr ? ValueOf(ChosenSlot) : 0;
        }
    }
    sealed class Item { public float Cx, Cy, W, H, P; public int Index; }

    static List<Group> BuildGroupsGrid3x2(List<SKRectI> rects, float[] pList)
    {
        var items = rects.Select((r, i) => new Item
        {
            Cx = r.Left + r.Width * 0.5f,
            Cy = r.Top + r.Height * 0.5f,
            W = r.Width,
            H = r.Height,
            P = pList[i],
            Index = i
        }).OrderBy(z => z.Cy).ToList();

        float hmed = items.Select(z => z.H).OrderBy(x => x).ElementAt(items.Count / 2);
        float rowThr = 0.6f * hmed;

        var rows = new List<List<Item>>();
        foreach (var it in items)
        {
            if (rows.Count == 0) rows.Add(new List<Item> { it });
            else
            {
                var last = rows.Last();
                var cyMed = last.Select(z => z.Cy).OrderBy(x => x).ElementAt(last.Count / 2);
                if (Math.Abs(it.Cy - cyMed) <= rowThr) last.Add(it);
                else rows.Add(new List<Item> { it });
            }
        }
        foreach (var r in rows) r.Sort((a, b) => a.Cx.CompareTo(b.Cx));

        var groups = new List<Group>();
        for (int i = 0; i + 1 < rows.Count; i += 2)
        {
            var top = rows[i]; var bot = rows[i + 1];
            int nt = top.Count / 3, nb = bot.Count / 3, n = Math.Min(nt, nb);
            for (int j = 0; j < n; j++)
            {
                var t3 = top.Skip(j * 3).Take(3).ToArray();
                var b3 = bot.Skip(j * 3).Take(3).ToArray();

                var g = new Group { Indices = new[] { t3[0].Index, t3[1].Index, t3[2].Index, b3[0].Index, b3[1].Index, b3[2].Index } };

                float bestP = -1f; int bestSlot = -1;
                for (int s = 0; s < 6; s++)
                {
                    int idx = g.Indices[s];
                    float p = pList[idx];
                    if (p > bestP) { bestP = p; bestSlot = s; }
                }
                g.ChosenSlot = bestSlot;
                groups.Add(g);
            }
        }
        return groups;
    }

    static SKBitmap MakeWarpedOverlay(
        SKBitmap warped,
        List<SKRectI> rects,
        float[] pList,
        IList<int> winnerIndices,
        List<Group> groups)
    {
        // Připrav mapu slotu 0..5 pro každý index (kvůli číslici v boxu)
        var slotByIndex = Enumerable.Repeat(-1, rects.Count).ToArray();
        for (int gi = 0; gi < groups.Count; gi++)
            for (int s = 0; s < 6; s++)
                slotByIndex[groups[gi].Indices[s]] = s;

        var winners = new HashSet<int>(winnerIndices);

        const int tablesPerRowBlock = 2;
        const int rowsPerTable = 5;
        const int colsPerTable = 3;
        const int totalTables = tablesPerRowBlock * 2;
        const int groupsPerRowPair = colsPerTable * tablesPerRowBlock; // očekáváme 6 šestic v jednom řádku tabulek

        var rowSums = new int[totalTables, rowsPerTable];
        var colSums = new int[totalTables, colsPerTable];
        var tableTotals = new int[totalTables];

        var rowBounds = new SKRect[totalTables, rowsPerTable];
        var colBounds = new SKRect[totalTables, colsPerTable];
        var tableBounds = new SKRect[totalTables];
        var rowHas = new bool[totalTables, rowsPerTable];
        var colHas = new bool[totalTables, colsPerTable];
        var tableHas = new bool[totalTables];

        var groupWidths = new List<float>();
        var groupHeights = new List<float>();

        // ---------- Overlay na WARPED ----------
        var visWarp = warped.Copy();
        using (var c = new SKCanvas(visWarp))
        {
            var green = new SKPaint { Color = new SKColor(40, 200, 40), Style = SKPaintStyle.Stroke, StrokeWidth = 2f, IsAntialias = true };
            var red = new SKPaint { Color = new SKColor(230, 40, 40), Style = SKPaintStyle.Stroke, StrokeWidth = 2f, IsAntialias = true };
            var blue = new SKPaint { Color = new SKColor(70, 130, 240), Style = SKPaintStyle.Stroke, StrokeWidth = 2.5f, PathEffect = SKPathEffect.CreateDash(new float[] { 6, 6 }, 0), IsAntialias = true };

            // texty
            var txt = new SKPaint { Color = SKColors.Blue, TextSize = 30, IsAntialias = true, Typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Bold) };
            var shadow = new SKPaint { Color = new SKColor(0, 0, 0, 180), TextSize = 30, IsAntialias = true, Typeface = txt.Typeface };

            // 2.1 rámečky šestic
            for (int gi = 0; gi < groups.Count; gi++)
            {
                var g = groups[gi];
                // union 6ti rectů
                var rs = g.Indices.Select(i => rects[i]).ToArray();
                int minX = rs.Min(r => r.Left), minY = rs.Min(r => r.Top);
                int maxX = rs.Max(r => r.Right), maxY = rs.Max(r => r.Bottom);
                var pad = 3;
                var groupRect = new SKRect(minX - pad, minY - pad, maxX + pad, maxY + pad);
                c.DrawRect(groupRect, blue);

                if (groupsPerRowPair <= 0) continue;

                int rowPairIndex = gi / groupsPerRowPair;
                int posInPair = gi % groupsPerRowPair;

                int tableRowBlock = rowPairIndex / rowsPerTable;
                int rowInTable = rowPairIndex % rowsPerTable;
                int tableColBlock = posInPair / colsPerTable;
                int colInTable = posInPair % colsPerTable;

                if (tableRowBlock >= 2 || tableColBlock >= tablesPerRowBlock)
                    continue;

                int tableIndex = tableRowBlock * tablesPerRowBlock + tableColBlock;
                if (tableIndex < 0 || tableIndex >= totalTables)
                    continue;

                groupWidths.Add(groupRect.Width);
                groupHeights.Add(groupRect.Height);

                if (!rowHas[tableIndex, rowInTable])
                {
                    rowBounds[tableIndex, rowInTable] = groupRect;
                    rowHas[tableIndex, rowInTable] = true;
                }
                else
                    rowBounds[tableIndex, rowInTable] = SKRect.Union(rowBounds[tableIndex, rowInTable], groupRect);

                if (!colHas[tableIndex, colInTable])
                {
                    colBounds[tableIndex, colInTable] = groupRect;
                    colHas[tableIndex, colInTable] = true;
                }
                else
                    colBounds[tableIndex, colInTable] = SKRect.Union(colBounds[tableIndex, colInTable], groupRect);

                if (!tableHas[tableIndex])
                {
                    tableBounds[tableIndex] = groupRect;
                    tableHas[tableIndex] = true;
                }
                else
                    tableBounds[tableIndex] = SKRect.Union(tableBounds[tableIndex], groupRect);

                bool hasWinner = false;
                int groupScore = 0;
                for (int s = 0; s < g.Indices.Length; s++)
                {
                    int idx = g.Indices[s];
                    if (winners.Contains(idx))
                    {
                        hasWinner = true;
                        int slot = slotByIndex[idx];
                        if (slot >= 0)
                            groupScore = g.ValueOf(slot);
                        break;
                    }
                }

                if (hasWinner)
                {
                    rowSums[tableIndex, rowInTable] += groupScore;
                    colSums[tableIndex, colInTable] += groupScore;
                    tableTotals[tableIndex] += groupScore;
                }
            }

            // 2.2 boxy + popisky
            for (int i = 0; i < rects.Count; i++)
            {
                var r = rects[i];
                bool isWin = winners.Contains(i);
                c.DrawRect(new SKRect(r.Left, r.Top, r.Right, r.Bottom), isWin ? green : red);

                // popisky: index, % fill, slot 0..5
                string idxLabel = $"#{i}";
                string fillLabel = $"{Math.Round(pList[i] * 100)}%";
                // ukazovat jen kdyz {Math.Round(pList[i] * 100)} je vic nez 50%?
                bool showFill = Math.Round(pList[i] * 100) > 40;


                string slotLabel = slotByIndex[i] >= 0 ? slotByIndex[i].ToString() : "?";

                // umístění: index vlevo nahoře, % vpravo dole, slot doprostřed
                if (showFill)
                {
                    DrawText(c, idxLabel, r.Left - 35, r.Top - 5, txt, shadow);
                    DrawText(c, fillLabel, r.Right + 35 - txt.MeasureText(fillLabel), r.Bottom + 10, txt, shadow);
                }
                // slot center
                float cx = r.Left + r.Width * 0.5f;
                float cy = r.Top + r.Height * 0.55f;
                var slotPaint = new SKPaint { Color = isWin ? green.Color : red.Color, TextSize = 40, IsAntialias = true, Typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Normal) };
                var slotShadow = new SKPaint { Color = new SKColor(0, 0, 0, 200), TextSize = 40, IsAntialias = true, Typeface = slotPaint.Typeface };
                float sw = slotPaint.MeasureText(slotLabel);
                DrawText(c, slotLabel, cx - sw / 2, cy, slotPaint, slotShadow);
            }

            if (groupWidths.Count > 0 && groupHeights.Count > 0)
            {
                float avgWidth = groupWidths.Average();
                float avgHeight = groupHeights.Average();
                float rowOffset = Math.Max(40f, avgWidth * 0.55f);
                float colOffset = Math.Max(35f, avgHeight * 0.65f);

                using var sumPaint = new SKPaint
                {
                    Color = new SKColor(255, 230, 90),
                    TextSize = Math.Max(36f, avgHeight * 0.65f),
                    IsAntialias = true,
                    Typeface = txt.Typeface
                };
                using var sumShadow = new SKPaint
                {
                    Color = new SKColor(0, 0, 0, 200),
                    TextSize = sumPaint.TextSize,
                    IsAntialias = true,
                    Typeface = txt.Typeface
                };
                using var totalPaint = new SKPaint
                {
                    Color = new SKColor(255, 255, 255),
                    TextSize = sumPaint.TextSize * 1.15f,
                    IsAntialias = true,
                    Typeface = txt.Typeface
                };
                using var totalShadow = new SKPaint
                {
                    Color = new SKColor(0, 0, 0, 220),
                    TextSize = totalPaint.TextSize,
                    IsAntialias = true,
                    Typeface = txt.Typeface
                };

                for (int table = 0; table < totalTables; table++)
                {
                    if (!tableHas[table]) continue;

                    for (int r = 0; r < rowsPerTable; r++)
                    {
                        if (!rowHas[table, r]) continue;
                        int value = rowSums[table, r];
                        string text = value.ToString();
                        var bounds = rowBounds[table, r];
                        float x = bounds.Right + rowOffset;
                        float y = bounds.MidY + sumPaint.TextSize * 0.35f;
                        DrawText(c, text, x, y, sumPaint, sumShadow);
                    }

                    for (int col = 0; col < colsPerTable; col++)
                    {
                        if (!colHas[table, col]) continue;
                        int value = colSums[table, col];
                        string text = value.ToString();
                        var bounds = colBounds[table, col];
                        float textWidth = sumPaint.MeasureText(text);
                        float x = bounds.MidX - textWidth * 0.5f;
                        float y = tableBounds[table].Bottom + colOffset + sumPaint.TextSize * 0.35f;
                        DrawText(c, text, x, y, sumPaint, sumShadow);
                    }

                    int total = tableTotals[table];
                    string totalText = total.ToString();
                    float totalX = tableBounds[table].Right + rowOffset;
                    float totalY = tableBounds[table].Bottom + colOffset + totalPaint.TextSize * 0.35f;
                    DrawText(c, totalText, totalX, totalY, totalPaint, totalShadow);
                }
            }
        }
        return visWarp;
    }



    static int TotalScoreFromItems(List<SKRectI> rects, float[] pList, float thr)
    {
        var groups = BuildGroupsGrid3x2(rects, pList);
        int total = 0;
        foreach (var g in groups) total += g.ScoreContribution(thr, pList);
        return total;
    }

    // ------------------------ Small utils ------------------------
    //static SKRect ToRect(SKRectI r) => new SKRect(r.Left, r.Top, r.Right, r.Bottom);
    static void DrawText(SKCanvas canvas, string s, float x, float y, SKPaint paint, SKPaint shadow)
    { canvas.DrawText(s, x + 1, y + 1, shadow); canvas.DrawText(s, x, y, paint); }
}

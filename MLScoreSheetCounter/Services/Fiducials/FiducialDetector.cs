using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Maui.Storage;
using MLScoreSheetCounter;
using SkiaSharp;

namespace YourApp.Services;

internal static class FiducialDetector
{
    public static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) DetectSourceFiducials(SKBitmap photo, string yoloOnnxLogical)
    {
        using var detector = new OnnxYoloFidDetector(yoloOnnxLogical, imgsz: 1024);
        System.Diagnostics.Debug.WriteLine(OnnxYoloFidDetector.DebugListOutputs(detector._session));
        var rawDetections = detector.Detect(photo);

        if (rawDetections.Count < 3)
        {
            throw new InvalidOperationException($"Nedostatek fiducialů na fotce: {rawDetections.Count}");
        }

        var deduplicated = DeduplicateBoxes(rawDetections, iouThr: 0.6f);
        var candidates = deduplicated.Select(d => (d.Cx, d.Cy, d.Conf)).ToList();

        if (candidates.Count >= 4)
        {
            return AssignByGeometry(candidates, photo.Width, photo.Height);
        }

        var approximated = OrderByCorners(
            candidates.Select(p => new SKPoint((float)p.Cx, (float)p.Cy)).ToList(),
            photo.Width,
            photo.Height
        );

        var known = new Dictionary<string, (float x, float y, float conf)>();
        string[] names = { "TL", "TR", "BR", "BL" };
        for (int i = 0; i < candidates.Count && i < 4; i++)
        {
            known[names[i]] = (approximated[i].X, approximated[i].Y, (float)candidates[i].Conf);
        }

        var missing = CloseParallelogram(known);
        if (missing.name != null)
        {
            known[missing.name] = (missing.x, missing.y, 0f);
        }

        return (
            new SKPoint(known["TL"].x, known["TL"].y),
            new SKPoint(known["TR"].x, known["TR"].y),
            new SKPoint(known["BR"].x, known["BR"].y),
            new SKPoint(known["BL"].x, known["BL"].y)
        );
    }

    public static async Task<(SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL)> DetectDestinationFiducialsAsync(
        string fidPngLogicalName,
        int templateWidth,
        int templateHeight)
    {
        try
        {
            using var stream = await FileSystem.OpenAppPackageFileAsync(fidPngLogicalName);
            using var bitmap = SKBitmap.Decode(stream);
            if (bitmap == null)
            {
                throw new FileNotFoundException(fidPngLogicalName);
            }

            return DetectFiducialsCentersInTemplate(bitmap);
        }
        catch
        {
            return (
                new SKPoint(0, 0),
                new SKPoint(templateWidth - 1, 0),
                new SKPoint(templateWidth - 1, templateHeight - 1),
                new SKPoint(0, templateHeight - 1)
            );
        }
    }

    private static List<OnnxYoloFidDetector.Det> DeduplicateBoxes(List<OnnxYoloFidDetector.Det> detections, float iouThr)
    {
        var sorted = detections.OrderByDescending(d => d.Conf).ToList();
        var keep = new List<OnnxYoloFidDetector.Det>();
        foreach (var detection in sorted)
        {
            bool duplicate = keep.Any(existing => IoUxywh(detection, existing) > iouThr);
            if (!duplicate)
            {
                keep.Add(detection);
            }
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
        float intersection = iw * ih;
        float union = (ax2 - ax1) * (ay1 - ay2) + (bx2 - bx1) * (by1 - by2) - intersection;
        union = Math.Abs(union);
        return union <= 0 ? 0 : intersection / union;
    }

    private static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) AssignByGeometry(
        List<(float x, float y, float conf)> candidates,
        int width,
        int height)
    {
        var targets = new[] { new SKPoint(0, 0), new SKPoint(width, 0), new SKPoint(width, height), new SKPoint(0, height) };
        double diagonal = Math.Sqrt((double)width * width + (double)height * height);
        var sample = candidates.Take(Math.Min(10, candidates.Count)).ToList();

        double bestCost = double.MaxValue;
        (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) best = default;

        var indexes = Enumerable.Range(0, sample.Count).ToArray();
        foreach (var combination in Combinations(indexes, 4))
        {
            var subset = combination.Select(i => sample[i]).ToArray();
            for (int p0 = 0; p0 < 4; p0++)
            {
                for (int p1 = 0; p1 < 4; p1++)
                {
                    if (p1 == p0) continue;
                    for (int p2 = 0; p2 < 4; p2++)
                    {
                        if (p2 == p0 || p2 == p1) continue;
                        int p3 = 6 - p0 - p1 - p2;
                        double cost = 0;
                        double confSum = 0;
                        var s0 = new SKPoint(subset[p0].x, subset[p0].y);
                        var s1 = new SKPoint(subset[p1].x, subset[p1].y);
                        var s2 = new SKPoint(subset[p2].x, subset[p2].y);
                        var s3 = new SKPoint(subset[p3].x, subset[p3].y);
                        var points = new[] { s0, s1, s2, s3 };
                        for (int k = 0; k < 4; k++)
                        {
                            cost += Distance(points[k], targets[k]) / diagonal;
                            confSum += subset[(k == 0 ? p0 : k == 1 ? p1 : k == 2 ? p2 : p3)].conf;
                        }

                        cost -= 0.05 * (confSum / 4.0);
                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            best = (s0, s1, s2, s3);
                        }
                    }
                }
            }
        }

        if (bestCost == double.MaxValue)
        {
            throw new InvalidOperationException("Nelze přiřadit 4 body.");
        }

        return best;
    }

    private static IEnumerable<int[]> Combinations(int[] source, int choose)
    {
        int n = source.Length;
        var indexes = Enumerable.Range(0, choose).ToArray();
        while (true)
        {
            yield return indexes.Select(i => source[i]).ToArray();
            int i;
            for (i = choose - 1; i >= 0; i--)
            {
                if (indexes[i] != i + n - choose) break;
            }

            if (i < 0) yield break;

            indexes[i]++;
            for (int j = i + 1; j < choose; j++)
            {
                indexes[j] = indexes[j - 1] + 1;
            }
        }
    }

    private static double Distance(SKPoint a, SKPoint b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    private static (string name, float x, float y) CloseParallelogram(Dictionary<string, (float x, float y, float conf)> known)
    {
        var names = new[] { "TL", "TR", "BR", "BL" };
        var missing = names.FirstOrDefault(n => !known.ContainsKey(n));
        if (missing == null)
        {
            return (null!, 0, 0);
        }

        var values = known.ToDictionary(kv => kv.Key, kv => new SKPoint(kv.Value.x, kv.Value.y));
        SKPoint point;
        switch (missing)
        {
            case "TL":
                if (values.ContainsKey("TR") && values.ContainsKey("BR") && values.ContainsKey("BL"))
                {
                    point = new SKPoint(values["TR"].X + (values["BL"].X - values["BR"].X), values["TR"].Y + (values["BL"].Y - values["BR"].Y));
                }
                else
                {
                    point = Average(values.Values);
                }
                break;
            case "TR":
                if (values.ContainsKey("TL") && values.ContainsKey("BR") && values.ContainsKey("BL"))
                {
                    point = new SKPoint(values["TL"].X + (values["BR"].X - values["BL"].X), values["TL"].Y + (values["BR"].Y - values["BL"].Y));
                }
                else
                {
                    point = Average(values.Values);
                }
                break;
            case "BR":
                if (values.ContainsKey("TL") && values.ContainsKey("TR") && values.ContainsKey("BL"))
                {
                    point = new SKPoint(values["BL"].X + (values["TR"].X - values["TL"].X), values["BL"].Y + (values["TR"].Y - values["TL"].Y));
                }
                else
                {
                    point = Average(values.Values);
                }
                break;
            default:
                if (values.ContainsKey("TL") && values.ContainsKey("TR") && values.ContainsKey("BR"))
                {
                    point = new SKPoint(values["TL"].X + (values["BR"].X - values["TR"].X), values["TL"].Y + (values["BR"].Y - values["TR"].Y));
                }
                else
                {
                    point = Average(values.Values);
                }
                break;
        }

        return (missing, point.X, point.Y);
    }

    private static SKPoint Average(IEnumerable<SKPoint> points)
    {
        float sumX = 0;
        float sumY = 0;
        int count = 0;
        foreach (var point in points)
        {
            sumX += point.X;
            sumY += point.Y;
            count++;
        }

        return new SKPoint(sumX / count, sumY / count);
    }

    private static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) DetectFiducialsCentersInTemplate(SKBitmap source)
    {
        using var gray = BitmapPreprocessor.ToGray(source);
        var (binary, _) = BinarizeObjectsWhite(gray);
        var components = ConnectedComponents(binary);
        var top = components.OrderByDescending(c => c.Count).Take(4).ToList();
        if (top.Count < 4)
        {
            throw new InvalidOperationException("Ve fiducial šabloně nejsou 4 markery.");
        }

        var points = top.Select(c => new SKPoint((float)c.Cx, (float)c.Cy)).ToList();
        var ordered = OrderByCorners(points, source.Width, source.Height);
        return (ordered[0], ordered[1], ordered[2], ordered[3]);
    }

    private static (SKBitmap bin, bool inverted) BinarizeObjectsWhite(SKBitmap gray)
    {
        var histogram = new byte[256];
        int total = gray.Width * gray.Height;
        unsafe
        {
            for (int y = 0; y < gray.Height; y++)
            {
                var row = (byte*)gray.GetPixels() + y * gray.RowBytes;
                for (int x = 0; x < gray.Width; x++)
                {
                    histogram[row[x]]++;
                }
            }
        }

        byte threshold = Otsu(histogram, total);

        var a = new SKBitmap(gray.Width, gray.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        var b = new SKBitmap(gray.Width, gray.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            for (int y = 0; y < gray.Height; y++)
            {
                var srcRow = (byte*)gray.GetPixels() + y * gray.RowBytes;
                var dstRowA = (byte*)a.GetPixels() + y * a.RowBytes;
                var dstRowB = (byte*)b.GetPixels() + y * b.RowBytes;
                for (int x = 0; x < gray.Width; x++)
                {
                    byte v = srcRow[x];
                    dstRowA[x] = (byte)(v > threshold ? 255 : 0);
                    dstRowB[x] = (byte)(v <= threshold ? 255 : 0);
                }
            }
        }

        double ratioA = WhiteRatio(a);
        double ratioB = WhiteRatio(b);
        bool inverted = ratioB > ratioA;
        return inverted ? (b, true) : (a, false);
    }

    private static double WhiteRatio(SKBitmap bitmap)
    {
        long sum = 0;
        unsafe
        {
            for (int y = 0; y < bitmap.Height; y++)
            {
                var row = (byte*)bitmap.GetPixels() + y * bitmap.RowBytes;
                for (int x = 0; x < bitmap.Width; x++)
                {
                    sum += row[x];
                }
            }
        }

        return (sum / 255.0) / (bitmap.Width * (double)bitmap.Height);
    }

    private static byte Otsu(byte[] hist, int total)
    {
        long sum = 0;
        for (int t = 0; t < 256; t++)
        {
            sum += t * (long)hist[t];
        }

        long sumB = 0;
        int wB = 0;
        double maxVar = -1;
        int threshold = 0;
        for (int t = 0; t < 256; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            int wF = total - wB;
            if (wF == 0) break;
            sumB += t * (long)hist[t];
            double mB = (double)sumB / wB;
            double mF = (double)(sum - sumB) / wF;
            double varBetween = wB * wF * (mB - mF) * (mB - mF);
            if (varBetween > maxVar)
            {
                maxVar = varBetween;
                threshold = t;
            }
        }

        return (byte)threshold;
    }

    private struct Component
    {
        public int MinX;
        public int MinY;
        public int MaxX;
        public int MaxY;
        public int Count;
        public double Cx;
        public double Cy;
    }

    private static List<Component> ConnectedComponents(SKBitmap bin)
    {
        int w = bin.Width;
        int h = bin.Height;
        var visited = new bool[w * h];
        var components = new List<Component>(256);
        var queue = new Queue<(int x, int y)>();

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
                    queue.Enqueue((x, y));

                    int minx = x, miny = y, maxx = x, maxy = y, count = 0;
                    long sumx = 0, sumy = 0;

                    while (queue.Count > 0)
                    {
                        var (qx, qy) = queue.Dequeue();
                        count++;
                        sumx += qx;
                        sumy += qy;
                        if (qx < minx) minx = qx;
                        if (qy < miny) miny = qy;
                        if (qx > maxx) maxx = qx;
                        if (qy > maxy) maxy = qy;

                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (dx == 0 && dy == 0) continue;
                                int nx = qx + dx;
                                int ny = qy + dy;
                                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                                int nid = ny * w + nx;
                                if (visited[nid]) continue;
                                byte* neighbourRow = basePtr + ny * stride;
                                if (neighbourRow[nx] == 0) continue;
                                visited[nid] = true;
                                queue.Enqueue((nx, ny));
                            }
                        }
                    }

                    components.Add(new Component
                    {
                        MinX = minx,
                        MinY = miny,
                        MaxX = maxx,
                        MaxY = maxy,
                        Count = count,
                        Cx = (double)sumx / count,
                        Cy = (double)sumy / count
                    });
                }
            }
        }

        return components;
    }

    private static List<SKPoint> OrderByCorners(List<SKPoint> pts, int width, int height)
    {
        var corners = new[]
        {
            new SKPoint(0, 0),
            new SKPoint(width - 1, 0),
            new SKPoint(width - 1, height - 1),
            new SKPoint(0, height - 1)
        };

        var output = new List<SKPoint>(4);
        var points = new List<SKPoint>(pts);
        for (int i = 0; i < 4 && points.Count > 0; i++)
        {
            int bestIndex = 0;
            double bestDistance = double.MaxValue;
            for (int j = 0; j < points.Count; j++)
            {
                var distance = (points[j].X - corners[i].X) * (points[j].X - corners[i].X) +
                               (points[j].Y - corners[i].Y) * (points[j].Y - corners[i].Y);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestIndex = j;
                }
            }

            output.Add(points[bestIndex]);
            points.RemoveAt(bestIndex);
        }

        return output;
    }
}

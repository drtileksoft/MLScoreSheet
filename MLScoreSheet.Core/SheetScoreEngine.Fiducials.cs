using SkiaSharp;

namespace MLScoreSheet.Core;

public static partial class SheetScoreEngine
{
    private static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) DetectSrcFidsWithOnnx(
        SKBitmap photo,
        IResourceProvider resourceProvider,
        string yoloOnnxLogical)
    {
        using var detector = new OnnxYoloFidDetector(resourceProvider, yoloOnnxLogical, imgsz: 1024);
        System.Diagnostics.Debug.WriteLine(OnnxYoloFidDetector.DebugListOutputs(detector._session));
        var rawDetections = detector.Detect(photo);

        if (rawDetections.Count < 3)
            throw new InvalidOperationException($"Not enough fiducials in the photo: {rawDetections.Count}");

        var detections = DedupBoxes(rawDetections, iouThr: 0.6f);

        var points = detections.Select(d => (d.Cx, d.Cy, d.Conf)).ToList();
        if (points.Count >= 4)
        {
            var ordered = AssignByGeometry(points, photo.Width, photo.Height);
            return (ordered.TL, ordered.TR, ordered.BR, ordered.BL);
        }

        var approx = OrderByCorners(points
            .Select(p => new SKPoint((float)p.Cx, (float)p.Cy))
            .ToList(),
            photo.Width,
            photo.Height);

        var known = new Dictionary<string, (float x, float y, float conf)>();
        string[] names = { "TL", "TR", "BR", "BL" };
        for (int i = 0; i < points.Count && i < 4; i++)
        {
            known[names[i]] = ((float)approx[i].X, (float)approx[i].Y, (float)points[i].Conf);
        }

        var missing = CloseParallelogram(known);
        if (missing.name != null)
            known[missing.name] = (missing.x, missing.y, 0f);

        return (
            new SKPoint(known["TL"].x, known["TL"].y),
            new SKPoint(known["TR"].x, known["TR"].y),
            new SKPoint(known["BR"].x, known["BR"].y),
            new SKPoint(known["BL"].x, known["BL"].y));
    }

    private static List<OnnxYoloFidDetector.Det> DedupBoxes(List<OnnxYoloFidDetector.Det> detections, float iouThr)
    {
        var sorted = detections.OrderByDescending(d => d.Conf).ToList();
        var keep = new List<OnnxYoloFidDetector.Det>();
        foreach (var detection in sorted)
        {
            bool duplicate = keep.Any(existing => IoUxywh(detection, existing) > iouThr);
            if (!duplicate)
                keep.Add(detection);
        }
        return keep;
    }

    private static float IoUxywh(OnnxYoloFidDetector.Det a, OnnxYoloFidDetector.Det b)
    {
        (float ax1, float ay1, float ax2, float ay2) = (a.Cx - a.W / 2, a.Cy - a.H / 2, a.Cx + a.W / 2, a.Cy + a.H / 2);
        (float bx1, float by1, float bx2, float by2) = (b.Cx - b.W / 2, b.Cy - b.H / 2, b.Cx + b.W / 2, b.Cy + b.H / 2);
        float ix1 = Math.Max(ax1, bx1);
        float iy1 = Math.Max(ay1, by1);
        float ix2 = Math.Min(ax2, bx2);
        float iy2 = Math.Min(ay2, by2);
        float iw = Math.Max(0, ix2 - ix1);
        float ih = Math.Max(0, iy2 - iy1);
        float intersection = iw * ih;
        float unionArea = (ax2 - ax1) * (ay1 - ay2) + (bx2 - bx1) * (by1 - by2) - intersection;
        unionArea = Math.Abs(unionArea);
        return unionArea <= 0 ? 0 : intersection / unionArea;
    }

    private static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) AssignByGeometry(
        List<(float x, float y, float conf)> points,
        int width,
        int height)
    {
        var targets = new[]
        {
            new SKPoint(0, 0),
            new SKPoint(width, 0),
            new SKPoint(width, height),
            new SKPoint(0, height)
        };
        double diagonal = Math.Sqrt((double)width * width + (double)height * height);
        var candidates = points.Take(Math.Min(10, points.Count)).ToList();

        double bestCost = double.MaxValue;
        (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) best = default;

        var indices = Enumerable.Range(0, candidates.Count).ToArray();
        foreach (var combination in Combinations(indices, 4))
        {
            var subset = combination.Select(i => candidates[i]).ToArray();
            foreach (var permutation in PermutationsOfFour())
            {
                var assigned = permutation.Select(index => subset[index]).ToArray();
                double cost = 0;
                double confSum = 0;
                for (int k = 0; k < 4; k++)
                {
                    var source = new SKPoint(assigned[k].x, assigned[k].y);
                    cost += Distance(source, targets[k]) / diagonal;
                    confSum += assigned[k].conf;
                }
                cost -= 0.05 * (confSum / 4.0);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    best = (
                        new SKPoint(assigned[0].x, assigned[0].y),
                        new SKPoint(assigned[1].x, assigned[1].y),
                        new SKPoint(assigned[2].x, assigned[2].y),
                        new SKPoint(assigned[3].x, assigned[3].y));
                }
            }
        }

        if (bestCost == double.MaxValue)
            throw new InvalidOperationException("Unable to assign 4 points.");

        return best;

        static double Distance(SKPoint a, SKPoint b)
            => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

        static IEnumerable<int[]> PermutationsOfFour()
        {
            int[] values = { 0, 1, 2, 3 };
            foreach (var p0 in values)
            {
                foreach (var p1 in values)
                {
                    if (p1 == p0) continue;
                    foreach (var p2 in values)
                    {
                        if (p2 == p0 || p2 == p1) continue;
                        int p3 = 6 - p0 - p1 - p2;
                        yield return new[] { p0, p1, p2, p3 };
                    }
                }
            }
        }
    }

    private static IEnumerable<int[]> Combinations(int[] source, int choose)
    {
        int n = source.Length;
        var indices = Enumerable.Range(0, choose).ToArray();
        while (true)
        {
            yield return indices.Select(i => source[i]).ToArray();

            int i;
            for (i = choose - 1; i >= 0; i--)
            {
                if (indices[i] != i + n - choose) break;
            }

            if (i < 0)
                yield break;

            indices[i]++;
            for (int j = i + 1; j < choose; j++)
                indices[j] = indices[j - 1] + 1;
        }
    }

    private static (string name, float x, float y) CloseParallelogram(
        Dictionary<string, (float x, float y, float conf)> known)
    {
        var names = new[] { "TL", "TR", "BR", "BL" };
        var missing = names.FirstOrDefault(n => !known.ContainsKey(n));
        if (missing == null)
            return (null!, 0, 0);

        var points = known.ToDictionary(kv => kv.Key, kv => new SKPoint(kv.Value.x, kv.Value.y));
        SKPoint candidate = missing switch
        {
            "TL" when points.ContainsKey("TR") && points.ContainsKey("BR") && points.ContainsKey("BL")
                => new SKPoint(points["TR"].X + (points["BL"].X - points["BR"].X), points["TR"].Y + (points["BL"].Y - points["BR"].Y)),
            "TR" when points.ContainsKey("TL") && points.ContainsKey("BR") && points.ContainsKey("BL")
                => new SKPoint(points["TL"].X + (points["BR"].X - points["BL"].X), points["TL"].Y + (points["BR"].Y - points["BL"].Y)),
            "BR" when points.ContainsKey("TL") && points.ContainsKey("TR") && points.ContainsKey("BL")
                => new SKPoint(points["BL"].X + (points["TR"].X - points["TL"].X), points["BL"].Y + (points["TR"].Y - points["TL"].Y)),
            _ when points.ContainsKey("TL") && points.ContainsKey("TR") && points.ContainsKey("BR")
                => new SKPoint(points["TL"].X + (points["BR"].X - points["TR"].X), points["TL"].Y + (points["BR"].Y - points["TR"].Y)),
            _ => Average(points.Values)
        };

        return (missing, candidate.X, candidate.Y);

        static SKPoint Average(IEnumerable<SKPoint> values)
        {
            float sumX = 0;
            float sumY = 0;
            int count = 0;
            foreach (var point in values)
            {
                sumX += point.X;
                sumY += point.Y;
                count++;
            }
            return new SKPoint(sumX / count, sumY / count);
        }
    }

    private static async Task<(SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL)> TryDetectDstFidsOrCornersAsync(
        IResourceProvider resourceProvider,
        string fidPngLogicalName,
        int width,
        int height)
    {
        try
        {
            using var fid = await resourceProvider.OpenReadAsync(fidPngLogicalName);
            using var bitmap = SKBitmap.Decode(fid);
            if (bitmap == null)
                throw new FileNotFoundException(fidPngLogicalName);
            return DetectFiducialsCentersInTemplate(bitmap);
        }
        catch
        {
            return (
                new SKPoint(0, 0),
                new SKPoint(width - 1, 0),
                new SKPoint(width - 1, height - 1),
                new SKPoint(0, height - 1));
        }
    }

    private static (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) DetectFiducialsCentersInTemplate(SKBitmap src)
    {
        using var gray = ToGray(src);
        var (bin, _) = BinarizeObjectsWhite(gray);
        var components = ConnectedComponents(bin);
        var top = components.OrderByDescending(c => c.Count).Take(4).ToList();
        if (top.Count < 4)
            throw new InvalidOperationException("The fiducial template does not contain 4 markers.");

        var points = top.Select(c => new SKPoint((float)c.Cx, (float)c.Cy)).ToList();
        var ordered = OrderByCorners(points, src.Width, src.Height);
        return (ordered[0], ordered[1], ordered[2], ordered[3]);
    }

    private static List<SKPoint> OrderByCorners(List<SKPoint> points, int width, int height)
    {
        var corners = new[]
        {
            new SKPoint(0, 0),
            new SKPoint(width - 1, 0),
            new SKPoint(width - 1, height - 1),
            new SKPoint(0, height - 1)
        };
        var result = new List<SKPoint>(4);
        var pool = new List<SKPoint>(points);
        for (int i = 0; i < 4 && pool.Count > 0; i++)
        {
            int bestIndex = 0;
            double bestDistance = double.MaxValue;
            for (int j = 0; j < pool.Count; j++)
            {
                double distance = (pool[j].X - corners[i].X) * (pool[j].X - corners[i].X)
                                + (pool[j].Y - corners[i].Y) * (pool[j].Y - corners[i].Y);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestIndex = j;
                }
            }
            result.Add(pool[bestIndex]);
            pool.RemoveAt(bestIndex);
        }
        return result;
    }
}

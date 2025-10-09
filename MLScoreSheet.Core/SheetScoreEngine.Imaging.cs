using SkiaSharp;

namespace MLScoreSheet.Core;

public static partial class SheetScoreEngine
{
    private static SKBitmap DecodeLandscapePhoto(Stream photoStream)
    {
        var decoded = SKBitmap.Decode(photoStream) ?? throw new InvalidOperationException("Cannot decode photo.");

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

    private static double GetFillPercentLocalContrast(SKBitmap src, SKRectI rect)
    {
        const int margin = 3;
        var expanded = new SKRectI(
            Math.Max(0, rect.Left - margin),
            Math.Max(0, rect.Top - margin),
            Math.Min(src.Width, rect.Right + margin),
            Math.Min(src.Height, rect.Bottom + margin));

        if (expanded.Width <= 0 || expanded.Height <= 0)
            return 0;

        rect = expanded;

        int ix = Math.Max(1, (int)Math.Round(rect.Width * 0.12));
        int iy = Math.Max(1, (int)Math.Round(rect.Height * 0.12));
        var roiRect = new SKRectI(rect.Left + ix, rect.Top + iy, rect.Right - ix, rect.Bottom - iy);
        if (roiRect.Width <= 2 || roiRect.Height <= 2)
            return 0;

        using var roi = new SKBitmap(roiRect.Width, roiRect.Height, src.ColorType, src.AlphaType);
        using (var canvas = new SKCanvas(roi))
            canvas.DrawBitmap(src, roiRect, new SKRect(0, 0, roiRect.Width, roiRect.Height));

        using var pixmap = new SKPixmap(roi.Info, roi.GetPixels(out _));
        Span<int> hist = stackalloc int[256];
        hist.Clear();
        int total = roi.Width * roi.Height;

        unsafe
        {
            byte* basePtr = (byte*)pixmap.GetPixels();
            int bpp = pixmap.Info.BytesPerPixel;
            for (int y = 0; y < roi.Height; y++)
            {
                byte* row = basePtr + y * pixmap.RowBytes;
                for (int x = 0; x < roi.Width; x++)
                {
                    byte B = row[x * bpp + 0];
                    byte G = row[x * bpp + 1];
                    byte R = row[x * bpp + 2];
                    int luminance = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    int darkness = 255 - luminance;
                    hist[darkness]++;
                }
            }
        }

        int pLo = PercentileFromHist(hist, total, 0.05);
        int pHi = PercentileFromHist(hist, total, 0.95);
        if (pHi <= pLo)
        {
            pLo = Math.Max(0, pLo - 1);
            pHi = Math.Min(255, pLo + 1);
        }
        double range = pHi - pLo;

        Span<int> normalizedHist = stackalloc int[256];
        normalizedHist.Clear();

        unsafe
        {
            byte* basePtr = (byte*)pixmap.GetPixels();
            int bpp = pixmap.Info.BytesPerPixel;
            for (int y = 0; y < roi.Height; y++)
            {
                byte* row = basePtr + y * pixmap.RowBytes;
                for (int x = 0; x < roi.Width; x++)
                {
                    byte B = row[x * bpp + 0];
                    byte G = row[x * bpp + 1];
                    byte R = row[x * bpp + 2];
                    int luminance = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    int darkness = 255 - luminance;
                    double normalized = (darkness - pLo) / range;
                    if (normalized < 0) normalized = 0;
                    else if (normalized > 1) normalized = 1;
                    int bucket = (int)Math.Round(normalized * 255.0);
                    normalizedHist[bucket]++;
                }
            }
        }

        int threshold = OtsuThreshold(normalizedHist, total);

        int blackPixels = 0;
        unsafe
        {
            byte* basePtr = (byte*)pixmap.GetPixels();
            int bpp = pixmap.Info.BytesPerPixel;
            for (int y = 0; y < roi.Height; y++)
            {
                byte* row = basePtr + y * pixmap.RowBytes;
                for (int x = 0; x < roi.Width; x++)
                {
                    byte B = row[x * bpp + 0];
                    byte G = row[x * bpp + 1];
                    byte R = row[x * bpp + 2];
                    int luminance = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    int darkness = 255 - luminance;
                    double normalized = (darkness - pLo) / range;
                    if (normalized < 0) normalized = 0;
                    else if (normalized > 1) normalized = 1;
                    int bucket = (int)Math.Round(normalized * 255.0);
                    if (bucket >= threshold)
                        blackPixels++;
                }
            }
        }

        double percentage = 100.0 * blackPixels / Math.Max(1, total);
        if (double.IsNaN(percentage)) percentage = 0;
        if (percentage < 0) percentage = 0;
        else if (percentage > 100) percentage = 100;
        return percentage;

        static int PercentileFromHist(Span<int> histogram, int totalPixels, double percentile)
        {
            int target = (int)Math.Round(percentile * totalPixels);
            int cumulative = 0;
            for (int i = 0; i < 256; i++)
            {
                cumulative += histogram[i];
                if (cumulative >= target)
                    return i;
            }
            return 255;
        }

        static int OtsuThreshold(Span<int> histogram, int totalPixels)
        {
            double sum = 0;
            for (int t = 0; t < 256; t++)
                sum += t * histogram[t];

            double sumBackground = 0;
            int weightBackground = 0;
            double maxVariance = -1;
            int threshold = 128;

            for (int t = 0; t < 256; t++)
            {
                weightBackground += histogram[t];
                if (weightBackground == 0) continue;
                int weightForeground = totalPixels - weightBackground;
                if (weightForeground == 0) break;

                sumBackground += t * histogram[t];
                double meanBackground = sumBackground / weightBackground;
                double meanForeground = (sum - sumBackground) / weightForeground;
                double varianceBetween = weightBackground * weightForeground * (meanBackground - meanForeground) * (meanBackground - meanForeground);
                if (varianceBetween > maxVariance)
                {
                    maxVariance = varianceBetween;
                    threshold = t;
                }
            }

            return threshold;
        }
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
        int width = bin.Width;
        int height = bin.Height;
        var visited = new bool[width * height];
        var components = new List<Component>(256);
        var queue = new Queue<(int x, int y)>();

        unsafe
        {
            byte* basePtr = (byte*)bin.GetPixels();
            int stride = bin.RowBytes;

            for (int y = 0; y < height; y++)
            {
                byte* row = basePtr + y * stride;
                for (int x = 0; x < width; x++)
                {
                    if (row[x] == 0) continue;
                    int id = y * width + x;
                    if (visited[id]) continue;

                    visited[id] = true;
                    queue.Enqueue((x, y));

                    int minX = x, minY = y, maxX = x, maxY = y, count = 0;
                    long sumX = 0, sumY = 0;

                    while (queue.Count > 0)
                    {
                        var (qx, qy) = queue.Dequeue();
                        count++;
                        sumX += qx;
                        sumY += qy;
                        if (qx < minX) minX = qx;
                        if (qy < minY) minY = qy;
                        if (qx > maxX) maxX = qx;
                        if (qy > maxY) maxY = qy;

                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (dx == 0 && dy == 0) continue;
                                int nx = qx + dx;
                                int ny = qy + dy;
                                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                                int neighborId = ny * width + nx;
                                if (visited[neighborId]) continue;
                                byte* neighborRow = basePtr + ny * stride;
                                if (neighborRow[nx] == 0) continue;
                                visited[neighborId] = true;
                                queue.Enqueue((nx, ny));
                            }
                        }
                    }

                    components.Add(new Component
                    {
                        MinX = minX,
                        MinY = minY,
                        MaxX = maxX,
                        MaxY = maxY,
                        Count = count,
                        Cx = (double)sumX / count,
                        Cy = (double)sumY / count
                    });
                }
            }
        }

        return components;
    }

    private static (SKBitmap bin, bool inverted) BinarizeObjectsWhite(SKBitmap gray)
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

    private static double MeanWhite(SKBitmap bin)
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

    private static SKBitmap ToGray(SKBitmap src)
    {
        var gray = new SKBitmap(src.Width, src.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            for (int y = 0; y < src.Height; y++)
            {
                var sp = (uint*)src.GetPixels() + y * src.Width;
                var dp = (byte*)gray.GetPixels() + y * gray.RowBytes;
                for (int x = 0; x < src.Width; x++)
                {
                    uint c = sp[x];
                    byte b = (byte)(c & 0xFF);
                    byte g = (byte)((c >> 8) & 0xFF);
                    byte r = (byte)((c >> 16) & 0xFF);
                    byte a = (byte)((c >> 24) & 0xFF);
                    int add = 255 - a;
                    int rOut = r + add, gOut = g + add, bOut = b + add;
                    int v = (int)Math.Round(0.299 * rOut + 0.587 * gOut + 0.114 * bOut);
                    dp[x] = (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
        return gray;
    }

    private static byte Otsu(byte[] hist, int total)
    {
        long sum = 0;
        for (int t = 0; t < 256; t++) sum += t * (long)hist[t];
        long sumB = 0;
        int wB = 0;
        int wF = 0;
        double maxVar = -1;
        int threshold = 0;
        for (int t = 0; t < 256; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            wF = total - wB;
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
}

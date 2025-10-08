using System;
using System.Collections.Generic;
using SkiaSharp;

namespace YourApp.Services;

internal static class FillRatioCalculator
{
    public static float[] ComputeRatios(SKBitmap warped, IReadOnlyList<SKRectI> rects, float padFraction, float openFraction)
    {
        var result = new float[rects.Count];
        for (int i = 0; i < rects.Count; i++)
        {
            result[i] = FillRatioFromRoi(warped, rects[i], padFraction, openFraction);
        }

        return result;
    }

    public static float[] ComputeLocalContrastRatios(SKBitmap warped, IReadOnlyList<SKRectI> rects)
    {
        var result = new float[rects.Count];
        for (int i = 0; i < rects.Count; i++)
        {
            result[i] = (float)(GetFillPercentLocalContrast(warped, rects[i]) / 100.0);
        }

        return result;
    }

    public static double GetFillPercentLocalContrast(SKBitmap src, SKRectI rect)
    {
        const int margin = 3;
        var expanded = new SKRectI(
            Math.Max(0, rect.Left - margin),
            Math.Max(0, rect.Top - margin),
            Math.Min(src.Width, rect.Right + margin),
            Math.Min(src.Height, rect.Bottom + margin)
        );
        if (expanded.Width <= 0 || expanded.Height <= 0)
        {
            return 0;
        }

        rect = expanded;

        int ix = Math.Max(1, (int)Math.Round(rect.Width * 0.12));
        int iy = Math.Max(1, (int)Math.Round(rect.Height * 0.12));
        var r = new SKRectI(rect.Left + ix, rect.Top + iy, rect.Right - ix, rect.Bottom - iy);
        if (r.Width <= 2 || r.Height <= 2)
        {
            return 0;
        }

        using var roi = new SKBitmap(r.Width, r.Height, src.ColorType, src.AlphaType);
        using (var canvas = new SKCanvas(roi))
        {
            canvas.DrawBitmap(src, r, new SKRect(0, 0, r.Width, r.Height));
        }

        using var pm = new SKPixmap(roi.Info, roi.GetPixels(out _));

        Span<int> hist = stackalloc int[256];
        hist.Clear();
        int total = roi.Width * roi.Height;
        unsafe
        {
            byte* p = (byte*)pm.GetPixels();
            int bpp = pm.Info.BytesPerPixel;
            for (int y = 0; y < roi.Height; y++)
            {
                byte* row = p + y * pm.RowBytes;
                for (int x = 0; x < roi.Width; x++)
                {
                    byte B = row[x * bpp + 0], G = row[x * bpp + 1], R = row[x * bpp + 2];
                    int y8 = (int)Math.Round(0.2126 * R + 0.7152 * G + 0.0722 * B);
                    int d = 255 - y8;
                    hist[d]++;
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
        Span<int> histN = stackalloc int[256];
        histN.Clear();
        int blacks = 0;
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
                    double dn = (d - pLo) / range;
                    if (dn < 0) dn = 0; else if (dn > 1) dn = 1;
                    int q = (int)Math.Round(dn * 255.0);
                    histN[q]++;
                }
            }
        }

        int qThr = OtsuThreshold(histN, total);
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
                    if (q >= qThr)
                    {
                        blacks++;
                    }
                }
            }
        }

        double pct = 100.0 * blacks / Math.Max(1, total);
        if (double.IsNaN(pct)) pct = 0;
        if (pct < 0) pct = 0; else if (pct > 100) pct = 100;
        return pct;
    }

    private static int PercentileFromHist(Span<int> histogram, int total, double p)
    {
        int target = (int)Math.Round(p * total);
        int cumulative = 0;
        for (int i = 0; i < 256; i++)
        {
            cumulative += histogram[i];
            if (cumulative >= target)
            {
                return i;
            }
        }

        return 255;
    }

    private static int OtsuThreshold(Span<int> histogram, int total)
    {
        double sum = 0;
        for (int t = 0; t < 256; t++)
        {
            sum += t * histogram[t];
        }

        double sumB = 0;
        int wB = 0;
        double varMax = -1;
        int threshold = 128;
        for (int t = 0; t < 256; t++)
        {
            wB += histogram[t];
            if (wB == 0) continue;
            int wF = total - wB;
            if (wF == 0) break;
            sumB += t * histogram[t];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double varBetween = wB * wF * (mB - mF) * (mB - mF);
            if (varBetween > varMax)
            {
                varMax = varBetween;
                threshold = t;
            }
        }

        return threshold;
    }

    private static float FillRatioFromRoi(SKBitmap warpedBgr, SKRectI r, float padFrac, float openFrac)
    {
        int x = r.Left, y = r.Top, w = r.Width, h = r.Height;
        if (w <= 0 || h <= 0)
        {
            return 0f;
        }

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
                for (int i = inner.Left; i < inner.Right; i++)
                {
                    hist[p[i]]++;
                }
            }
        }

        byte thr = Otsu(hist, total);
        int k = Math.Max(1, (int)Math.Round(Math.Min(inner.Width, inner.Height) * openFrac));
        if (k % 2 == 0)
        {
            k++;
        }

        int ones = 0;
        unsafe
        {
            var tmp = new byte[inner.Width * inner.Height];
            int idx = 0;
            for (int j = inner.Top; j < inner.Bottom; j++)
            {
                var p = (byte*)roi.GetPixels() + j * roi.RowBytes;
                for (int i = inner.Left; i < inner.Right; i++)
                {
                    tmp[idx++] = (byte)(p[i] <= thr ? 1 : 0);
                }
            }

            var eroded = MorphErode(tmp, inner.Width, inner.Height, k);
            var opened = MorphDilate(eroded, inner.Width, inner.Height, k);
            for (int t = 0; t < opened.Length; t++)
            {
                ones += opened[t];
            }
        }

        return (float)ones / (inner.Width * inner.Height);
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

    private static byte[] MorphErode(byte[] src, int w, int h, int k)
    {
        int r = k / 2;
        var dst = new byte[src.Length];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                bool ok = true;
                for (int dy = -r; dy <= r && ok; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= h)
                    {
                        ok = false;
                        break;
                    }

                    int row = yy * w;
                    for (int dx = -r; dx <= r; dx++)
                    {
                        int xx = x + dx;
                        if (xx < 0 || xx >= w)
                        {
                            ok = false;
                            break;
                        }

                        if (src[row + xx] == 0)
                        {
                            ok = false;
                            break;
                        }
                    }
                }

                dst[y * w + x] = (byte)(ok ? 1 : 0);
            }
        }

        return dst;
    }

    private static byte[] MorphDilate(byte[] src, int w, int h, int k)
    {
        int r = k / 2;
        var dst = new byte[src.Length];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                bool any = false;
                for (int dy = -r; dy <= r && !any; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= h)
                    {
                        continue;
                    }

                    int row = yy * w;
                    for (int dx = -r; dx <= r; dx++)
                    {
                        int xx = x + dx;
                        if (xx < 0 || xx >= w)
                        {
                            continue;
                        }

                        if (src[row + xx] != 0)
                        {
                            any = true;
                            break;
                        }
                    }
                }

                dst[y * w + x] = (byte)(any ? 1 : 0);
            }
        }

        return dst;
    }
}

using Microsoft.Maui.Storage;
using SkiaSharp;

namespace YourApp.Services;

public class PictureCleaner
{
    public string Clean(string photoPath)
    {
        if (string.IsNullOrWhiteSpace(photoPath) || !File.Exists(photoPath))
        {
            throw new FileNotFoundException("Soubor s fotkou nebyl nalezen.", photoPath);
        }

        using var source = SKBitmap.Decode(photoPath) ?? throw new InvalidOperationException("Nelze dekódovat fotku.");
        var pixelCount = source.Width * source.Height;
        if (pixelCount == 0)
        {
            return photoPath;
        }

        var luminance = new byte[pixelCount];
        var hist = new int[256];

        unsafe
        {
            using var pm = new SKPixmap(source.Info, source.GetPixels(out _));
            byte* ptr = (byte*)pm.GetPixels();
            int bpp = pm.Info.BytesPerPixel;
            int idx = 0;

            for (int y = 0; y < source.Height; y++)
            {
                byte* row = ptr + y * pm.RowBytes;
                for (int x = 0; x < source.Width; x++, idx++)
                {
                    byte b = row[x * bpp + 0];
                    byte g = row[x * bpp + 1];
                    byte r = row[x * bpp + 2];
                    byte lum = (byte)Math.Clamp((int)Math.Round(0.2126 * r + 0.7152 * g + 0.0722 * b), 0, 255);
                    luminance[idx] = lum;
                    hist[lum]++;
                }
            }
        }

        int low = Percentile(hist, pixelCount, 0.05);
        int high = Percentile(hist, pixelCount, 0.95);
        if (high <= low)
        {
            high = Math.Min(255, low + 1);
            low = Math.Max(0, low - 1);
        }

        var normalized = new byte[pixelCount];
        var histNormalized = new int[256];

        for (int i = 0; i < pixelCount; i++)
        {
            double value = (luminance[i] - low) / (double)(high - low);
            if (value < 0)
            {
                value = 0;
            }
            else if (value > 1)
            {
                value = 1;
            }

            byte scaled = (byte)Math.Round(value * 255.0);
            normalized[i] = scaled;
            histNormalized[scaled]++;
        }

        int threshold = OtsuThreshold(histNormalized, pixelCount);

        using var cleaned = new SKBitmap(source.Width, source.Height, SKColorType.Rgba8888, SKAlphaType.Premul);

        unsafe
        {
            using var dest = new SKPixmap(cleaned.Info, cleaned.GetPixels(out _));
            byte* ptr = (byte*)dest.GetPixels();
            int bpp = dest.Info.BytesPerPixel;
            int idx = 0;

            for (int y = 0; y < cleaned.Height; y++)
            {
                byte* row = ptr + y * dest.RowBytes;
                for (int x = 0; x < cleaned.Width; x++, idx++)
                {
                    byte v = normalized[idx] <= threshold ? (byte)0 : (byte)255;
                    row[x * bpp + 0] = v;
                    row[x * bpp + 1] = v;
                    row[x * bpp + 2] = v;
                    row[x * bpp + 3] = 255;
                }
            }
        }

        var outputDirectory = Path.Combine(FileSystem.Current.CacheDirectory, "cleaned");
        Directory.CreateDirectory(outputDirectory);
        var fileName = Path.GetFileNameWithoutExtension(photoPath);
        var cleanedPath = Path.Combine(outputDirectory, $"{fileName}_clean_{Guid.NewGuid():N}.png");

        using (var image = SKImage.FromBitmap(cleaned))
        using (var data = image.Encode(SKEncodedImageFormat.Png, 100))
        using (var stream = File.Create(cleanedPath))
        {
            data.SaveTo(stream);
        }

        return cleanedPath;
    }

    private static int Percentile(int[] hist, int total, double percentile)
    {
        int target = (int)Math.Round(percentile * (total - 1));
        int cumulative = 0;
        for (int i = 0; i < hist.Length; i++)
        {
            cumulative += hist[i];
            if (cumulative > target)
            {
                return i;
            }
        }

        return hist.Length - 1;
    }

    private static int OtsuThreshold(int[] hist, int total)
    {
        double sum = 0;
        for (int i = 0; i < hist.Length; i++)
        {
            sum += i * hist[i];
        }

        double sumB = 0;
        int wB = 0;
        int wF;
        double maxVariance = -1;
        int threshold = 0;

        for (int t = 0; t < hist.Length; t++)
        {
            wB += hist[t];
            if (wB == 0)
            {
                continue;
            }

            wF = total - wB;
            if (wF == 0)
            {
                break;
            }

            sumB += t * hist[t];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double variance = wB * wF * Math.Pow(mB - mF, 2);

            if (variance > maxVariance)
            {
                maxVariance = variance;
                threshold = t;
            }
        }

        return threshold;
    }
}

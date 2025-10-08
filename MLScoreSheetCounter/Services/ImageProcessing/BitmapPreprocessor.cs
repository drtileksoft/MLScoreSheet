using System;
using System.IO;
using SkiaSharp;

namespace YourApp.Services;

internal static class BitmapPreprocessor
{
    public static SKBitmap DecodeLandscapePhoto(Stream photoStream)
    {
        var decoded = SKBitmap.Decode(photoStream) ?? throw new InvalidOperationException("Nelze dekódovat foto.");

        if (decoded.Width >= decoded.Height)
        {
            return decoded;
        }

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

    public static SKBitmap ToGray(SKBitmap source)
    {
        var gray = new SKBitmap(source.Width, source.Height, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            for (int y = 0; y < source.Height; y++)
            {
                var sourceRow = (uint*)source.GetPixels() + y * source.Width;
                var targetRow = (byte*)gray.GetPixels() + y * gray.RowBytes;
                for (int x = 0; x < source.Width; x++)
                {
                    uint pixel = sourceRow[x];
                    byte b = (byte)(pixel & 0xFF);
                    byte g = (byte)((pixel >> 8) & 0xFF);
                    byte r = (byte)((pixel >> 16) & 0xFF);
                    byte a = (byte)((pixel >> 24) & 0xFF);
                    int add = 255 - a;
                    int rOut = r + add;
                    int gOut = g + add;
                    int bOut = b + add;
                    int value = (int)Math.Round(0.299 * rOut + 0.587 * gOut + 0.114 * bOut);
                    targetRow[x] = (byte)(value < 0 ? 0 : (value > 255 ? 255 : value));
                }
            }
        }
        return gray;
    }
}

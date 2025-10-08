using System;
using SkiaSharp;

namespace YourApp.Services;

internal static class HomographyCalculator
{
    public static float[] Compute(
        (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) src,
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
        return new float[]
        {
            (float)h[0], (float)h[1], (float)h[2],
            (float)h[3], (float)h[4], (float)h[5],
            (float)h[6], (float)h[7], 1f
        };
    }

    public static SKBitmap WarpToTemplate(SKBitmap src, float[] H, int width, int height)
    {
        var dst = new SKBitmap(width, height, SKColorType.Bgra8888, SKAlphaType.Premul);
        using var canvas = new SKCanvas(dst);
        canvas.Clear(SKColors.Black);

        var matrix = new SKMatrix
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

        canvas.SetMatrix(matrix);
        canvas.DrawBitmap(src, 0, 0);
        canvas.Flush();
        return dst;
    }

    private static double[] Solve8x8(double[,] A, double[] b)
    {
        const int n = 8;
        double[,] M = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                M[i, j] = A[i, j];
            }

            M[i, n] = b[i];
        }

        for (int i = 0; i < n; i++)
        {
            int pivot = i;
            for (int r = i + 1; r < n; r++)
            {
                if (Math.Abs(M[r, i]) > Math.Abs(M[pivot, i]))
                {
                    pivot = r;
                }
            }

            if (pivot != i)
            {
                for (int c = i; c <= n; c++)
                {
                    (M[i, c], M[pivot, c]) = (M[pivot, c], M[i, c]);
                }
            }

            double div = M[i, i];
            if (Math.Abs(div) < 1e-12)
            {
                continue;
            }

            for (int c = i; c <= n; c++)
            {
                M[i, c] /= div;
            }

            for (int r = 0; r < n; r++)
            {
                if (r == i) continue;
                double mul = M[r, i];
                for (int c = i; c <= n; c++)
                {
                    M[r, c] -= mul * M[i, c];
                }
            }
        }

        var x = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = M[i, n];
        }

        return x;
    }
}

using SkiaSharp;

namespace MLScoreSheet.Core;

public static partial class SheetScoreEngine
{
    private static float[] ComputeHomography(
        (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) src,
        (SKPoint TL, SKPoint TR, SKPoint BR, SKPoint BL) dst)
    {
        var s = new[] { src.TL, src.TR, src.BR, src.BL };
        var d = new[] { dst.TL, dst.TR, dst.BR, dst.BL };

        double[,] matrix = new double[8, 8];
        double[] rhs = new double[8];

        for (int i = 0; i < 4; i++)
        {
            double x = s[i].X, y = s[i].Y;
            double X = d[i].X, Y = d[i].Y;

            int r1 = i * 2;
            int r2 = i * 2 + 1;
            matrix[r1, 0] = x;
            matrix[r1, 1] = y;
            matrix[r1, 2] = 1;
            matrix[r1, 3] = 0;
            matrix[r1, 4] = 0;
            matrix[r1, 5] = 0;
            matrix[r1, 6] = -x * X;
            matrix[r1, 7] = -y * X;
            matrix[r2, 0] = 0;
            matrix[r2, 1] = 0;
            matrix[r2, 2] = 0;
            matrix[r2, 3] = x;
            matrix[r2, 4] = y;
            matrix[r2, 5] = 1;
            matrix[r2, 6] = -x * Y;
            matrix[r2, 7] = -y * Y;
            rhs[r1] = X;
            rhs[r2] = Y;
        }

        var h = Solve8x8(matrix, rhs);
        return new float[]
        {
            (float)h[0], (float)h[1], (float)h[2],
            (float)h[3], (float)h[4], (float)h[5],
            (float)h[6], (float)h[7], 1f
        };
    }

    private static double[] Solve8x8(double[,] matrix, double[] rhs)
    {
        int n = 8;
        double[,] augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) augmented[i, j] = matrix[i, j];
            augmented[i, n] = rhs[i];
        }

        for (int i = 0; i < n; i++)
        {
            int pivot = i;
            for (int r = i + 1; r < n; r++)
                if (Math.Abs(augmented[r, i]) > Math.Abs(augmented[pivot, i]))
                    pivot = r;
            if (pivot != i)
            {
                for (int c = i; c <= n; c++)
                    (augmented[i, c], augmented[pivot, c]) = (augmented[pivot, c], augmented[i, c]);
            }

            double div = augmented[i, i];
            if (Math.Abs(div) < 1e-12) continue;
            for (int c = i; c <= n; c++)
                augmented[i, c] /= div;

            for (int r = 0; r < n; r++)
            {
                if (r == i) continue;
                double mul = augmented[r, i];
                for (int c = i; c <= n; c++)
                    augmented[r, c] -= mul * augmented[i, c];
            }
        }

        var solution = new double[n];
        for (int i = 0; i < n; i++)
            solution[i] = augmented[i, n];
        return solution;
    }

    private static SKBitmap WarpToTemplate(SKBitmap src, float[] H, int width, int height)
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

    private static SKPoint ApplyHomography(float[] H, SKPoint pt)
    {
        float denom = H[6] * pt.X + H[7] * pt.Y + 1f;
        if (Math.Abs(denom) < 1e-6f)
            denom = denom >= 0 ? 1e-6f : -1e-6f;
        float x = (H[0] * pt.X + H[1] * pt.Y + H[2]) / denom;
        float y = (H[3] * pt.X + H[4] * pt.Y + H[5]) / denom;
        return new SKPoint(x, y);
    }
}

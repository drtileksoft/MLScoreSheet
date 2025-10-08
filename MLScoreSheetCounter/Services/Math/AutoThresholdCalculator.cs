using System;
using System.Linq;

namespace YourApp.Services;

internal static class AutoThresholdCalculator
{
    public static float SelectThreshold(
        float[] values,
        bool autoThreshold,
        float autoMin,
        float autoMax,
        float fixedThreshold,
        float minGap = 0.12f)
    {
        return autoThreshold
            ? AutoThresholdKMeans(values, autoMin, autoMax, fixedThreshold, minGap)
            : fixedThreshold;
    }

    public static float AutoThresholdKMeans(float[] values, float tmin, float tmax, float fallback, float minGap, int iterations = 20)
    {
        if (values.Length == 0)
        {
            return fallback;
        }

        var v = values.Select(x => Math.Clamp(x, 0f, 1f)).OrderBy(x => x).ToArray();
        float c0 = Percentile(v, 20f);
        float c1 = Percentile(v, 80f);
        for (int i = 0; i < iterations; i++)
        {
            float mid = 0.5f * (c0 + c1);
            var m0 = v.Where(x => x < mid).ToArray();
            var m1 = v.Where(x => x >= mid).ToArray();
            if (m0.Length == 0 || m1.Length == 0)
            {
                break;
            }

            float c0n = (float)m0.Average();
            float c1n = (float)m1.Average();
            if (Math.Abs(c0n - c0) < 1e-4 && Math.Abs(c1n - c1) < 1e-4)
            {
                c0 = c0n;
                c1 = c1n;
                break;
            }

            c0 = c0n;
            c1 = c1n;
        }

        if (c0 > c1)
        {
            (c0, c1) = (c1, c0);
        }

        float thr = (c0 + c1) * 0.5f;
        if ((c1 - c0) < minGap)
        {
            thr = fallback;
        }

        return Math.Clamp(thr, tmin, tmax);
    }

    private static float Percentile(float[] array, float p)
    {
        if (array.Length == 0)
        {
            return 0f;
        }

        float pos = (p / 100f) * (array.Length - 1);
        int lo = (int)Math.Floor(pos);
        int hi = (int)Math.Ceiling(pos);
        if (lo == hi)
        {
            return array[lo];
        }

        return array[lo] + (array[hi] - array[lo]) * (pos - lo);
    }
}

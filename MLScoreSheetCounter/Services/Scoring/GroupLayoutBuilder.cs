using System;
using System.Collections.Generic;
using System.Linq;
using SkiaSharp;

namespace YourApp.Services;

internal sealed class ScoreGroup
{
    public int[] Indices { get; init; } = Array.Empty<int>();
    public int ChosenSlot { get; set; } = -1;

    public int ValueOf(int slot) => slot;

    public int ScoreContribution(float threshold, float[] fillRatios)
    {
        if (ChosenSlot < 0)
        {
            return 0;
        }

        int index = Indices[ChosenSlot];
        return fillRatios[index] >= threshold ? ValueOf(ChosenSlot) : 0;
    }
}

internal static class GroupLayoutBuilder
{
    public static List<ScoreGroup> BuildGroupsGrid3x2(IReadOnlyList<SKRectI> rects, float[] fillRatios)
    {
        var items = rects.Select((r, i) => new Item
        {
            Cx = r.Left + r.Width * 0.5f,
            Cy = r.Top + r.Height * 0.5f,
            W = r.Width,
            H = r.Height,
            P = fillRatios[i],
            Index = i
        }).OrderBy(z => z.Cy).ToList();

        float hmed = items.Select(z => z.H).OrderBy(x => x).ElementAt(items.Count / 2);
        float rowThr = 0.6f * hmed;

        var rows = new List<List<Item>>();
        foreach (var it in items)
        {
            if (rows.Count == 0)
            {
                rows.Add(new List<Item> { it });
            }
            else
            {
                var last = rows.Last();
                var cyMed = last.Select(z => z.Cy).OrderBy(x => x).ElementAt(last.Count / 2);
                if (Math.Abs(it.Cy - cyMed) <= rowThr)
                {
                    last.Add(it);
                }
                else
                {
                    rows.Add(new List<Item> { it });
                }
            }
        }

        foreach (var row in rows)
        {
            row.Sort((a, b) => a.Cx.CompareTo(b.Cx));
        }

        var groups = new List<ScoreGroup>();
        for (int i = 0; i + 1 < rows.Count; i += 2)
        {
            var top = rows[i];
            var bottom = rows[i + 1];
            int nt = top.Count / 3;
            int nb = bottom.Count / 3;
            int n = Math.Min(nt, nb);
            for (int j = 0; j < n; j++)
            {
                var t3 = top.Skip(j * 3).Take(3).ToArray();
                var b3 = bottom.Skip(j * 3).Take(3).ToArray();
                var group = new ScoreGroup
                {
                    Indices = new[] { t3[0].Index, t3[1].Index, t3[2].Index, b3[0].Index, b3[1].Index, b3[2].Index }
                };

                float bestP = -1f;
                int bestSlot = -1;
                for (int slot = 0; slot < 6; slot++)
                {
                    int idx = group.Indices[slot];
                    float p = fillRatios[idx];
                    if (p > bestP)
                    {
                        bestP = p;
                        bestSlot = slot;
                    }
                }

                group.ChosenSlot = bestSlot;
                groups.Add(group);
            }
        }

        return groups;
    }

    public static int TotalScoreFromItems(IReadOnlyList<SKRectI> rects, float[] fillRatios, float threshold)
    {
        var groups = BuildGroupsGrid3x2(rects, fillRatios);
        int total = 0;
        foreach (var group in groups)
        {
            total += group.ScoreContribution(threshold, fillRatios);
        }

        return total;
    }

    private sealed class Item
    {
        public float Cx { get; init; }
        public float Cy { get; init; }
        public float W { get; init; }
        public float H { get; init; }
        public float P { get; init; }
        public int Index { get; init; }
    }
}

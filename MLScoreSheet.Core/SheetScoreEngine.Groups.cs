using SkiaSharp;

namespace MLScoreSheet.Core;

public static partial class SheetScoreEngine
{
    private sealed class Group
    {
        public int[] Indices { get; init; } = new int[6];
        public int ChosenSlot { get; set; } = -1;
        public int ValueOf(int slot) => slot;
        public int ScoreContribution(float thr, float[] pList)
        {
            if (ChosenSlot < 0) return 0;
            int idx = Indices[ChosenSlot];
            return pList[idx] >= thr ? ValueOf(ChosenSlot) : 0;
        }
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

    private static int TotalScoreFromItems(List<SKRectI> rects, float[] pList, float thr)
    {
        var groups = BuildGroupsGrid3x2(rects, pList);
        int total = 0;
        foreach (var g in groups)
            total += g.ScoreContribution(thr, pList);
        return total;
    }

    private static List<Group> BuildGroupsGrid3x2(List<SKRectI> rects, float[] pList)
    {
        var items = rects.Select((r, i) => new Item
        {
            Cx = r.Left + r.Width * 0.5f,
            Cy = r.Top + r.Height * 0.5f,
            W = r.Width,
            H = r.Height,
            P = pList[i],
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
                    last.Add(it);
                else
                    rows.Add(new List<Item> { it });
            }
        }

        foreach (var r in rows)
            r.Sort((a, b) => a.Cx.CompareTo(b.Cx));

        var groups = new List<Group>();
        for (int i = 0; i + 1 < rows.Count; i += 2)
        {
            var top = rows[i];
            var bot = rows[i + 1];
            int nt = top.Count / 3;
            int nb = bot.Count / 3;
            int n = Math.Min(nt, nb);
            for (int j = 0; j < n; j++)
            {
                var t3 = top.Skip(j * 3).Take(3).ToArray();
                var b3 = bot.Skip(j * 3).Take(3).ToArray();

                var g = new Group
                {
                    Indices = new[]
                    {
                        t3[0].Index, t3[1].Index, t3[2].Index,
                        b3[0].Index, b3[1].Index, b3[2].Index
                    }
                };

                float bestP = -1f;
                int bestSlot = -1;
                for (int s = 0; s < 6; s++)
                {
                    int idx = g.Indices[s];
                    float p = pList[idx];
                    if (p > bestP)
                    {
                        bestP = p;
                        bestSlot = s;
                    }
                }
                g.ChosenSlot = bestSlot;
                groups.Add(g);
            }
        }
        return groups;
    }
}

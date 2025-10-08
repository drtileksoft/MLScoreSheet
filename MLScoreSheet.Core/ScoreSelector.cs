using System;
using System.Collections.Generic;
using System.Linq;
using SkiaSharp;

namespace MLScoreSheet.Core;

public static class ScoreSelector3x2
{
        public sealed class Result
        {
            public int Total { get; set; }
            public float ThresholdUsed { get; set; }
            public List<int> WinnerIndices { get; set; } = new(); // indexy do původního rects/pList
        }

        /// <summary>
        /// Vypočti skóre + vítěze v každé 3×2 šestici (pořadí 0 1 2 / 3 4 5).
        /// Vstup: rects[i], pList[i] (0..1), thr (0..1).
        /// Rects nemusejí být dokonale seřazené – řádkování/párování si uděláme sami.
        /// </summary>
        public static Result SumWinnerTakesAll(
            IList<SKRectI> rects, IList<float> pList, float thr)
        {
            if (rects == null || pList == null || rects.Count != pList.Count || rects.Count == 0)
                return new Result { Total = 0, ThresholdUsed = thr };

            // Připrav pole [cx, cy, x, y, w, h, p, origIndex]
            var items = new List<float[]>(rects.Count);
            for (int i = 0; i < rects.Count; i++)
            {
                var r = rects[i];
                float w = r.Width, h = r.Height;
                items.Add(new float[] {
                r.MidX, r.MidY, r.Left, r.Top, w, h, pList[i], i
            });
            }

            // Seřaď podle cy
            items.Sort((a, b) => a[1].CompareTo(b[1]));

            // Prah sloučení do řádku: 0.6 * medián výšky
            float hMed = Median(items.Select(a => a[5]));
            float rowThresh = MathF.Max(1f, 0.6f * hMed);

            // Seskup podle řádků (scanline)
            var rows = new List<List<float[]>>();
            var cur = new List<float[]> { items[0] };
            for (int i = 1; i < items.Count; i++)
            {
                float cy = items[i][1];
                float medianCyCur = Median(cur.Select(x => x[1]));
                if (MathF.Abs(cy - medianCyCur) <= rowThresh)
                    cur.Add(items[i]);
                else
                {
                    rows.Add(SortByX(cur));
                    cur = new List<float[]> { items[i] };
                }
            }
            rows.Add(SortByX(cur));

            // Projdi dvojice řádků (horní+spodní), po trojicích sloupců
            int total = 0;
            var winners = new List<int>();

            for (int ri = 0; ri + 1 < rows.Count; ri += 2)
            {
                var top = rows[ri];
                var bot = rows[ri + 1];

                int nt = top.Count / 3;
                int nb = bot.Count / 3;
                int nGroups = Math.Min(nt, nb);

                for (int g = 0; g < nGroups; g++)
                {
                    // top trojice: indexy g*3..g*3+2
                    // bot trojice: indexy g*3..g*3+2
                    int t0 = g * 3, b0 = g * 3;

                    // posbírat kandidáty nad prahem, s (value, conf, origIndex)
                    var cand = new List<(int value, float conf, int origIdx)>(6);

                    for (int k = 0; k < 3; k++)
                    {
                        float pt = top[t0 + k][6];
                        if (pt >= thr)
                        {
                            int orig = (int)top[t0 + k][7];
                            cand.Add((k /*0..2*/, pt, orig));
                        }

                        float pb = bot[b0 + k][6];
                        if (pb >= thr)
                        {
                            int orig = (int)bot[b0 + k][7];
                            cand.Add((3 + k /*3..5*/, pb, orig));
                        }
                    }

                    if (cand.Count > 0)
                    {
                        // vezmi jen JEDNOHO – s nejvyšším % černé
                        var best = cand[0];
                        for (int i = 1; i < cand.Count; i++)
                            if (cand[i].conf > best.conf) best = cand[i];

                        total += best.value;          // přičti skóre 0..5
                        winners.Add(best.origIdx);     // pro overlay: jen tenhle bude zelený
                    }
                }
            }

            return new Result { Total = total, ThresholdUsed = thr, WinnerIndices = winners };
        }

        // --------------- helpers ---------------
        private static List<float[]> SortByX(List<float[]> row)
        {
            row.Sort((a, b) => a[0].CompareTo(b[0]));
            return row;
        }

        private static float Median(IEnumerable<float> data)
        {
            var arr = data.ToArray();
            Array.Sort(arr);
            if (arr.Length == 0) return 0f;
            int m = arr.Length / 2;
            return (arr.Length % 2 == 1) ? arr[m] : 0.5f * (arr[m - 1] + arr[m]);
        }
    }

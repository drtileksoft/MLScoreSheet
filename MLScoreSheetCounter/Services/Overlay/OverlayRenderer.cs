using System;
using System.Collections.Generic;
using System.Linq;
using SkiaSharp;

namespace YourApp.Services;

internal static class OverlayRenderer
{
    public static SKBitmap CreateWarpedOverlay(
        SKBitmap warped,
        IReadOnlyList<SKRectI> rects,
        float[] fillRatios,
        IList<int> winnerIndices,
        IReadOnlyList<ScoreGroup> groups,
        float visibilityThreshold)
    {
        var slotByIndex = Enumerable.Repeat(-1, rects.Count).ToArray();
        for (int gi = 0; gi < groups.Count; gi++)
        {
            for (int s = 0; s < groups[gi].Indices.Length; s++)
            {
                slotByIndex[groups[gi].Indices[s]] = s;
            }
        }

        var winners = new HashSet<int>(winnerIndices);

        const int tablesPerRowBlock = 2;
        const int rowsPerTable = 5;
        const int colsPerTable = 3;
        const int totalTables = tablesPerRowBlock * 2;
        const int groupsPerRowPair = colsPerTable * tablesPerRowBlock;

        var rowSums = new int[totalTables, rowsPerTable];
        var colSums = new int[totalTables, colsPerTable];
        var tableTotals = new int[totalTables];

        var rowBounds = new SKRect[totalTables, rowsPerTable];
        var colBounds = new SKRect[totalTables, colsPerTable];
        var tableBounds = new SKRect[totalTables];
        var rowHas = new bool[totalTables, rowsPerTable];
        var colHas = new bool[totalTables, colsPerTable];
        var tableHas = new bool[totalTables];

        var groupWidths = new List<float>();
        var groupHeights = new List<float>();

        var visWarp = warped.Copy();
        using (var canvas = new SKCanvas(visWarp))
        {
            var green = new SKPaint { Color = new SKColor(40, 200, 40), Style = SKPaintStyle.Stroke, StrokeWidth = 2f, IsAntialias = true };
            var red = new SKPaint { Color = new SKColor(230, 40, 40), Style = SKPaintStyle.Stroke, StrokeWidth = 2f, IsAntialias = true };
            var blue = new SKPaint { Color = new SKColor(70, 130, 240), Style = SKPaintStyle.Stroke, StrokeWidth = 2.5f, PathEffect = SKPathEffect.CreateDash(new float[] { 6, 6 }, 0), IsAntialias = true };

            var textPaint = new SKPaint { Color = SKColors.Blue, TextSize = 30, IsAntialias = true, Typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Bold) };
            var shadowPaint = new SKPaint { Color = new SKColor(0, 0, 0, 180), TextSize = 30, IsAntialias = true, Typeface = textPaint.Typeface };

            for (int gi = 0; gi < groups.Count; gi++)
            {
                var group = groups[gi];
                var rectangles = group.Indices.Select(i => rects[i]).ToArray();
                int minX = rectangles.Min(r => r.Left);
                int minY = rectangles.Min(r => r.Top);
                int maxX = rectangles.Max(r => r.Right);
                int maxY = rectangles.Max(r => r.Bottom);
                const int pad = 3;
                var groupRect = new SKRect(minX - pad, minY - pad, maxX + pad, maxY + pad);
                canvas.DrawRect(groupRect, blue);

                if (groupsPerRowPair <= 0)
                {
                    continue;
                }

                int rowPairIndex = gi / groupsPerRowPair;
                int posInPair = gi % groupsPerRowPair;

                int tableRowBlock = rowPairIndex / rowsPerTable;
                int rowInTable = rowPairIndex % rowsPerTable;
                int tableColBlock = posInPair / colsPerTable;
                int colInTable = posInPair % colsPerTable;

                if (tableRowBlock >= 2 || tableColBlock >= tablesPerRowBlock)
                {
                    continue;
                }

                int tableIndex = tableRowBlock * tablesPerRowBlock + tableColBlock;
                if (tableIndex < 0 || tableIndex >= totalTables)
                {
                    continue;
                }

                groupWidths.Add(groupRect.Width);
                groupHeights.Add(groupRect.Height);

                if (!rowHas[tableIndex, rowInTable])
                {
                    rowBounds[tableIndex, rowInTable] = groupRect;
                    rowHas[tableIndex, rowInTable] = true;
                }
                else
                {
                    rowBounds[tableIndex, rowInTable] = SKRect.Union(rowBounds[tableIndex, rowInTable], groupRect);
                }

                if (!colHas[tableIndex, colInTable])
                {
                    colBounds[tableIndex, colInTable] = groupRect;
                    colHas[tableIndex, colInTable] = true;
                }
                else
                {
                    colBounds[tableIndex, colInTable] = SKRect.Union(colBounds[tableIndex, colInTable], groupRect);
                }

                if (!tableHas[tableIndex])
                {
                    tableBounds[tableIndex] = groupRect;
                    tableHas[tableIndex] = true;
                }
                else
                {
                    tableBounds[tableIndex] = SKRect.Union(tableBounds[tableIndex], groupRect);
                }

                bool hasWinner = false;
                int groupScore = 0;
                for (int s = 0; s < group.Indices.Length; s++)
                {
                    int idx = group.Indices[s];
                    if (winners.Contains(idx))
                    {
                        hasWinner = true;
                        int slot = slotByIndex[idx];
                        if (slot >= 0)
                        {
                            groupScore = group.ValueOf(slot);
                        }
                        break;
                    }
                }

                if (hasWinner)
                {
                    rowSums[tableIndex, rowInTable] += groupScore;
                    colSums[tableIndex, colInTable] += groupScore;
                    tableTotals[tableIndex] += groupScore;
                }
            }

            for (int i = 0; i < rects.Count; i++)
            {
                var r = rects[i];
                bool isWin = winners.Contains(i);
                canvas.DrawRect(new SKRect(r.Left, r.Top, r.Right, r.Bottom), isWin ? green : red);

                string idxLabel = $"#{i}";
                string fillLabel = $"{Math.Round(fillRatios[i] * 100)}%";
                bool showFill = fillRatios[i] > visibilityThreshold;
                string slotLabel = slotByIndex[i] >= 0 ? slotByIndex[i].ToString() : "?";

                if (showFill)
                {
                    DrawText(canvas, idxLabel, r.Left - 35, r.Top - 5, textPaint, shadowPaint);
                    DrawText(canvas, fillLabel, r.Right + 35 - textPaint.MeasureText(fillLabel), r.Bottom + 10, textPaint, shadowPaint);
                }

                float cx = r.Left + r.Width * 0.5f;
                float cy = r.Top + r.Height * 0.55f;
                using var slotPaint = new SKPaint { Color = isWin ? green.Color : red.Color, TextSize = 40, IsAntialias = true, Typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Normal) };
                using var slotShadow = new SKPaint { Color = new SKColor(0, 0, 0, 200), TextSize = 40, IsAntialias = true, Typeface = slotPaint.Typeface };
                float sw = slotPaint.MeasureText(slotLabel);
                DrawText(canvas, slotLabel, cx - sw / 2, cy, slotPaint, slotShadow);
            }

            if (groupWidths.Count > 0 && groupHeights.Count > 0)
            {
                float avgWidth = groupWidths.Average();
                float avgHeight = groupHeights.Average();
                float rowOffset = Math.Max(40f, avgWidth * 0.55f);
                float colOffset = Math.Max(35f, avgHeight * 0.65f);

                using var sumPaint = new SKPaint
                {
                    Color = new SKColor(255, 230, 90),
                    TextSize = Math.Max(36f, avgHeight * 0.65f),
                    IsAntialias = true,
                    Typeface = textPaint.Typeface
                };
                using var sumShadow = new SKPaint
                {
                    Color = new SKColor(0, 0, 0, 200),
                    TextSize = sumPaint.TextSize,
                    IsAntialias = true,
                    Typeface = textPaint.Typeface
                };
                using var totalPaint = new SKPaint
                {
                    Color = new SKColor(255, 255, 255),
                    TextSize = sumPaint.TextSize * 1.15f,
                    IsAntialias = true,
                    Typeface = textPaint.Typeface
                };
                using var totalShadow = new SKPaint
                {
                    Color = new SKColor(0, 0, 0, 220),
                    TextSize = totalPaint.TextSize,
                    IsAntialias = true,
                    Typeface = textPaint.Typeface
                };

                for (int table = 0; table < totalTables; table++)
                {
                    if (!tableHas[table]) continue;

                    for (int r = 0; r < rowsPerTable; r++)
                    {
                        if (!rowHas[table, r]) continue;
                        int value = rowSums[table, r];
                        string text = value.ToString();
                        var bounds = rowBounds[table, r];
                        float x = bounds.Right + rowOffset;
                        float y = bounds.MidY + sumPaint.TextSize * 0.35f;
                        DrawText(canvas, text, x, y, sumPaint, sumShadow);
                    }

                    for (int c = 0; c < colsPerTable; c++)
                    {
                        if (!colHas[table, c]) continue;
                        int value = colSums[table, c];
                        string text = value.ToString();
                        var bounds = colBounds[table, c];
                        float textWidth = sumPaint.MeasureText(text);
                        float x = bounds.MidX - textWidth * 0.5f;
                        float y = tableBounds[table].Bottom + colOffset + sumPaint.TextSize * 0.35f;
                        DrawText(canvas, text, x, y, sumPaint, sumShadow);
                    }

                    int total = tableTotals[table];
                    string totalText = total.ToString();
                    float totalX = tableBounds[table].Right + rowOffset;
                    float totalY = tableBounds[table].Bottom + colOffset + totalPaint.TextSize * 0.35f;
                    DrawText(canvas, totalText, totalX, totalY, totalPaint, totalShadow);
                }
            }
        }

        return visWarp;
    }

    private static void DrawText(SKCanvas canvas, string text, float x, float y, SKPaint paint, SKPaint shadow)
    {
        canvas.DrawText(text, x + 1, y + 1, shadow);
        canvas.DrawText(text, x, y, paint);
    }
}

using SkiaSharp;

namespace MLScoreSheet.Core;

public static partial class SheetScoreEngine
{
    private const int TablesPerRowBlock = 2;
    private const int RowsPerTable = 5;
    private const int ColumnsPerTable = 3;
    private const int TotalTables = TablesPerRowBlock * 2;
    private const int GroupsPerRowPair = ColumnsPerTable * TablesPerRowBlock;

    private static ScoreOverlayResult.OverlayDetails ComputeOverlayDetails(
        List<SKRectI> rects,
        IList<int> winnerIndices,
        List<Group> groups)
    {
        var slotByIndex = CreateSlotIndexMap(rects.Count, groups);

        var winners = new HashSet<int>(winnerIndices);
        var winnerMap = new bool[rects.Count];
        foreach (var idx in winnerIndices)
        {
            if (idx >= 0 && idx < winnerMap.Length)
                winnerMap[idx] = true;
        }

        var rowSums = new int[TotalTables, RowsPerTable];
        var colSums = new int[TotalTables, ColumnsPerTable];
        var tableTotals = new int[TotalTables];

        for (int gi = 0; gi < groups.Count; gi++)
        {
            var g = groups[gi];

            int rowPairIndex = gi / GroupsPerRowPair;
            int posInPair = gi % GroupsPerRowPair;

            int tableRowBlock = rowPairIndex / RowsPerTable;
            int rowInTable = rowPairIndex % RowsPerTable;
            int tableColBlock = posInPair / ColumnsPerTable;
            int colInTable = posInPair % ColumnsPerTable;

            if (tableRowBlock >= 2 || tableColBlock >= TablesPerRowBlock)
                continue;

            int tableIndex = tableRowBlock * TablesPerRowBlock + tableColBlock;
            if (tableIndex < 0 || tableIndex >= TotalTables)
                continue;

            bool hasWinner = false;
            int groupScore = 0;
            for (int s = 0; s < g.Indices.Length; s++)
            {
                int idx = g.Indices[s];
                if (idx < 0 || idx >= winnerMap.Length)
                    continue;

                if (winners.Contains(idx))
                {
                    hasWinner = true;
                    int slot = slotByIndex[idx];
                    if (slot >= 0)
                        groupScore = g.ValueOf(slot);
                    break;
                }
            }

            if (!hasWinner)
                continue;

            rowSums[tableIndex, rowInTable] += groupScore;
            colSums[tableIndex, colInTable] += groupScore;
            tableTotals[tableIndex] += groupScore;
        }

        return new ScoreOverlayResult.OverlayDetails
        {
            WinnerMap = winnerMap,
            RowSums = rowSums,
            ColumnSums = colSums,
            TableTotals = tableTotals
        };
    }

    private static int[] CreateSlotIndexMap(int rectCount, List<Group> groups)
    {
        var slotByIndex = Enumerable.Repeat(-1, rectCount).ToArray();
        for (int gi = 0; gi < groups.Count; gi++)
        {
            var g = groups[gi];
            for (int s = 0; s < g.Indices.Length; s++)
            {
                int idx = g.Indices[s];
                if (idx >= 0 && idx < slotByIndex.Length)
                    slotByIndex[idx] = s;
            }
        }

        return slotByIndex;
    }

    private static SKBitmap MakeWarpedOverlay(
        SKBitmap warped,
        List<SKRectI> rects,
        float[] pList,
        IList<int> winnerIndices,
        List<Group> groups,
        IReadOnlyList<SKPoint> fiducialsWarped,
        float visibilityThreshold,
        ScoreOverlayResult.OverlayDetails details)
    {
        var slotByIndex = CreateSlotIndexMap(rects.Count, groups);
        var winners = new HashSet<int>(winnerIndices);

        var rowBounds = new SKRect[TotalTables, RowsPerTable];
        var colBounds = new SKRect[TotalTables, ColumnsPerTable];
        var tableBounds = new SKRect[TotalTables];
        var rowHas = new bool[TotalTables, RowsPerTable];
        var colHas = new bool[TotalTables, ColumnsPerTable];
        var tableHas = new bool[TotalTables];

        var groupWidths = new List<float>();
        var groupHeights = new List<float>();

        var visWarp = warped.Copy();
        using (var c = new SKCanvas(visWarp))
        {
            var green = new SKPaint { Color = new SKColor(40, 200, 40), Style = SKPaintStyle.Stroke, StrokeWidth = 2f, IsAntialias = true };
            var red = new SKPaint { Color = new SKColor(230, 40, 40), Style = SKPaintStyle.Stroke, StrokeWidth = 2f, IsAntialias = true };
            var blue = new SKPaint { Color = new SKColor(70, 130, 240), Style = SKPaintStyle.Stroke, StrokeWidth = 2.5f, PathEffect = SKPathEffect.CreateDash(new float[] { 6, 6 }, 0), IsAntialias = true };
            var orangeStroke = new SKPaint { Color = new SKColor(255, 140, 0), Style = SKPaintStyle.Stroke, StrokeWidth = 3f, IsAntialias = true };
            var orangeFill = new SKPaint { Color = new SKColor(255, 140, 0, 160), Style = SKPaintStyle.Fill, IsAntialias = true };

            var txt = new SKPaint { Color = SKColors.Blue, TextSize = 30, IsAntialias = true, Typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Bold) };
            var shadow = new SKPaint { Color = new SKColor(0, 0, 0, 180), TextSize = 30, IsAntialias = true, Typeface = txt.Typeface };

            if (fiducialsWarped?.Count > 0)
            {
                const float radius = 14f;
                const float cross = 20f;
                foreach (var pt in fiducialsWarped)
                {
                    c.DrawCircle(pt, radius, orangeFill);
                    c.DrawCircle(pt, radius, orangeStroke);
                    c.DrawLine(pt.X - cross, pt.Y, pt.X + cross, pt.Y, orangeStroke);
                    c.DrawLine(pt.X, pt.Y - cross, pt.X, pt.Y + cross, orangeStroke);
                }
            }

            for (int gi = 0; gi < groups.Count; gi++)
            {
                var g = groups[gi];
                var rs = g.Indices.Select(i => rects[i]).ToArray();
                int minX = rs.Min(r => r.Left);
                int minY = rs.Min(r => r.Top);
                int maxX = rs.Max(r => r.Right);
                int maxY = rs.Max(r => r.Bottom);
                var pad = 3;
                var groupRect = new SKRect(minX - pad, minY - pad, maxX + pad, maxY + pad);
                c.DrawRect(groupRect, blue);

                if (GroupsPerRowPair <= 0) continue;

                int rowPairIndex = gi / GroupsPerRowPair;
                int posInPair = gi % GroupsPerRowPair;

                int tableRowBlock = rowPairIndex / RowsPerTable;
                int rowInTable = rowPairIndex % RowsPerTable;
                int tableColBlock = posInPair / ColumnsPerTable;
                int colInTable = posInPair % ColumnsPerTable;

                if (tableRowBlock >= 2 || tableColBlock >= TablesPerRowBlock)
                    continue;

                int tableIndex = tableRowBlock * TablesPerRowBlock + tableColBlock;
                if (tableIndex < 0 || tableIndex >= TotalTables)
                    continue;

                groupWidths.Add(groupRect.Width);
                groupHeights.Add(groupRect.Height);

                if (!rowHas[tableIndex, rowInTable])
                {
                    rowBounds[tableIndex, rowInTable] = groupRect;
                    rowHas[tableIndex, rowInTable] = true;
                }
                else
                    rowBounds[tableIndex, rowInTable] = SKRect.Union(rowBounds[tableIndex, rowInTable], groupRect);

                if (!colHas[tableIndex, colInTable])
                {
                    colBounds[tableIndex, colInTable] = groupRect;
                    colHas[tableIndex, colInTable] = true;
                }
                else
                    colBounds[tableIndex, colInTable] = SKRect.Union(colBounds[tableIndex, colInTable], groupRect);

                if (!tableHas[tableIndex])
                {
                    tableBounds[tableIndex] = groupRect;
                    tableHas[tableIndex] = true;
                }
                else
                    tableBounds[tableIndex] = SKRect.Union(tableBounds[tableIndex], groupRect);
            }

            for (int i = 0; i < rects.Count; i++)
            {
                var r = rects[i];
                bool isWin = winners.Contains(i);
                c.DrawRect(new SKRect(r.Left, r.Top, r.Right, r.Bottom), isWin ? green : red);

                string idxLabel = $"#{i}";
                string fillLabel = $"{Math.Round(pList[i] * 100)}%";
                bool showFill = pList[i] > visibilityThreshold;
                string slotLabel = slotByIndex[i] >= 0 ? slotByIndex[i].ToString() : "?";

                if (showFill)
                {
                    DrawText(c, idxLabel, r.Left - 35, r.Top - 5, txt, shadow);
                    DrawText(c, fillLabel, r.Right + 35 - txt.MeasureText(fillLabel), r.Bottom + 10, txt, shadow);
                }

                float cx = r.Left + r.Width * 0.5f;
                float cy = r.Top + r.Height * 0.55f;
                using var slotPaint = new SKPaint { Color = isWin ? green.Color : red.Color, TextSize = 40, IsAntialias = true, Typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Normal) };
                using var slotShadow = new SKPaint { Color = new SKColor(0, 0, 0, 200), TextSize = 40, IsAntialias = true, Typeface = slotPaint.Typeface };
                float sw = slotPaint.MeasureText(slotLabel);
                DrawText(c, slotLabel, cx - sw / 2, cy + 10, slotPaint, slotShadow);
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
                    Typeface = txt.Typeface
                };
                using var sumShadow = new SKPaint
                {
                    Color = new SKColor(0, 0, 0, 200),
                    TextSize = sumPaint.TextSize,
                    IsAntialias = true,
                    Typeface = txt.Typeface
                };
                using var totalPaint = new SKPaint
                {
                    Color = new SKColor(255, 255, 255),
                    TextSize = sumPaint.TextSize * 1.15f,
                    IsAntialias = true,
                    Typeface = txt.Typeface
                };
                using var totalShadow = new SKPaint
                {
                    Color = new SKColor(0, 0, 0, 220),
                    TextSize = totalPaint.TextSize,
                    IsAntialias = true,
                    Typeface = txt.Typeface
                };

                for (int table = 0; table < TotalTables; table++)
                {
                    if (!tableHas[table])
                        continue;

                    for (int r = 0; r < RowsPerTable; r++)
                    {
                        if (!rowHas[table, r])
                            continue;

                        int value = details.RowSums[table, r];
                        string text = value.ToString();
                        var bounds = rowBounds[table, r];
                        float x = bounds.Right + rowOffset;
                        float y = bounds.MidY + sumPaint.TextSize * 0.35f;
                        DrawText(c, text, x, y, sumPaint, sumShadow);
                    }

                    for (int col = 0; col < ColumnsPerTable; col++)
                    {
                        if (!colHas[table, col])
                            continue;

                        int value = details.ColumnSums[table, col];
                        string text = value.ToString();
                        var bounds = colBounds[table, col];
                        float textWidth = sumPaint.MeasureText(text);
                        float x = bounds.MidX - textWidth * 0.5f;
                        float y = tableBounds[table].Bottom + colOffset + sumPaint.TextSize * 0.35f;
                        DrawText(c, text, x, y, sumPaint, sumShadow);
                    }

                    int total = details.TableTotals[table];
                    string totalText = total.ToString();
                    float totalX = tableBounds[table].Right + rowOffset;
                    float totalY = tableBounds[table].Bottom + colOffset + totalPaint.TextSize * 0.35f;
                    DrawText(c, totalText, totalX, totalY, totalPaint, totalShadow);
                }
            }
        }

        return visWarp;
    }

    private static void DrawText(SKCanvas canvas, string s, float x, float y, SKPaint paint, SKPaint shadow)
    {
        canvas.DrawText(s, x + 1, y + 1, shadow);
        canvas.DrawText(s, x, y, paint);
    }
}

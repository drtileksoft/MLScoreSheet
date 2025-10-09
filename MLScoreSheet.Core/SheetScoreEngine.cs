using SkiaSharp;

namespace MLScoreSheet.Core;

public static partial class SheetScoreEngine
{
    public sealed class ScoreOverlayResult : IDisposable
    {
        public sealed class OverlayDetails
        {
            public bool[] WinnerMap { get; init; } = Array.Empty<bool>();
            public int[,] RowSums { get; init; } = new int[0, 0];
            public int[,] ColumnSums { get; init; } = new int[0, 0];
            public int[] TableTotals { get; init; } = Array.Empty<int>();
        }

        public int Total { get; init; }
        public float ThresholdUsed { get; init; }
        public SKBitmap? Overlay { get; init; }
        public OverlayDetails Details { get; init; } = new();

        public void Dispose() => Overlay?.Dispose();
    }

    public static async Task<ScoreOverlayResult> ComputeTotalScoreAsync(
        Stream photoStream,
        IResourceProvider resourceProvider,
        string yoloOnnxLogical = "best.onnx",
        string rectsJsonLogical = "boxes_rects.json",
        string fidPngLogical = "ocrscoresheetfiducials.png",
        float fixedThreshold = 0.35f,
        bool autoThreshold = false,
        float autoMin = 0.25f,
        float autoMax = 0.65f,
        float padFrac = 0.08f,
        float openFrac = 0.03f)
    {
        using var photo = DecodeLandscapePhoto(photoStream);
        var tpl = await LoadRectsJsonAsync(resourceProvider, rectsJsonLogical);

        var src = DetectSrcFidsWithOnnx(photo, resourceProvider, yoloOnnxLogical);
        var dst = await TryDetectDstFidsOrCornersAsync(resourceProvider, fidPngLogical, tpl.SizeW, tpl.SizeH);

        var H = ComputeHomography(src, dst);
        using var warped = WarpToTemplate(photo, H, tpl.SizeW, tpl.SizeH);

        var localContrastCalibration = new LocalContrastCalibration();
        var pList = new float[tpl.Rects.Count];
        for (int i = 0; i < tpl.Rects.Count; i++)
            pList[i] = (float)(GetFillPercentLocalContrast(warped, tpl.Rects[i], localContrastCalibration) / 100.0);

        float thr = autoThreshold ? AutoThresholdKMeans(pList, autoMin, autoMax, fixedThreshold, 0.12f)
                                  : fixedThreshold;

        var res = ScoreSelector3x2.SumWinnerTakesAll(tpl.Rects, pList, thr);
        var groups = BuildGroupsGrid3x2(tpl.Rects, pList);
        var details = ComputeOverlayDetails(tpl.Rects, res.WinnerIndices, groups);

        return new ScoreOverlayResult
        {
            Total = res.Total,
            ThresholdUsed = thr,
            Overlay = null,
            Details = details
        };
    }

    public static async Task<ScoreOverlayResult> ComputeTotalScoreWithOverlayAsync(
        Stream photoStream,
        IResourceProvider resourceProvider,
        string yoloOnnxLogical = "best.onnx",
        string rectsJsonLogical = "boxes_rects.json",
        string fidPngLogical = "ocrscoresheetfiducials.png",
        float fixedThreshold = 0.35f,
        bool autoThreshold = false,
        float autoMin = 0.25f,
        float autoMax = 0.65f,
        float padFrac = 0.08f,
        float openFrac = 0.03f,
        float overlayVisibilityThreshold = 0.30f)
    {
        using var photo = DecodeLandscapePhoto(photoStream);
        var tpl = await LoadRectsJsonAsync(resourceProvider, rectsJsonLogical);

        var src = DetectSrcFidsWithOnnx(photo, resourceProvider, yoloOnnxLogical);
        var dst = await TryDetectDstFidsOrCornersAsync(resourceProvider, fidPngLogical, tpl.SizeW, tpl.SizeH);

        var H = ComputeHomography(src, dst);
        var warped = WarpToTemplate(photo, H, tpl.SizeW, tpl.SizeH);

        var fidWarped = new[]
        {
            ApplyHomography(H, src.TL),
            ApplyHomography(H, src.TR),
            ApplyHomography(H, src.BR),
            ApplyHomography(H, src.BL)
        };

        var localContrastCalibration = new LocalContrastCalibration();
        var pList = new float[tpl.Rects.Count];
        for (int i = 0; i < tpl.Rects.Count; i++)
            pList[i] = (float)(GetFillPercentLocalContrast(warped, tpl.Rects[i], localContrastCalibration) / 100.0);

        float thr = autoThreshold ? AutoThresholdKMeans(pList, autoMin, autoMax, fixedThreshold, 0.12f)
                                  : fixedThreshold;

        var res = ScoreSelector3x2.SumWinnerTakesAll(tpl.Rects, pList, thr);
        var groups = BuildGroupsGrid3x2(tpl.Rects, pList);
        var details = ComputeOverlayDetails(tpl.Rects, res.WinnerIndices, groups);

        var overlay = MakeWarpedOverlay(
            warped,
            tpl.Rects,
            pList,
            res.WinnerIndices,
            groups,
            fidWarped,
            overlayVisibilityThreshold,
            details
        );

        return new ScoreOverlayResult
        {
            Total = res.Total,
            ThresholdUsed = thr,
            Overlay = overlay,
            Details = details
        };
    }
}

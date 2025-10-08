using System;
using System.IO;
using System.Threading.Tasks;
using MLScoreSheetCounter;
using SkiaSharp;

namespace YourApp.Services;

public static class SheetScoreEngine
{
    public sealed class ScoreOverlayResult : IDisposable
    {
        public int Total { get; init; }
        public float ThresholdUsed { get; init; }
        public SKBitmap Overlay { get; init; } = default!;

        public void Dispose() => Overlay?.Dispose();
    }

    public static async Task<int> ComputeTotalScoreAsync(
        Stream photoStream,
        string yoloOnnxLogical = "yolo_fids.onnx",
        string rectsJsonLogical = "boxes_rects.json",
        string fidPngLogical = "ocrscoresheetfiducials.png",
        float fixedThreshold = 0.35f,
        bool autoThreshold = false,
        float autoMin = 0.25f,
        float autoMax = 0.65f,
        float padFrac = 0.08f,
        float openFrac = 0.03f)
    {
        using var photo = BitmapPreprocessor.DecodeLandscapePhoto(photoStream);
        var template = await TemplateRepository.LoadAsync(rectsJsonLogical).ConfigureAwait(false);

        var sourceFids = FiducialDetector.DetectSourceFiducials(photo, yoloOnnxLogical);
        var destinationFids = await FiducialDetector.DetectDestinationFiducialsAsync(fidPngLogical, template.SizeW, template.SizeH).ConfigureAwait(false);

        var homography = HomographyCalculator.Compute(sourceFids, destinationFids);
        using var warped = HomographyCalculator.WarpToTemplate(photo, homography, template.SizeW, template.SizeH);

        var fillRatios = FillRatioCalculator.ComputeRatios(warped, template.Rects, padFrac, openFrac);
        float threshold = AutoThresholdCalculator.SelectThreshold(fillRatios, autoThreshold, autoMin, autoMax, fixedThreshold);

        return GroupLayoutBuilder.TotalScoreFromItems(template.Rects, fillRatios, threshold);
    }

    public static async Task<ScoreOverlayResult> ComputeTotalScoreWithOverlayAsync(
        Stream photoStream,
        string yoloOnnxLogical = "yolo_fids.onnx",
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
        using var photo = BitmapPreprocessor.DecodeLandscapePhoto(photoStream);
        var template = await TemplateRepository.LoadAsync(rectsJsonLogical).ConfigureAwait(false);

        var sourceFids = FiducialDetector.DetectSourceFiducials(photo, yoloOnnxLogical);
        var destinationFids = await FiducialDetector.DetectDestinationFiducialsAsync(fidPngLogical, template.SizeW, template.SizeH).ConfigureAwait(false);

        var homography = HomographyCalculator.Compute(sourceFids, destinationFids);
        using var warped = HomographyCalculator.WarpToTemplate(photo, homography, template.SizeW, template.SizeH);

        var fillRatios = FillRatioCalculator.ComputeLocalContrastRatios(warped, template.Rects);
        float threshold = AutoThresholdCalculator.SelectThreshold(fillRatios, autoThreshold, autoMin, autoMax, fixedThreshold);

        var scoringResult = ScoreSelector3x2.SumWinnerTakesAll(template.Rects, fillRatios, threshold);
        var groups = GroupLayoutBuilder.BuildGroupsGrid3x2(template.Rects, fillRatios);
        var overlay = OverlayRenderer.CreateWarpedOverlay(warped, template.Rects, fillRatios, scoringResult.WinnerIndices, groups, overlayVisibilityThreshold);

        return new ScoreOverlayResult
        {
            Total = scoringResult.Total,
            ThresholdUsed = threshold,
            Overlay = overlay
        };
    }
}

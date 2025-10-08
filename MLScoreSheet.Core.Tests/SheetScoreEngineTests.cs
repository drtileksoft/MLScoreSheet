using MLScoreSheet.Core;
using MLScoreSheet.Core.Tests;
using System.IO;
using System.Text.Json;
using Xunit;

public sealed class SheetScoreEngineTests
{
    private readonly TestResourceProvider _resourceProvider = new();

    [Fact]
    public async Task ComputeTotalScoreAsync_ReturnsExpectedScore_ForPhoto2()
    {
        var photoPath = TestResourceProvider.GetAssetPath("photo.jpeg");
        using var photoStream = File.OpenRead(photoPath);

        var total = await SheetScoreEngine.ComputeTotalScoreAsync(
            photoStream,
            _resourceProvider,
            fixedThreshold: 0.30f);

        Assert.Equal(43, total);
    }

    [Fact]
    public async Task ComputeTotalScoreWithOverlayAsync_ProducesOverlayAndThreshold_ForPhoto2()
    {
        var photoPath = TestResourceProvider.GetAssetPath("photo.jpeg");
        using var photoStream = File.OpenRead(photoPath);

        var result = await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
            photoStream,
            _resourceProvider,
            fixedThreshold: 0.30f,
            overlayVisibilityThreshold: 0.24f);

        Assert.Equal(43, result.Total);
        Assert.Equal(0.30f, result.ThresholdUsed, precision: 3);
        Assert.NotNull(result.Overlay);
        Assert.True(result.Overlay.Width > 0);
        Assert.True(result.Overlay.Height > 0);
    }

    [Fact]
    public async Task ComputeTotalScoreWithOverlayAsync_ReturnsExpectedOverlayDetails_ForPhoto2()
    {
        var photoPath = TestResourceProvider.GetAssetPath("photo2.jpeg");
        using var photoStream = File.OpenRead(photoPath);

        var result = await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
            photoStream,
            _resourceProvider,
            fixedThreshold: 0.25f,
            overlayVisibilityThreshold: 0.24f);

        var details = result.Details;
        var expected = LoadOverlayExpectations("photo_overlay_expected.json");

        // Uncomment to update the snapshot file if the expectations change
        //OverlayExpectationsIo.SaveDetailsSnapshot(result.Details, "photo_overlay_expected.json");

        Assert.Equal(expected.WinnerMap.Length, details.WinnerMap.Length);
        for (int i = 0; i < details.WinnerMap.Length; i++)
        {
            Assert.Equal(expected.WinnerMap[i] == 1, details.WinnerMap[i]);
        }

        AssertMatrixEqual(expected.RowSums, details.RowSums);
        AssertMatrixEqual(expected.ColumnSums, details.ColumnSums);
        Assert.Equal(expected.TableTotals, details.TableTotals);
    }

    private static OverlayExpectations LoadOverlayExpectations(string fileName)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Assets", fileName);
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<OverlayExpectations>(json) ?? new OverlayExpectations();
    }

    private static void AssertMatrixEqual(int[][] expected, int[,] actual)
    {
        Assert.Equal(expected.Length, actual.GetLength(0));
        for (int r = 0; r < expected.Length; r++)
        {
            Assert.Equal(expected[r].Length, actual.GetLength(1));
            for (int c = 0; c < expected[r].Length; c++)
            {
                Assert.Equal(expected[r][c], actual[r, c]);
            }
        }
    }


    private sealed class TestResourceProvider : IResourceProvider
    {
        private readonly string _assetDirectory;

        public TestResourceProvider()
        {
            _assetDirectory = Path.Combine(AppContext.BaseDirectory, "Assets");
        }

        public Task<Stream> OpenReadAsync(string logicalName)
        {
            var path = Path.Combine(_assetDirectory, logicalName);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Asset '{logicalName}' not found at '{path}'.");
            }

            Stream stream = File.OpenRead(path);
            return Task.FromResult(stream);
        }

        public static string GetAssetPath(string logicalName)
        {
            var baseDirectory = Path.Combine(AppContext.BaseDirectory, "Assets");
            return Path.Combine(baseDirectory, logicalName);
        }
    }
}

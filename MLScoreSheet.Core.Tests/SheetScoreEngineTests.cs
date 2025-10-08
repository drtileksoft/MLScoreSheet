using System.IO;
using MLScoreSheet.Core;
using Xunit;

public sealed class SheetScoreEngineTests
{
    private readonly TestResourceProvider _resourceProvider = new();

    [Fact]
    public async Task ComputeTotalScoreAsync_ReturnsExpectedScore_ForPhoto2()
    {
        var photoPath = TestResourceProvider.GetAssetPath("photo2.jpeg");
        using var photoStream = File.OpenRead(photoPath);

        var total = await SheetScoreEngine.ComputeTotalScoreAsync(
            photoStream,
            _resourceProvider,
            fixedThreshold: 0.25f,
            autoThreshold: false,
            padFrac: 0.08f,
            openFrac: 0.03f);

        Assert.Equal(43, total);
    }

    [Fact]
    public async Task ComputeTotalScoreWithOverlayAsync_ProducesOverlayAndThreshold_ForPhoto2()
    {
        var photoPath = TestResourceProvider.GetAssetPath("photo2.jpeg");
        using var photoStream = File.OpenRead(photoPath);

        await using var result = await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
            photoStream,
            _resourceProvider,
            fixedThreshold: 0.25f,
            autoThreshold: false,
            overlayVisibilityThreshold: 0.24f);

        Assert.Equal(43, result.Total);
        Assert.Equal(0.25f, result.ThresholdUsed, precision: 3);
        Assert.NotNull(result.Overlay);
        Assert.True(result.Overlay.Width > 0);
        Assert.True(result.Overlay.Height > 0);
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

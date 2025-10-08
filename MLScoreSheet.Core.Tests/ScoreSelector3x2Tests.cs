using MLScoreSheet.Core;
using SkiaSharp;
using Xunit;

public class ScoreSelector3x2Tests
{
    [Fact]
    public void SumWinnerTakesAll_ReturnsZeroWhenNoData()
    {
        var result = ScoreSelector3x2.SumWinnerTakesAll(Array.Empty<SKRectI>(), Array.Empty<float>(), 0.5f);

        Assert.Equal(0, result.Total);
        Assert.Equal(0.5f, result.ThresholdUsed);
        Assert.Empty(result.WinnerIndices);
    }

    [Fact]
    public void SumWinnerTakesAll_SelectsHighestPerGroup()
    {
        var rects = new List<SKRectI>
        {
            new(0, 0, 10, 10), new(10, 0, 20, 10), new(20, 0, 30, 10),
            new(0, 10, 10, 20), new(10, 10, 20, 20), new(20, 10, 30, 20)
        };

        var probs = new List<float> { 0.2f, 0.9f, 0.7f, 0.1f, 0.8f, 0.6f };

        var result = ScoreSelector3x2.SumWinnerTakesAll(rects, probs, 0.5f);

        Assert.Equal(1, result.Total);
        Assert.Equal(0.5f, result.ThresholdUsed);
        Assert.Contains(1, result.WinnerIndices);
        Assert.Equal(1, result.WinnerIndices.Count);
    }

    [Fact]
    public void SumWinnerTakesAll_ReturnsZeroWhenCountsDiffer()
    {
        var rects = new List<SKRectI> { new(0, 0, 10, 10) };
        var probs = new List<float>();

        var result = ScoreSelector3x2.SumWinnerTakesAll(rects, probs, 0.4f);

        Assert.Equal(0, result.Total);
        Assert.Equal(0.4f, result.ThresholdUsed);
        Assert.Empty(result.WinnerIndices);
    }

    //[Fact]
    //public void SumWinnerTakesAll_RespectsGeometryWhenRectsUnordered()
    //{
    //    var rects = new List<SKRectI>
    //    {
    //        new(0, 20, 10, 30), // bottom left
    //        new(0, 0, 10, 10),  // top left
    //        new(20, 20, 30, 30),
    //        new(20, 0, 30, 10),
    //        new(40, 20, 50, 30),
    //        new(40, 0, 50, 10)
    //    };

    //    var probs = new List<float> { 0.95f, 0.1f, 0.96f, 0.7f, 0.7f, 0.85f };

    //    var result = ScoreSelector3x2.SumWinnerTakesAll(rects, probs, 0.5f);

    //    Assert.Equal(9, result.Total);
    //    Assert.Contains(0, result.WinnerIndices); // bottom-left
    //    Assert.Contains(2, result.WinnerIndices); // bottom-middle
    //    Assert.Contains(5, result.WinnerIndices); // top-right
    //    Assert.Equal(3, result.WinnerIndices.Count);
    //}
}

using System.Collections.Generic;
using SkiaSharp;
using Xunit;

namespace MLScoreSheetCounter.Tests;

public class ScoreSelector3x2Tests
{
    [Fact]
    public void SumWinnerTakesAll_ReturnsZeroWhenInputInvalid()
    {
        var result = ScoreSelector3x2.SumWinnerTakesAll(null!, null!, 0.5f);
        Assert.Equal(0, result.Total);
        Assert.Empty(result.WinnerIndices);
    }

    [Fact]
    public void SumWinnerTakesAll_SelectsHighestConfidenceAcrossRows()
    {
        var rects = new List<SKRectI>
        {
            new(0, 0, 10, 10),
            new(10, 0, 20, 10),
            new(20, 0, 30, 10),
            new(0, 30, 10, 40),
            new(10, 30, 20, 40),
            new(20, 30, 30, 40)
        };

        var probabilities = new List<float> { 0.9f, 0.2f, 0.3f, 0.1f, 0.8f, 0.7f };

        var result = ScoreSelector3x2.SumWinnerTakesAll(rects, probabilities, 0.5f);

        Assert.Equal(4, result.Total);
        Assert.Single(result.WinnerIndices);
        Assert.Contains(4, result.WinnerIndices);
    }

    [Fact]
    public void SumWinnerTakesAll_IgnoresIncompleteGroups()
    {
        var rects = new List<SKRectI>
        {
            new(0, 0, 10, 10),
            new(10, 0, 20, 10),
            new(20, 0, 30, 10),
            new(30, 0, 40, 10),
            new(40, 0, 50, 10),
            new(50, 0, 60, 10),
            new(0, 30, 10, 40),
            new(10, 30, 20, 40),
            new(20, 30, 30, 40)
        };

        var probabilities = new List<float> { 0.1f, 0.6f, 0.2f, 0.9f, 0.8f, 0.4f, 0.3f, 0.95f, 0.1f };

        var result = ScoreSelector3x2.SumWinnerTakesAll(rects, probabilities, 0.5f);

        Assert.Equal(4, result.Total);
        Assert.Single(result.WinnerIndices);
        Assert.Equal(7, result.WinnerIndices[0]);
    }
}

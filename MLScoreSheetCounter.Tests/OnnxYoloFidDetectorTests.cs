using System.Collections.Generic;
using Xunit;

namespace MLScoreSheetCounter.Tests;

public class OnnxYoloFidDetectorTests
{
    [Fact]
    public void IoUxywh_ReturnsExpectedForOverlappingBoxes()
    {
        var a = new OnnxYoloFidDetector.Det { Cx = 10, Cy = 10, W = 10, H = 10 };
        var b = new OnnxYoloFidDetector.Det { Cx = 12, Cy = 12, W = 10, H = 10 };

        var iou = OnnxYoloFidDetector.IoUxywh(a, b);

        Assert.InRange(iou, 0.39f, 0.41f);
    }

    [Fact]
    public void IoUxywh_ReturnsZeroForNonOverlappingBoxes()
    {
        var a = new OnnxYoloFidDetector.Det { Cx = 0, Cy = 0, W = 4, H = 4 };
        var b = new OnnxYoloFidDetector.Det { Cx = 20, Cy = 20, W = 4, H = 4 };

        var iou = OnnxYoloFidDetector.IoUxywh(a, b);

        Assert.Equal(0f, iou);
    }

    [Fact]
    public void DebugListOutputs_FormatsMetadata()
    {
        var metadata = new[]
        {
            new KeyValuePair<string, IReadOnlyList<long?>>("output0", new long?[] { 1, 3, 640, 640 }),
            new KeyValuePair<string, IReadOnlyList<long?>>("output1", new long?[] { 1, null, 10 })
        };

        var text = OnnxYoloFidDetector.DebugListOutputs(metadata);

        var expected = "output0: [1,3,640,640]" + System.Environment.NewLine +
                       "output1: [1,,10]" + System.Environment.NewLine;

        Assert.Equal(expected, text);
    }
}

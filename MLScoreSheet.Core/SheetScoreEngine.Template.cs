namespace MLScoreSheet.Core;

public static partial class SheetScoreEngine
{
    private sealed class TemplateData
    {
        public int SizeW { get; init; }
        public int SizeH { get; init; }
        public List<SkiaSharp.SKRectI> Rects { get; init; } = new();
    }

    private static async Task<TemplateData> LoadRectsJsonAsync(IResourceProvider resourceProvider, string logicalName)
    {
        using var stream = await resourceProvider.OpenReadAsync(logicalName);
        using var memoryStream = new MemoryStream();
        await stream.CopyToAsync(memoryStream);
        memoryStream.Position = 0;

        using var doc = await System.Text.Json.JsonDocument.ParseAsync(memoryStream);
        var root = doc.RootElement;

        var size = root.GetProperty("size").EnumerateArray().ToArray();
        int width = size[0].GetInt32();
        int height = size[1].GetInt32();

        var rects = new List<SkiaSharp.SKRectI>();
        foreach (var rect in root.GetProperty("rects").EnumerateArray())
        {
            var values = rect.EnumerateArray().ToArray();
            int x = values[0].GetInt32();
            int y = values[1].GetInt32();
            int w = values[2].GetInt32();
            int h = values[3].GetInt32();
            rects.Add(new SkiaSharp.SKRectI(x, y, x + w, y + h));
        }

        return new TemplateData { SizeW = width, SizeH = height, Rects = rects };
    }
}

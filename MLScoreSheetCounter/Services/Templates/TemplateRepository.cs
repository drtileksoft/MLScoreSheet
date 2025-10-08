using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Maui.Storage;
using SkiaSharp;

namespace YourApp.Services;

internal sealed class TemplateData
{
    public int SizeW { get; init; }
    public int SizeH { get; init; }
    public List<SKRectI> Rects { get; init; } = new();
}

internal static class TemplateRepository
{
    public static async Task<TemplateData> LoadAsync(string logicalName)
    {
        using var stream = await FileSystem.OpenAppPackageFileAsync(logicalName);
        using var memory = new MemoryStream();
        await stream.CopyToAsync(memory).ConfigureAwait(false);
        memory.Position = 0;

        using var document = await System.Text.Json.JsonDocument.ParseAsync(memory).ConfigureAwait(false);
        var root = document.RootElement;

        var size = root.GetProperty("size").EnumerateArray().ToArray();
        int width = size[0].GetInt32();
        int height = size[1].GetInt32();

        var rects = new List<SKRectI>();
        foreach (var element in root.GetProperty("rects").EnumerateArray())
        {
            var values = element.EnumerateArray().ToArray();
            int x = values[0].GetInt32();
            int y = values[1].GetInt32();
            int w = values[2].GetInt32();
            int h = values[3].GetInt32();
            rects.Add(new SKRectI(x, y, x + w, y + h));
        }

        return new TemplateData
        {
            SizeW = width,
            SizeH = height,
            Rects = rects
        };
    }
}

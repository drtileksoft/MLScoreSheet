using System.Globalization;
using MLScoreSheet.Core;
using SkiaSharp;

try
{
    //if (args.Length == 0)
    //{
    //    PrintUsage();
    //    return 1;
    //}

    string? inputPath = "photo2.jpeg";
    string? outputPath = "output.jpg";
    float? calculationThreshold = 0.25f;
    float? overlayThreshold = 0.24f;
    bool autoThreshold = false;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--input":
                inputPath = ReadNext(args, ref i);
                break;
            case "--output":
                outputPath = ReadNext(args, ref i);
                break;
            case "--calc-threshold":
                calculationThreshold = ParseFloat(ReadNext(args, ref i));
                break;
            case "--overlay-threshold":
                overlayThreshold = ParseFloat(ReadNext(args, ref i));
                break;
            case "--auto-threshold":
                autoThreshold = true;
                break;
            case "--help":
            case "-h":
                PrintUsage();
                return 0;
            default:
                throw new ArgumentException($"Unknown argument: {args[i]}");
                break;
        }
    }

    if (string.IsNullOrWhiteSpace(inputPath))
    {
        throw new ArgumentException("Missing path to the input photo.");
    }

    if (!File.Exists(inputPath))
    {
        throw new FileNotFoundException($"File {inputPath} does not exist.", inputPath);
    }

    outputPath ??= Path.ChangeExtension(inputPath, ".overlay.png");
    float calcThr = calculationThreshold ?? 0.35f;
    float overlayThr = overlayThreshold ?? 0.30f;

    var provider = new FileResourceProvider(Path.Combine(AppContext.BaseDirectory, "Assets"));

    await using var photoStream = File.OpenRead(inputPath);
    using var result = await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
        photoStream,
        provider,
        fixedThreshold: calcThr,
        autoThreshold: autoThreshold,
        overlayVisibilityThreshold: overlayThr);

    Console.WriteLine($"Total score: {result.Total}");
    Console.WriteLine($"Applied threshold: {result.ThresholdUsed:F2}");

    Console.WriteLine($"Saving overlay to {outputPath}...");
    SaveOverlay(result.Overlay, outputPath);
    Console.WriteLine($"Overlay saved to: {outputPath}");
    return 0;
}
catch (ArgumentException ex)
{
    Console.Error.WriteLine($"Argument error: {ex.Message}");
    PrintUsage();
    return 1;
}
catch (FileNotFoundException ex)
{
    Console.Error.WriteLine(ex.Message);
    return 1;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 2;
}

static string ReadNext(string[] args, ref int index)
{
    if (index + 1 >= args.Length)
    {
        throw new ArgumentException($"Argument {args[index]} requires a value.");
    }

    index++;
    return args[index];
}

static float ParseFloat(string value)
{
    if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float result))
    {
        throw new ArgumentException($"Invalid numeric value: {value}");
    }

    return result;
}

static void SaveOverlay(SKBitmap overlay, string outputPath)
{
    using var image = SKImage.FromBitmap(overlay);
    using var data = image.Encode(SKEncodedImageFormat.Png, 95);
    using var fileStream = File.Create(outputPath);
    data.SaveTo(fileStream);
}

static void PrintUsage()
{
    Console.WriteLine("Usage: mlscoresheet-console [--input <file>] [--output <file>] [--calc-threshold <value>] [--overlay-threshold <value>] [--auto-threshold]");
}

internal sealed class FileResourceProvider : IResourceProvider
{
    private readonly string _baseDirectory;

    public FileResourceProvider(string baseDirectory)
    {
        _baseDirectory = baseDirectory;
    }

    public Task<Stream> OpenReadAsync(string logicalName)
    {
        var path = Path.Combine(_baseDirectory, logicalName);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Resource {logicalName} was not found.", path);
        }

        Stream stream = File.OpenRead(path);
        return Task.FromResult(stream);
    }
}

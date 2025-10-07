using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Android.Content;
using Android.Media;
using Android.OS;
using Android.Provider;
using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Storage;
using Environment = Android.OS.Environment;

namespace MLScoreSheetCounter.Services;

public partial class GallerySaver : IGallerySaver
{
    private const string AlbumName = "MLScoreSheet";

    public async Task SaveImageAsync(string filePath, string fileName, CancellationToken cancellationToken = default)
    {
        var context = Platform.CurrentActivity ?? Platform.AppContext;
        var mimeType = GetMimeType(fileName);

        if (Build.VERSION.SdkInt >= BuildVersionCodes.Q)
        {
            await SaveWithMediaStoreAsync(context, filePath, fileName, mimeType, cancellationToken);
        }
        else
        {
            await SaveLegacyAsync(context, filePath, fileName, cancellationToken);
        }
    }

    private static async Task SaveWithMediaStoreAsync(Context context, string filePath, string fileName, string mimeType, CancellationToken cancellationToken)
    {
        var values = new ContentValues();
        values.Put(MediaStore.MediaColumns.DisplayName, fileName);
        values.Put(MediaStore.MediaColumns.MimeType, mimeType);
        values.Put(MediaStore.MediaColumns.RelativePath, $"{Environment.DirectoryPictures}/{AlbumName}");
        values.Put(MediaStore.MediaColumns.IsPending, 1);

        var resolver = context.ContentResolver;
        var uri = resolver.Insert(MediaStore.Images.Media.ExternalContentUri, values);
        if (uri == null)
        {
            throw new InvalidOperationException("Nepodařilo se vytvořit položku v galerii.");
        }

        await using (var input = File.OpenRead(filePath))
        await using (var output = resolver.OpenOutputStream(uri) ?? throw new InvalidOperationException("Nelze otevřít výstupní proud."))
        {
            await input.CopyToAsync(output, cancellationToken);
        }

        values.Put(MediaStore.MediaColumns.IsPending, 0);
        resolver.Update(uri, values, null, null);
    }

    private static async Task SaveLegacyAsync(Context context, string filePath, string fileName, CancellationToken cancellationToken)
    {
        var picturesPath = Environment.GetExternalStoragePublicDirectory(Environment.DirectoryPictures)?.AbsolutePath
            ?? FileSystem.Current.CacheDirectory;
        var directory = Path.Combine(picturesPath, AlbumName);
        Directory.CreateDirectory(directory);

        var destinationPath = Path.Combine(directory, fileName);
        await using (var input = File.OpenRead(filePath))
        await using (var output = File.Create(destinationPath))
        {
            await input.CopyToAsync(output, cancellationToken);
        }

        MediaScannerConnection.ScanFile(context, new[] { destinationPath }, null, null);
    }

    private static string GetMimeType(string fileName)
    {
        var extension = Path.GetExtension(fileName).ToLowerInvariant();
        return extension switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            _ => "image/jpeg"
        };
    }
}

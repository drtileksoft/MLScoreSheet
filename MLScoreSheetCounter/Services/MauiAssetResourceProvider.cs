using System.IO;
using Microsoft.Maui.Storage;
using MLScoreSheet.Core;

namespace MLScoreSheetCounter.Services;

public sealed class MauiAssetResourceProvider : IResourceProvider
{
    public Task<Stream> OpenReadAsync(string logicalName)
    {
        return FileSystem.Current.OpenAppPackageFileAsync(logicalName);
    }
}

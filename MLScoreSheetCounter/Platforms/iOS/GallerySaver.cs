using System;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using MLScoreSheetCounter.Services;
using Photos;

namespace MLScoreSheetCounter.Services;

public partial class GallerySaver : IGallerySaver
{
    public async Task SaveImageAsync(string filePath, string fileName, CancellationToken cancellationToken = default)
    {
        var status = PHPhotoLibrary.AuthorizationStatusForAccessLevel(PHAccessLevel.AddOnly);
        if (status == PHAuthorizationStatus.NotDetermined)
        {
            status = await PHPhotoLibrary.RequestAuthorizationAsync(PHAccessLevel.AddOnly);
        }

        if (status is not (PHAuthorizationStatus.Authorized or PHAuthorizationStatus.Limited))
        {
            throw new InvalidOperationException("Aplikace nemá oprávnění ukládat do fotogalerie.");
        }

        var url = NSUrl.FromFilename(filePath);
        if (url == null)
        {
            throw new InvalidOperationException("Nelze otevřít soubor pro uložení.");
        }

        await PHPhotoLibrary.SharedPhotoLibrary.PerformChangesAsync(() =>
        {
            var request = PHAssetCreationRequest.CreationRequestForAsset();
            request.AddResource(PHAssetResourceType.Photo, url, new PHAssetResourceCreationOptions());
        });
    }
}

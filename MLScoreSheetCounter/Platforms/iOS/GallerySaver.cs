using System;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using MLScoreSheetCounter.Platforms.iOS;
using MLScoreSheetCounter.Services;
using Photos;

namespace MLScoreSheetCounter.Services;

public partial class GallerySaver : IGallerySaver
{
    public async Task SaveImageAsync(string filePath, string fileName, CancellationToken cancellationToken = default)
    {
        var status = PHPhotoLibrary.GetAuthorizationStatus(PHAccessLevel.AddOnly);
        if (status == PHAuthorizationStatus.NotDetermined)
        {
            status = await PHPhotoLibrary.RequestAuthorizationAsync(PHAccessLevel.AddOnly);
        }

        if (status is not (PHAuthorizationStatus.Authorized or PHAuthorizationStatus.Limited))
        {
            throw new InvalidOperationException("The app does not have permission to save to the photo library.");
        }

        var url = NSUrl.FromFilename(filePath);
        if (url == null)
        {
            throw new InvalidOperationException("Unable to open the file for saving.");
        }


        await PhotoSaver.SavePhotoFileAsync(url);

    }
}

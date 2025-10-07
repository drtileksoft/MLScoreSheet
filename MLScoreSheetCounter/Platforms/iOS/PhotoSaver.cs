using System;
using System.IO;
using System.Threading.Tasks;
using Foundation;
using Photos;
using UIKit;

namespace MLScoreSheetCounter.Platforms.iOS
{
    /// <summary>
    /// Helpery pro ukládání fotek do Fotek (Photos) s podporou async.
    /// </summary>
    public static class PhotoSaver
    {
        // --- 1) Async wrapper pro PHPhotoLibrary.PerformChanges ---
        public static Task<bool> PerformChangesAsync(this PHPhotoLibrary library, Action changeHandler)
        {
            if (library is null) throw new ArgumentNullException(nameof(library));
            if (changeHandler is null) throw new ArgumentNullException(nameof(changeHandler));

            var tcs = new TaskCompletionSource<bool>();

            library.PerformChanges(
                changeHandler,
                (success, error) =>
                {
                    if (error != null)
                    {
                        tcs.TrySetException(new NSErrorException(error));
                    }
                    else
                    {
                        tcs.TrySetResult(success);
                    }
                });

            return tcs.Task;
        }

        // --- 2) Zajištění oprávnění: iOS 14+ používá AccessLevel ---
        public static async Task EnsureAddOnlyAuthorizationAsync()
        {
            var status = PHPhotoLibrary.GetAuthorizationStatus(PHAccessLevel.AddOnly);
            if (status == PHAuthorizationStatus.Authorized)
                return;

            status = await RequestAddOnlyAuthorizationAsync();
            if (status != PHAuthorizationStatus.Authorized)
                throw new UnauthorizedAccessException("Přístup k ukládání do Fotek nebyl povolen.");
        }

        private static Task<PHAuthorizationStatus> RequestAddOnlyAuthorizationAsync()
        {
            var tcs = new TaskCompletionSource<PHAuthorizationStatus>();
            PHPhotoLibrary.RequestAuthorization(PHAccessLevel.AddOnly, s => tcs.TrySetResult(s));
            return tcs.Task;
        }

        // --- 3) Uložení fotky ze souboru (NSUrl) ---
        /// <summary>
        /// Uloží fotku (soubor) do knihovny Fotek.
        /// </summary>
        /// <param name="fileUrl">NSUrl na lokální soubor (např. v /tmp nebo /Documents).</param>
        /// <param name="shouldMoveFile">
        /// true = soubor se po přidání přesune (musí být v místě, kam má app právo zapisovat/měnit),
        /// false = zůstane a data se zkopírují.
        /// </param>
        public static async Task SavePhotoFileAsync(NSUrl fileUrl, bool shouldMoveFile = false)
        {
            if (fileUrl is null) throw new ArgumentNullException(nameof(fileUrl));

            await EnsureAddOnlyAuthorizationAsync();

            var opts = new PHAssetResourceCreationOptions
            {
                ShouldMoveFile = shouldMoveFile
            };

            await PHPhotoLibrary.SharedPhotoLibrary.PerformChangesAsync(() =>
            {
                var req = PHAssetCreationRequest.CreationRequestForAsset();
                req.AddResource(PHAssetResourceType.Photo, fileUrl, opts);
            });
        }

        // --- 4) Uložení fotky z paměti (NSData/byte[]/UIImage) ---
        public static async Task SavePhotoDataAsync(NSData data)
        {
            if (data is null) throw new ArgumentNullException(nameof(data));

            await EnsureAddOnlyAuthorizationAsync();

            await PHPhotoLibrary.SharedPhotoLibrary.PerformChangesAsync(() =>
            {
                var req = PHAssetCreationRequest.CreationRequestForAsset();
                var opts = new PHAssetResourceCreationOptions();
                req.AddResource(PHAssetResourceType.Photo, data, opts);
            });
        }

        public static Task SavePhotoBytesAsync(byte[] bytes) =>
            SavePhotoDataAsync(NSData.FromArray(bytes));

        public static async Task SaveUIImageAsync(UIImage image, nfloat jpegQuality)
        {
            if (image is null) throw new ArgumentNullException(nameof(image));

            // Ulož jako JPEG (můžeš změnit na PNG: image.AsPNG())
            using var data = image.AsJPEG(jpegQuality);
            await SavePhotoDataAsync(data);
        }

        // --- 5) Pomocník: uloží stream do /tmp a pak do Fotek ---
        public static async Task SaveStreamAsync(Stream stream, string fileName = "photo.jpg", bool shouldMoveFile = true)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));

            var tempPath = Path.Combine(Path.GetTempPath(), fileName);
            using (var fs = File.Create(tempPath))
                await stream.CopyToAsync(fs);

            var url = NSUrl.FromFilename(tempPath);
            await SavePhotoFileAsync(url, shouldMoveFile);
        }
    }
}

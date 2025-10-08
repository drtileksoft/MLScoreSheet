using System;
using System.IO;
using System.Threading.Tasks;
using Foundation;
using Photos;
using UIKit;

namespace MLScoreSheetCounter.Platforms.iOS
{
    /// <summary>
    /// Helpers for saving photos to the Photos library with async support.
    /// </summary>
    public static class PhotoSaver
    {
        // --- 1) Async wrapper for PHPhotoLibrary.PerformChanges ---
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

        // --- 2) Ensure permissions: iOS 14+ uses AccessLevel ---
        public static async Task EnsureAddOnlyAuthorizationAsync()
        {
            var status = PHPhotoLibrary.GetAuthorizationStatus(PHAccessLevel.AddOnly);
            if (status == PHAuthorizationStatus.Authorized)
                return;

            status = await RequestAddOnlyAuthorizationAsync();
            if (status != PHAuthorizationStatus.Authorized)
                throw new UnauthorizedAccessException("Saving to the Photos library was not authorized.");
        }

        private static Task<PHAuthorizationStatus> RequestAddOnlyAuthorizationAsync()
        {
            var tcs = new TaskCompletionSource<PHAuthorizationStatus>();
            PHPhotoLibrary.RequestAuthorization(PHAccessLevel.AddOnly, s => tcs.TrySetResult(s));
            return tcs.Task;
        }

        // --- 3) Save a photo from a file (NSUrl) ---
        /// <summary>
        /// Saves a photo (file) to the Photos library.
        /// </summary>
        /// <param name="fileUrl">NSUrl to a local file (for example in /tmp or /Documents).</param>
        /// <param name="shouldMoveFile">
        /// true = move the file after adding (the app must have permission to modify the location),
        /// false = keep the file in place and copy the data.
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

        // --- 4) Save a photo from memory (NSData/byte[]/UIImage) ---
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

            // Save as JPEG (switch to PNG with image.AsPNG() if needed)
            using var data = image.AsJPEG(jpegQuality);
            await SavePhotoDataAsync(data);
        }

        // --- 5) Helper: save a stream to /tmp and then to Photos ---
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

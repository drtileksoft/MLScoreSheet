using System;
using System.Threading.Tasks;
using AVFoundation;
using Foundation;
using Photos;
using UIKit;

namespace MLScoreSheetCounter.Platforms.iOS
{
    public static class IosPermissions
    {
        // CAMERA -----------------------------------------------------------------
        public static async Task EnsureCameraAsync()
        {
            var status = AVCaptureDevice.GetAuthorizationStatus(AVAuthorizationMediaType.Video);
            if (status == AVAuthorizationStatus.NotDetermined)
            {
                var granted = await RequestCameraAsync();
                if (!granted) ThrowCameraDenied();
            }
            else if (status != AVAuthorizationStatus.Authorized)
            {
                ThrowCameraDenied();
            }
        }

        private static Task<bool> RequestCameraAsync()
        {
            var tcs = new TaskCompletionSource<bool>();
            AVCaptureDevice.RequestAccessForMediaType(AVAuthorizationMediaType.Video, granted => tcs.TrySetResult(granted));
            return tcs.Task;
        }

        private static void ThrowCameraDenied()
        {
            throw new UnauthorizedAccessException(
                "Camera access was denied. You can allow it in Settings → Privacy → Camera.");
        }

        // PHOTOS (READ/WRITE) -----------------------------------------------------
        public static async Task EnsurePhotoReadWriteAsync()
        {
            var status = PHPhotoLibrary.GetAuthorizationStatus(PHAccessLevel.ReadWrite);
            if (status == PHAuthorizationStatus.NotDetermined)
            {
                status = await RequestPhotosAsync(PHAccessLevel.ReadWrite);
            }

            // iOS can return "Limited" – that is sufficient for reading
            if (status != PHAuthorizationStatus.Authorized && status != PHAuthorizationStatus.Limited)
            {
                ThrowPhotosDenied("reading from the photo library");
            }
        }

        public static async Task EnsurePhotoAddOnlyAsync()
        {
            var status = PHPhotoLibrary.GetAuthorizationStatus(PHAccessLevel.AddOnly);
            if (status == PHAuthorizationStatus.NotDetermined)
            {
                status = await RequestPhotosAsync(PHAccessLevel.AddOnly);
            }

            if (status != PHAuthorizationStatus.Authorized)
            {
                ThrowPhotosDenied("saving to the photo library");
            }
        }

        private static Task<PHAuthorizationStatus> RequestPhotosAsync(PHAccessLevel level)
        {
            var tcs = new TaskCompletionSource<PHAuthorizationStatus>();
            PHPhotoLibrary.RequestAuthorization(level, s => tcs.TrySetResult(s));
            return tcs.Task;
        }

        private static void ThrowPhotosDenied(string what)
        {
            throw new UnauthorizedAccessException(
                $"Permission for {what} was denied. You can allow it in Settings → Privacy → Photos.");
        }

        // Optional: open the app settings
        public static void OpenAppSettings()
        {
            var url = new NSUrl(UIApplication.OpenSettingsUrlString);
            if (UIApplication.SharedApplication.CanOpenUrl(url))
                UIApplication.SharedApplication.OpenUrl(url);
        }
    }
}

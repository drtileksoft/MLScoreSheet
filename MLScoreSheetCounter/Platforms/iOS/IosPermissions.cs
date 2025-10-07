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
                "Přístup ke kameře byl odepřen. Povolit lze v Nastavení → Soukromí → Fotoaparát.");
        }

        // PHOTOS (READ/WRITE) -----------------------------------------------------
        public static async Task EnsurePhotoReadWriteAsync()
        {
            var status = PHPhotoLibrary.GetAuthorizationStatus(PHAccessLevel.ReadWrite);
            if (status == PHAuthorizationStatus.NotDetermined)
            {
                status = await RequestPhotosAsync(PHAccessLevel.ReadWrite);
            }

            // iOS může vrátit "Limited" – to pro čtení stačí
            if (status != PHAuthorizationStatus.Authorized && status != PHAuthorizationStatus.Limited)
            {
                ThrowPhotosDenied("čtení z knihovny fotek");
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
                ThrowPhotosDenied("ukládání do knihovny fotek");
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
                $"Oprávnění pro {what} bylo odepřeno. Povolit lze v Nastavení → Soukromí → Fotky.");
        }

        // Volitelné: otevření nastavení appky
        public static void OpenAppSettings()
        {
            var url = new NSUrl(UIApplication.OpenSettingsUrlString);
            if (UIApplication.SharedApplication.CanOpenUrl(url))
                UIApplication.SharedApplication.OpenUrl(url);
        }
    }
}

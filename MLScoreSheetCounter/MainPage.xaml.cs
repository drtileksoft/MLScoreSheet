using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using SkiaSharp;
using System.Threading;
using System.Threading.Tasks;
using MLScoreSheetCounter.Services;
using YourApp.Services;

namespace MLScoreSheetCounter;

public partial class MainPage : ContentPage
{
    private readonly IGallerySaver _gallerySaver;
    private string? _lastImagePath;
    private const float CrossedThreshold = 0.40f;

    public MainPage()
    {
        InitializeComponent();
        _gallerySaver = ServiceHelper.GetRequiredService<IGallerySaver>();
    }

    private void SetProcessingState(bool isProcessing)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            ProcessingContainer.IsVisible = isProcessing;
            ProcessingIndicator.IsRunning = isProcessing;
            ProcessingLabel.IsVisible = isProcessing;
            PickPhotoButton.IsEnabled = !isProcessing;
            TakePhotoButton.IsEnabled = !isProcessing;
            OverlayCheckBox.IsEnabled = !isProcessing;
            SaveToGalleryButton.IsEnabled = !isProcessing && !string.IsNullOrEmpty(_lastImagePath);
        });
    }

    private void ResetPreview()
    {
        PreviewContainer.Reset();
        Preview.Source = null;
        _lastImagePath = null;
        UpdateSaveButtonState();
    }

    private void UpdateSaveButtonState()
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            SaveToGalleryButton.IsEnabled = !string.IsNullOrEmpty(_lastImagePath) && !(ProcessingIndicator?.IsRunning ?? false);
        });
    }

    private async void OnPickPhoto(object sender, EventArgs e)
    {
        try
        {
            
            var file = await MediaPicker.PickPhotoAsync(new MediaPickerOptions
            {
                Title = "Vyber fotku score sheetu"
            });
            if (file == null)
            {
                return;
            }

            var copyPath = Path.Combine(FileSystem.Current.CacheDirectory, Path.GetFileName(file.FullPath));
            using (var src = File.OpenRead(file.FullPath))
            using (var dst = File.Create(copyPath))
            {
                await src.CopyToAsync(dst);
            }

            await RunDetection(copyPath);
        }
        catch (Exception ex)
        {
            await DisplayAlert("Chyba", ex.Message, "OK");
        }
    }

    private async void OnTakePhoto(object sender, EventArgs e)
    {
#if IOS
            // před focením získej povolení ke kameře
            await MLScoreSheetCounter.Platforms.iOS.IosPermissions.EnsureCameraAsync();
#endif
#if ANDROID || IOS
        try
        {
            

            var result = await MediaPicker.CapturePhotoAsync(new MediaPickerOptions
            {
                Title = "Vyfoť score sheet"
            });
            if (result == null)
            {
                return;
            }

            var copyPath = Path.Combine(FileSystem.Current.CacheDirectory, $"{DateTime.Now:yyyyMMdd_HHmmss}.jpg");
            using (var src = await result.OpenReadAsync())
            using (var dst = File.Create(copyPath))
            {
                await src.CopyToAsync(dst);
            }

            await RunDetection(copyPath);
        }
        catch (FeatureNotSupportedException)
        {
            await DisplayAlert("Chyba", "Zařízení nemá podporu pro kameru.", "OK");
        }
        catch (PermissionException)
        {
            await DisplayAlert("Chyba", "Chybí oprávnění ke kameře/úložišti.", "OK");
        }
        catch (Exception ex)
        {
            await DisplayAlert("Chyba", ex.Message, "OK");
        }
#else
        await DisplayAlert("Info", "Fotit lze jen na Android/iOS.", "OK");
#endif
    }

    private async Task RunDetection(string photoPath)
    {
        SetProcessingState(true);
        try
        {
            ResetPreview();
            var showOverlay = OverlayCheckBox?.IsChecked ?? false;

            if (showOverlay)
            {
                using var res = await Task.Run(async () =>
                {
                    using var photoWithOverlay = File.OpenRead(photoPath);
                    return await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
                        photoWithOverlay,
                        fixedThreshold: CrossedThreshold,
                        autoThreshold: false);
                });

                ResultLabel.Text = $"TOTAL = {res.Total}";

                var overlayPath = Path.Combine(FileSystem.Current.CacheDirectory, $"overlay_warped_{Guid.NewGuid():N}.png");
                using (var img = SKImage.FromBitmap(res.Overlay))
                using (var data = img.Encode(SKEncodedImageFormat.Png, 95))
                using (var fs = File.Create(overlayPath))
                {
                    data.SaveTo(fs);
                }

                _lastImagePath = overlayPath;
                Preview.Source = ImageSource.FromFile(overlayPath);
            }
            else
            {
                var total = await Task.Run(async () =>
                {
                    using var photo = File.OpenRead(photoPath);
                    return await SheetScoreEngine.ComputeTotalScoreAsync(
                        photo,
                        fixedThreshold: CrossedThreshold,
                        autoThreshold: false);
                });

                ResultLabel.Text = $"TOTAL = {total}";
                _lastImagePath = photoPath;
                Preview.Source = ImageSource.FromFile(photoPath);
            }
            UpdateSaveButtonState();
        }
        catch (Exception ex)
        {
            await DisplayAlert("Chyba", ex.Message, "OK");
        }
        finally
        {
            SetProcessingState(false);
        }
    }

    private async void OnSaveToGalleryClicked(object sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(_lastImagePath))
        {
            return;
        }

        try
        {
            if (!await EnsureSavePermissionAsync())
            {
                await DisplayAlert("Chyba", "Bez oprávnění nelze uložit do galerie.", "OK");
                return;
            }

            var fileName = Path.GetFileName(_lastImagePath);
            await _gallerySaver.SaveImageAsync(_lastImagePath, fileName, CancellationToken.None);
            await DisplayAlert("Hotovo", "Obrázek byl uložen do galerie.", "OK");
        }
        catch (Exception ex)
        {
            await DisplayAlert("Chyba", ex.Message, "OK");
        }
    }

    private static async Task<bool> EnsureSavePermissionAsync()
    {
#if ANDROID
        if (DeviceInfo.Current.Version.Major >= 13)
        {
            var status = await Permissions.RequestAsync<Permissions.Photos>();
            return status is PermissionStatus.Granted or PermissionStatus.Limited;
        }

        var writeStatus = await Permissions.RequestAsync<Permissions.StorageWrite>();
        return writeStatus == PermissionStatus.Granted;
#elif IOS
        var status = await Permissions.RequestAsync<Permissions.Photos>();
        return status is PermissionStatus.Granted or PermissionStatus.Limited;
#else
        return true;
#endif
    }
}

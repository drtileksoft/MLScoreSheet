using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using SkiaSharp;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using MLScoreSheet.Core;
using MLScoreSheetCounter.Services;

namespace MLScoreSheetCounter;

public partial class MainPage : ContentPage
{
    private readonly IGallerySaver _gallerySaver;
    private readonly IResourceProvider _resourceProvider = new MauiAssetResourceProvider();
    private string? _lastImagePath;
    public MainPage()
    {
        InitializeComponent();
        _gallerySaver = ServiceHelper.GetRequiredService<IGallerySaver>();
        UpdateThresholdLabels();
        ShowPreview(false);
    }

    private float GetCalculationThreshold() => (float)(CalculationTresholdSlider?.Value ?? 0.30);

    private float GetVisibilityThreshold() => (float)(VisibilityTresholdSlider?.Value ?? 0.30);

    private void OnCalculationThresholdChanged(object sender, ValueChangedEventArgs e) => UpdateThresholdLabels();

    private void OnVisibilityThresholdChanged(object sender, ValueChangedEventArgs e) => UpdateThresholdLabels();

    private void UpdateThresholdLabels()
    {
        void UpdateLabel(Label? label, double value)
        {
            if (label == null)
            {
                return;
            }

            label.Text = FormatThresholdValue(value);
        }

        var calcValue = CalculationTresholdSlider?.Value ?? 0.30;
        var visibilityValue = VisibilityTresholdSlider?.Value ?? 0.30;

        MainThread.BeginInvokeOnMainThread(() =>
        {
            UpdateLabel(CalculationThresholdValueLabel, calcValue);
            UpdateLabel(VisibilityThresholdValueLabel, visibilityValue);
        });
    }

    private static string FormatThresholdValue(double value) => value.ToString("0.00", CultureInfo.InvariantCulture);

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
        Preview.Source = null;
        _lastImagePath = null;
        UpdateSaveButtonState();
        ShowPreview(false);
    }

    private void UpdateSaveButtonState()
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            SaveToGalleryButton.IsEnabled = !string.IsNullOrEmpty(_lastImagePath) && !(ProcessingIndicator?.IsRunning ?? false);
        });
    }

    private void ShowPreview(bool isVisible)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            if (PreviewContainer != null)
            {
                PreviewContainer.IsVisible = isVisible;
            }

            if (PreviewPlaceholder != null)
            {
                PreviewPlaceholder.IsVisible = !isVisible;
            }
        });
    }

    private async void OnPickPhoto(object sender, EventArgs e)
    {
        try
        {
            
            var file = await MediaPicker.PickPhotoAsync(new MediaPickerOptions
            {
                Title = "Pick a photo of the scoresheet"
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
            await DisplayAlert("Error", ex.Message, "OK");
        }
    }

    private async void OnTakePhoto(object sender, EventArgs e)
    {
#if IOS
            // Obtain camera permission before capturing photos
            await MLScoreSheetCounter.Platforms.iOS.IosPermissions.EnsureCameraAsync();
#endif
#if ANDROID || IOS
        try
        {
            

            var result = await MediaPicker.CapturePhotoAsync(new MediaPickerOptions
            {
                Title = "Capture a photo of the score sheet"
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
            await DisplayAlert("Error", "This device does not support the camera.", "OK");
        }
        catch (PermissionException)
        {
            await DisplayAlert("Error", "Camera or storage permissions are missing.", "OK");
        }
        catch (Exception ex)
        {
            await DisplayAlert("Error", ex.Message, "OK");
        }
#else
        await DisplayAlert("Info", "Capturing photos is only supported on Android and iOS.", "OK");
#endif
    }

    private async Task RunDetection(string photoPath)
    {
        SetProcessingState(true);
        try
        {
            ResetPreview();
            var showOverlay = OverlayCheckBox?.IsChecked ?? false;
            var calculationThreshold = GetCalculationThreshold();
            var visibilityThreshold = GetVisibilityThreshold();

            if (showOverlay)
            {
                using var res = await Task.Run(async () =>
                {
                    using var photoWithOverlay = File.OpenRead(photoPath);
                    return await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
                        photoWithOverlay,
                        _resourceProvider,
                        fixedThreshold: calculationThreshold,
                        autoThreshold: false,
                        overlayVisibilityThreshold: visibilityThreshold);
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
                ShowPreview(true);
            }
            else
            {
                var total = await Task.Run(async () =>
                {
                    using var photo = File.OpenRead(photoPath);
                    return await SheetScoreEngine.ComputeTotalScoreAsync(
                        photo,
                        _resourceProvider,
                        fixedThreshold: calculationThreshold,
                        autoThreshold: false);
                });

                ResultLabel.Text = $"TOTAL = {total}";
                _lastImagePath = photoPath;
                Preview.Source = ImageSource.FromFile(photoPath);
                ShowPreview(true);
            }
            UpdateSaveButtonState();
        }
        catch (Exception ex)
        {
            await DisplayAlert("Error", ex.Message, "OK");
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
                await DisplayAlert("Error", "Cannot save to the gallery without permission.", "OK");
                return;
            }

            var fileName = Path.GetFileName(_lastImagePath);
            await _gallerySaver.SaveImageAsync(_lastImagePath, fileName, CancellationToken.None);
            await DisplayAlert("Success", "The image was saved to the gallery.", "OK");
        }
        catch (Exception ex)
        {
            await DisplayAlert("Error", ex.Message, "OK");
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

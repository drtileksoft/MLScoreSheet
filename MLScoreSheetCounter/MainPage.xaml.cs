using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using SkiaSharp;
using System.Threading.Tasks;
using YourApp.Services;

namespace MLScoreSheetCounter;

public partial class MainPage : ContentPage
{
    public MainPage()
    {
        InitializeComponent();
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
        });
    }

    private async void OnPickPhoto(object sender, EventArgs e)
    {
        try
        {
            var file = await FilePicker.PickAsync(new PickOptions
            {
                PickerTitle = "Vyber fotku score sheetu",
                FileTypes = FilePickerFileType.Images
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
            var showOverlay = OverlayCheckBox?.IsChecked ?? false;

            if (showOverlay)
            {
                using var res = await Task.Run(async () =>
                {
                    using var photoWithOverlay = File.OpenRead(photoPath);
                    return await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
                        photoWithOverlay,
                        fixedThreshold: 0.60f,
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

                Preview.Source = ImageSource.FromFile(overlayPath);
            }
            else
            {
                var total = await Task.Run(async () =>
                {
                    using var photo = File.OpenRead(photoPath);
                    return await SheetScoreEngine.ComputeTotalScoreAsync(
                        photo,
                        fixedThreshold: 0.60f,
                        autoThreshold: false);
                });

                ResultLabel.Text = $"TOTAL = {total}";
                Preview.Source = ImageSource.FromFile(photoPath);
            }
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
}

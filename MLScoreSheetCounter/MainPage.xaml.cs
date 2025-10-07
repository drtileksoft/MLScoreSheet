using Microsoft.Maui.Controls;
using SkiaSharp;
using YourApp.Services;

namespace MLScoreSheetCounter;

public partial class MainPage : ContentPage
{
    private readonly PictureCleaner _pictureCleaner = new();

    public MainPage()
    {
        InitializeComponent();
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
        try
        {
            string cleanedPhotoPath;
            try
            {
                cleanedPhotoPath = _pictureCleaner.Clean(photoPath);
            }
            catch (Exception cleanEx)
            {
                cleanedPhotoPath = photoPath;
                await DisplayAlert("Info", $"Nepodařilo se vyčistit fotku: {cleanEx.Message}. Použije se původní verze.", "OK");
            }
            var showOverlay = OverlayCheckBox?.IsChecked ?? false;

            if (showOverlay)
            {
                using var photoWithOverlay = File.OpenRead(cleanedPhotoPath);
                using var res = await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(
                    photoWithOverlay,
                    fixedThreshold: 0.72f,
                    autoThreshold: false);

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
                using var photo = File.OpenRead(cleanedPhotoPath);
                var total = await SheetScoreEngine.ComputeTotalScoreAsync(
                    photo,
                    fixedThreshold: 0.74f,
                    autoThreshold: false);

                ResultLabel.Text = $"TOTAL = {total}";
                Preview.Source = ImageSource.FromFile(cleanedPhotoPath);
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("Chyba", ex.Message, "OK");
        }
    }
}

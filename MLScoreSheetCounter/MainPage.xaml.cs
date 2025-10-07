using Microsoft.Maui.Controls;
using MLScoreSheetCounter;
using SkiaSharp;
using YourApp.Services;

namespace MLScoreSheetCounter;

public partial class MainPage : ContentPage
{
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
            if (file == null) return;

            // zkopíruj do cache (kvůli jednoduššímu přístupu)
            var copyPath = Path.Combine(FileSystem.Current.CacheDirectory, Path.GetFileName(file.FullPath));
            using (var src = File.OpenRead(file.FullPath))
            using (var dst = File.Create(copyPath))
                await src.CopyToAsync(dst);

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
            if (result == null) return;

            var copyPath = Path.Combine(FileSystem.Current.CacheDirectory, $"{DateTime.Now:yyyyMMdd_HHmmss}.jpg");
            using (var src = await result.OpenReadAsync())
            using (var dst = File.Create(copyPath))
                await src.CopyToAsync(dst);

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
            using var photo = File.OpenRead(photoPath);
            // pevný práh
            // return await SheetScoreEngine.ComputeTotalScoreAsync(photo, fixedThreshold: 0.35f, autoThreshold: false);

            // auto-threshold (k-means)
            // var result = await SheetScoreEngine.ComputeTotalScoreAsync(photo, fixedThreshold: 0.74f, autoThreshold: true);


            //var res = await SheetScoreEngine.ComputeTotalScoreAsync(photo, fixedThreshold: 0.74f, autoThreshold: false);
            //ResultLabel.Text = $"TOTAL = {res}";


            // 2) Debug overlay:
            using var res = await SheetScoreEngine.ComputeTotalScoreWithOverlayAsync(photo, fixedThreshold: 0.74f, autoThreshold: false);


            var path = Path.Combine(FileSystem.CacheDirectory, "overlay_warpedXX.png");
            using (var img = SKImage.FromBitmap(res.Overlay))
            using (var data = img.Encode(SKEncodedImageFormat.Png, 95))
            using (var fs = File.Create(path)) data.SaveTo(fs);
            Preview.Source = ImageSource.FromFile(path);

            ResultLabel.Text = $"TOTAL = {res.Total}";
        }
        catch (Exception ex)
        {
            await DisplayAlert("Chyba", ex.Message, "OK");
        }
    }
}

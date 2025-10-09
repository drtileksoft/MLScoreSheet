using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Graphics;
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
        ClearScoreDetailsDisplay();
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

            UpdatePlaceholderVisibility();
        });
    }

    private void ShowScoreDetails(bool isVisible)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            if (ScoreDetailsScroll != null)
            {
                ScoreDetailsScroll.IsVisible = isVisible;
            }

            UpdatePlaceholderVisibility();
        });
    }

    private void UpdatePlaceholderVisibility()
    {
        if (PreviewPlaceholder == null)
        {
            return;
        }

        bool hasPreview = PreviewContainer?.IsVisible ?? false;
        bool hasDetails = ScoreDetailsScroll?.IsVisible ?? false;
        PreviewPlaceholder.IsVisible = !hasPreview && !hasDetails;
    }

    private void ClearScoreDetailsDisplay()
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            if (ScoreDetailsGrid != null)
            {
                ScoreDetailsGrid.Children.Clear();
                ScoreDetailsGrid.RowDefinitions.Clear();
                ScoreDetailsGrid.ColumnDefinitions.Clear();
            }

            if (ScoreDetailsScroll != null)
            {
                ScoreDetailsScroll.IsVisible = false;
            }

            UpdatePlaceholderVisibility();
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
        catch (Exception)
        {
            await DisplayAlert("Error", "Please retake the photo or adjust it (crop and straighten) before trying again.", "OK");
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
        catch (Exception)
        {
            await DisplayAlert("Error", "Please retake the photo or adjust it (crop and straighten) before trying again.", "OK");
        }
#else
        await DisplayAlert("Info", "Capturing photos is only supported on Android and iOS.", "OK");
#endif
    }

    private void PopulateScoreDetails(MLScoreSheet.Core.SheetScoreEngine.ScoreOverlayResult.OverlayDetails details)
    {
        const int tablesPerRowBlock = 2;
        const int totalTables = tablesPerRowBlock * 2;

        MainThread.BeginInvokeOnMainThread(() =>
        {
            if (ScoreDetailsGrid == null)
            {
                return;
            }

            ScoreDetailsGrid.Children.Clear();
            ScoreDetailsGrid.RowDefinitions.Clear();
            ScoreDetailsGrid.ColumnDefinitions.Clear();

            for (int r = 0; r < 2; r++)
            {
                ScoreDetailsGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            }

            for (int c = 0; c < tablesPerRowBlock; c++)
            {
                ScoreDetailsGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Star });
            }

            for (int tableIndex = 0; tableIndex < totalTables; tableIndex++)
            {
                var tableView = CreateScoreDetailsView(details, tableIndex);
                int row = tableIndex / tablesPerRowBlock;
                int column = tableIndex % tablesPerRowBlock;
                Grid.SetRow(tableView, row);
                Grid.SetColumn(tableView, column);
                ScoreDetailsGrid.Children.Add(tableView);
            }

            if (ScoreDetailsScroll != null)
            {
                ScoreDetailsScroll.IsVisible = true;
            }

            UpdatePlaceholderVisibility();
        });
    }

    private View CreateScoreDetailsView(MLScoreSheet.Core.SheetScoreEngine.ScoreOverlayResult.OverlayDetails details, int tableIndex)
    {
        // Odvod�me po�et ��dk� z 2D pole RowSums: [table, row]
        int rowsPerTable = details.RowSums.GetLength(1);

        // --- lok�ln� helpery ---
        static Border Cell(string text,
                           bool bold = false,
                           bool end = false,
                           Color? bg = null,
                           Color? textColor = null)
        {
            return new Border
            {
                Stroke = Colors.Transparent,
                Background = new SolidColorBrush(bg ?? Colors.Transparent),
                Content = new Label
                {
                    Text = text,
                    FontAttributes = bold ? FontAttributes.Bold : FontAttributes.None,
                    FontSize = 14,
                    TextColor = textColor ?? Colors.Black,
                    VerticalTextAlignment = TextAlignment.Center,
                    HorizontalTextAlignment = end ? TextAlignment.End : TextAlignment.Start,
                    Margin = new Thickness(8, 6) // padding bu�ky
                }
            };
        }

        static void Add(Grid g, View v, int row, int col, int colSpan = 1)
        {
            Grid.SetRow(v, row);
            Grid.SetColumn(v, col);
            if (colSpan > 1) Grid.SetColumnSpan(v, colSpan);
            g.Children.Add(v);
        }

        // --- kontejner ---
        var frame = new Frame
        {
            Padding = new Thickness(12),
            BackgroundColor = Colors.White,
            BorderColor = Colors.LightGray,
            CornerRadius = 12,
            HasShadow = false
        };

        // 2 sloupce: popisek | hodnota
        var grid = new Grid
        {
            RowSpacing = 0,
            ColumnSpacing = 0,
            ColumnDefinitions =
        {
            new ColumnDefinition(GridLength.Star),
            new ColumnDefinition(GridLength.Auto)
        }
        };

        // ��dky: caption + (rows) + footer
        grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto)); // caption
        for (int r = 0; r < rowsPerTable; r++)
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
        grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto)); // footer (Total)

        int row = 0;

        // --- Caption ---
        Add(grid, Cell($"Table {tableIndex + 1}", bold: true, bg: Colors.Transparent), row, 0, colSpan: 2);
        row++;

        // --- ��dky se sou�ty ---
        for (int r = 0; r < rowsPerTable; r++)
        {
            Add(grid, Cell($"Row {r + 1}", textColor: Colors.DarkGray), row, 0);

            var rowSum = details.RowSums[tableIndex, r].ToString(CultureInfo.InvariantCulture);
            Add(grid, Cell(rowSum, bold: true, end: true), row, 1);

            // slab� odd�lovac� linka pod ka�d�m ��dkem (krom� posledn�ho p�ed footrem)
            if (r < rowsPerTable - 1)
            {
                row++;
                grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
                Add(grid, new BoxView { HeightRequest = 1, BackgroundColor = Colors.LightGray }, row, 0, colSpan: 2);
            }

            row++;
        }

        // --- Footer s Total ---
        var footerBg = Color.FromArgb("#F0F2F5");
        Add(grid, Cell("Total", bold: true, bg: footerBg), row, 0);
        var total = details.TableTotals[tableIndex].ToString(CultureInfo.InvariantCulture);
        Add(grid, Cell(total, bold: true, end: true, bg: footerBg), row, 1);

        frame.Content = grid;
        return frame;
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

                var overlayBitmap = res.Overlay ?? throw new InvalidOperationException("Overlay image was not created.");
                var overlayPath = Path.Combine(FileSystem.Current.CacheDirectory, $"overlay_warped_{Guid.NewGuid():N}.png");
                using (var img = SKImage.FromBitmap(overlayBitmap))
                using (var data = img.Encode(SKEncodedImageFormat.Png, 95))
                using (var fs = File.Create(overlayPath))
                {
                    data.SaveTo(fs);
                }

                _lastImagePath = overlayPath;
                Preview.Source = ImageSource.FromFile(overlayPath);
                ShowPreview(true);
                ShowScoreDetails(false);
            }
            else
            {
                using var result = await Task.Run(async () =>
                {
                    using var photo = File.OpenRead(photoPath);
                    return await SheetScoreEngine.ComputeTotalScoreAsync(
                        photo,
                        _resourceProvider,
                        fixedThreshold: calculationThreshold,
                        autoThreshold: false);
                });

                ResultLabel.Text = $"TOTAL = {result.Total}";
                _lastImagePath = null;
                Preview.Source = null;
                ShowPreview(false);
                PopulateScoreDetails(result.Details);
            }
            UpdateSaveButtonState();
        }
        catch (Exception)
        {
            await DisplayAlert("Error", "Please retake the photo or adjust it (crop and straighten) before trying again.", "OK");
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
        catch (Exception)
        {
            await DisplayAlert("Error", "Please retake the photo or adjust it (crop and straighten) before trying again.", "OK");
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

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

    private void PopulateScoreDetails(ScoreOverlayResult.OverlayDetails details)
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

    private View CreateScoreDetailsView(ScoreOverlayResult.OverlayDetails details, int tableIndex)
    {
        const int rowsPerTable = 5;
        const int columnsPerTable = 3;

        var frame = new Frame
        {
            Padding = new Thickness(12),
            BackgroundColor = Colors.White,
            BorderColor = Colors.LightGray,
            CornerRadius = 12,
            HasShadow = false
        };

        var grid = new Grid
        {
            ColumnSpacing = 6,
            RowSpacing = 6
        };

        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        for (int c = 0; c < columnsPerTable; c++)
        {
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Star });
        }

        void Place(View view, int column, int row, int columnSpan = 1)
        {
            Grid.SetColumn(view, column);
            Grid.SetRow(view, row);
            if (columnSpan > 1)
            {
                Grid.SetColumnSpan(view, columnSpan);
            }

            grid.Children.Add(view);
        }

        int currentRow = 0;

        var header = new Label
        {
            Text = $"Table {tableIndex + 1}",
            FontAttributes = FontAttributes.Bold,
            FontSize = 18,
            TextColor = Colors.Black
        };
        Place(header, 0, currentRow, columnsPerTable + 1);
        currentRow++;

        for (int r = 0; r < rowsPerTable; r++)
        {
            var rowLabel = new Label
            {
                Text = $"Row {r + 1}",
                TextColor = Colors.DarkGray
            };
            Place(rowLabel, 0, currentRow);

            var valueLabel = new Label
            {
                Text = details.RowSums[tableIndex, r].ToString(CultureInfo.InvariantCulture),
                FontAttributes = FontAttributes.Bold,
                HorizontalTextAlignment = TextAlignment.End,
                TextColor = Colors.Black
            };
            Place(valueLabel, 1, currentRow, columnsPerTable);
            currentRow++;
        }

        var separator = new BoxView
        {
            HeightRequest = 1,
            BackgroundColor = Colors.LightGray,
            HorizontalOptions = LayoutOptions.Fill
        };
        Place(separator, 0, currentRow, columnsPerTable + 1);
        currentRow++;

        var columnHeader = new Label
        {
            Text = "Column sums",
            FontAttributes = FontAttributes.Bold,
            TextColor = Colors.Black
        };
        Place(columnHeader, 0, currentRow);

        for (int c = 0; c < columnsPerTable; c++)
        {
            var columnValue = new Label
            {
                Text = details.ColumnSums[tableIndex, c].ToString(CultureInfo.InvariantCulture),
                FontAttributes = FontAttributes.Bold,
                HorizontalTextAlignment = TextAlignment.Center,
                TextColor = Colors.Black
            };
            Place(columnValue, c + 1, currentRow);
        }

        currentRow++;

        var totalLabel = new Label
        {
            Text = "Total",
            FontAttributes = FontAttributes.Bold,
            TextColor = Colors.Black
        };
        Place(totalLabel, 0, currentRow);

        var totalValue = new Label
        {
            Text = details.TableTotals[tableIndex].ToString(CultureInfo.InvariantCulture),
            FontAttributes = FontAttributes.Bold,
            HorizontalTextAlignment = TextAlignment.End,
            TextColor = Colors.Black
        };
        Place(totalValue, 1, currentRow, columnsPerTable);

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

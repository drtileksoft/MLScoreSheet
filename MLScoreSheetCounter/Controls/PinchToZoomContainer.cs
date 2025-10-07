using System.Threading.Tasks;
using Microsoft.Maui.Controls;

namespace MLScoreSheetCounter.Controls;

public class PinchToZoomContainer : ScrollView
{
    private const double MaxZoomScale = 20;

    public PinchToZoomContainer()
    {
        Orientation = ScrollOrientation.Both;
        ZoomMode = ScrollViewZoomMode.Enabled;
        MaximumZoomScale = MaxZoomScale;
        MinimumZoomScale = 1;
        HorizontalScrollBarVisibility = ScrollBarVisibility.Never;
        VerticalScrollBarVisibility = ScrollBarVisibility.Never;

        var doubleTap = new TapGestureRecognizer { NumberOfTapsRequired = 2 };
        doubleTap.Tapped += OnDoubleTapped;
        GestureRecognizers.Add(doubleTap);
    }

    private async void OnDoubleTapped(object? sender, TappedEventArgs e)
    {
        await ResetAsync();
    }

    public async Task ResetAsync()
    {
        if (CurrentZoomScale != 1)
        {
            SetValue(CurrentZoomScaleProperty, 1d);
        }

        await ScrollToAsync(0, 0, false);
    }

    public void Reset()
    {
        _ = ResetAsync();
    }
}

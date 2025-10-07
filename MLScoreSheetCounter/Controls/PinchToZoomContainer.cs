using System;
using Microsoft.Maui.Controls;

namespace MLScoreSheetCounter.Controls;

public class PinchToZoomContainer : ContentView
{
    private double _currentScale = 1;
    private double _startScale = 1;
    private double _xOffset;
    private double _yOffset;

    public PinchToZoomContainer()
    {
        var pinch = new PinchGestureRecognizer();
        pinch.PinchUpdated += OnPinchUpdated;
        GestureRecognizers.Add(pinch);

        var pan = new PanGestureRecognizer();
        pan.PanUpdated += OnPanUpdated;
        GestureRecognizers.Add(pan);

        var doubleTap = new TapGestureRecognizer { NumberOfTapsRequired = 2 };
        doubleTap.Tapped += (_, _) => Reset();
        GestureRecognizers.Add(doubleTap);
    }

    public void Reset()
    {
        _currentScale = 1;
        _startScale = 1;
        _xOffset = 0;
        _yOffset = 0;

        if (Content != null)
        {
            Content.AnchorX = 0.5;
            Content.AnchorY = 0.5;
            Content.Scale = 1;
            Content.TranslationX = 0;
            Content.TranslationY = 0;
        }
    }

    private void OnPinchUpdated(object? sender, PinchGestureUpdatedEventArgs e)
    {
        if (Content == null)
        {
            return;
        }

        switch (e.Status)
        {
            case GestureStatus.Started:
                _startScale = Content.Scale;
                Content.AnchorX = 0;
                Content.AnchorY = 0;
                break;
            case GestureStatus.Running:
                var targetScale = Math.Clamp(_startScale * e.Scale, 1, 6);

                var renderedX = Content.X + _xOffset;
                var deltaX = renderedX / Width;
                var deltaWidth = Width / (Content.Width * _startScale);
                var originX = (e.ScaleOrigin.X - deltaX) * deltaWidth;

                var renderedY = Content.Y + _yOffset;
                var deltaY = renderedY / Height;
                var deltaHeight = Height / (Content.Height * _startScale);
                var originY = (e.ScaleOrigin.Y - deltaY) * deltaHeight;

                var targetX = _xOffset - (originX * Content.Width) * (targetScale - _startScale);
                var targetY = _yOffset - (originY * Content.Height) * (targetScale - _startScale);

                Content.TranslationX = Clamp(targetX, -GetMaxTranslationX(targetScale), GetMaxTranslationX(targetScale));
                Content.TranslationY = Clamp(targetY, -GetMaxTranslationY(targetScale), GetMaxTranslationY(targetScale));
                Content.Scale = targetScale;

                _currentScale = targetScale;
                break;
            case GestureStatus.Completed:
                _xOffset = Content.TranslationX;
                _yOffset = Content.TranslationY;
                break;
        }
    }

    private void OnPanUpdated(object? sender, PanUpdatedEventArgs e)
    {
        if (Content == null)
        {
            return;
        }

        switch (e.StatusType)
        {
            case GestureStatus.Running:
                if (_currentScale <= 1)
                {
                    return;
                }

                var newX = _xOffset + e.TotalX;
                var newY = _yOffset + e.TotalY;

                Content.TranslationX = Clamp(newX, -GetMaxTranslationX(_currentScale), GetMaxTranslationX(_currentScale));
                Content.TranslationY = Clamp(newY, -GetMaxTranslationY(_currentScale), GetMaxTranslationY(_currentScale));
                break;
            case GestureStatus.Completed:
                _xOffset = Content.TranslationX;
                _yOffset = Content.TranslationY;
                break;
        }
    }

    private double GetMaxTranslationX(double scale)
    {
        if (Content == null || Width <= 0)
        {
            return 0;
        }

        var scaledWidth = Content.Width * scale;
        var maxTranslate = (scaledWidth - Width) / 2;
        return Math.Max(0, maxTranslate);
    }

    private double GetMaxTranslationY(double scale)
    {
        if (Content == null || Height <= 0)
        {
            return 0;
        }

        var scaledHeight = Content.Height * scale;
        var maxTranslate = (scaledHeight - Height) / 2;
        return Math.Max(0, maxTranslate);
    }

    private static double Clamp(double value, double min, double max)
    {
        if (value < min)
        {
            return min;
        }

        if (value > max)
        {
            return max;
        }

        return value;
    }
}

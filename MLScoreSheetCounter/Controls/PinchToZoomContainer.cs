using System;
using Microsoft.Maui.Controls;

namespace MLScoreSheetCounter.Controls;

public class PinchToZoomContainer : ContentView
{
    private const double MaxZoomScale = 20;
    private double _currentScale = 1;
    private double _startScale = 1;
    private double _xOffset;
    private double _yOffset;
    private double _startX;
    private double _startY;

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
        _startX = 0;
        _startY = 0;

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
                _startScale = _currentScale = Content.Scale;
                _startX = _xOffset = Content.TranslationX;
                _startY = _yOffset = Content.TranslationY;
                Content.AnchorX = 0.5;
                Content.AnchorY = 0.5;
                break;
            case GestureStatus.Running:
                if (Width <= 0 || Height <= 0 || Content.Width <= 0 || Content.Height <= 0)
                {
                    return;
                }

                var targetScale = Math.Clamp(_startScale * e.Scale, 1, MaxZoomScale);

                var focusX = (e.ScaleOrigin.X - 0.5) * Width;
                var focusY = (e.ScaleOrigin.Y - 0.5) * Height;
                var scaleDelta = targetScale / _startScale;

                var targetX = _startX - focusX * (scaleDelta - 1);
                var targetY = _startY - focusY * (scaleDelta - 1);

                var maxX = GetMaxTranslationX(targetScale);
                var maxY = GetMaxTranslationY(targetScale);

                Content.TranslationX = Clamp(targetX, -maxX, maxX);
                Content.TranslationY = Clamp(targetY, -maxY, maxY);
                Content.Scale = targetScale;

                _xOffset = Content.TranslationX;
                _yOffset = Content.TranslationY;
                _currentScale = targetScale;
                break;
            case GestureStatus.Completed:
                _xOffset = Content.TranslationX;
                _yOffset = Content.TranslationY;
                _currentScale = Content.Scale;
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

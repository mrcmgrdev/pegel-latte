"""Tests for scale reading and interpolation."""

from pegel_latte.models import ScaleReading
from pegel_latte.detection.scale import interpolate_water_level


def test_interpolate_between_two_readings():
    # Gauge: 200 at y=100 (top), 100 at y=300 (bottom)
    # Numbers decrease going down (higher y = lower water)
    readings = [
        ScaleReading(value_cm=200, y_position=100),
        ScaleReading(value_cm=100, y_position=300),
    ]
    # Waterline at y=200 (midpoint) → should be 150 cm
    result = interpolate_water_level(readings, waterline_y=200, roi_height=400)
    assert result is not None
    assert abs(result - 150.0) < 0.1


def test_interpolate_at_boundary():
    readings = [
        ScaleReading(value_cm=300, y_position=50),
        ScaleReading(value_cm=200, y_position=150),
    ]
    # Waterline at exact reading position
    result = interpolate_water_level(readings, waterline_y=50, roi_height=200)
    assert result is not None
    assert abs(result - 300.0) < 0.1


def test_interpolate_extrapolate_above():
    readings = [
        ScaleReading(value_cm=200, y_position=100),
        ScaleReading(value_cm=100, y_position=200),
    ]
    # Waterline above both readings (y=50)
    result = interpolate_water_level(readings, waterline_y=50, roi_height=300)
    assert result is not None
    assert result > 200  # should extrapolate higher


def test_interpolate_extrapolate_below():
    readings = [
        ScaleReading(value_cm=200, y_position=100),
        ScaleReading(value_cm=100, y_position=200),
    ]
    # Waterline below both readings (y=250)
    result = interpolate_water_level(readings, waterline_y=250, roi_height=300)
    assert result is not None
    assert result < 100  # should extrapolate lower


def test_interpolate_insufficient_readings():
    readings = [ScaleReading(value_cm=100, y_position=100)]
    result = interpolate_water_level(readings, waterline_y=150, roi_height=200)
    assert result is None


def test_interpolate_empty_readings():
    result = interpolate_water_level([], waterline_y=100, roi_height=200)
    assert result is None

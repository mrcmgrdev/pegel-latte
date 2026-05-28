"""Tests for data models."""

from pathlib import Path
from datetime import datetime

from pegel_latte.models import WaterLevelReading, ImageMetadata, ScaleReading


def test_water_level_reading_defaults():
    reading = WaterLevelReading()
    assert reading.water_level_cm is None
    assert reading.confidence == 0.0
    assert reading.error is None
    assert reading.scale_readings == []


def test_water_level_reading_to_dict():
    reading = WaterLevelReading(
        water_level_cm=142.5,
        confidence=0.85,
        source_image=Path("test.jpg"),
        metadata=ImageMetadata(
            timestamp=datetime(2026, 5, 28, 10, 30, 0),
            latitude=48.2,
            longitude=16.3,
        ),
        raw_ocr_text="140\n150",
    )
    d = reading.to_dict()
    assert d["water_level_cm"] == 142.5
    assert d["confidence"] == 0.85
    assert d["timestamp"] == "2026-05-28T10:30:00"
    assert d["latitude"] == 48.2
    assert d["longitude"] == 16.3


def test_water_level_reading_properties():
    ts = datetime(2026, 1, 1)
    reading = WaterLevelReading(
        metadata=ImageMetadata(timestamp=ts, latitude=48.0, longitude=16.0)
    )
    assert reading.timestamp == ts
    assert reading.latitude == 48.0
    assert reading.longitude == 16.0

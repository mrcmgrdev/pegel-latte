"""Integration tests with sample images."""

from pathlib import Path

import pytest

from pegel_latte.reader import read_water_level, read_directory, SUPPORTED_EXTENSIONS


DATA_DIR = Path(__file__).parent.parent / "data" / "real"


@pytest.fixture
def sample_images():
    """Get list of sample images from data/ directory."""
    if not DATA_DIR.exists():
        pytest.skip("data/ directory not found")
    images = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        pytest.skip("No sample images found in data/")
    return images


def test_reader_does_not_crash(sample_images):
    """Ensure reader handles all sample images without crashing."""
    for img_path in sample_images:
        reading = read_water_level(img_path)
        assert reading is not None
        assert reading.source_image == img_path
        # Should either have a result or an error, not both
        if reading.water_level_cm is not None:
            assert reading.error is None
            assert reading.confidence > 0


def test_read_directory(sample_images):
    """Test batch processing of directory."""
    readings = read_directory(DATA_DIR)
    assert len(readings) == len(sample_images)


def test_reading_has_gauge_detection(sample_images):
    """All images should at least detect a gauge region (even fallback)."""
    for img_path in sample_images:
        reading = read_water_level(img_path)
        assert reading.gauge_detection is not None
        assert reading.gauge_detection.roi_width > 0
        assert reading.gauge_detection.roi_height > 0

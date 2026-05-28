"""Main reader orchestrator — processes images and produces water level readings."""

from pathlib import Path

import cv2
import numpy as np

from pegel_latte.detection.gauge import detect_gauge
from pegel_latte.detection.waterline import detect_waterline
from pegel_latte.detection.scale import read_scale, interpolate_water_level
from pegel_latte.metadata import extract_metadata
from pegel_latte.models import WaterLevelReading


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def read_water_level(image_path: Path, verbose: bool = False) -> WaterLevelReading:
    """
    Process a single image and return the water level reading.

    Pipeline:
    1. Load image
    2. Extract EXIF metadata (timestamp, GPS)
    3. Detect gauge region
    4. Detect waterline
    5. OCR scale markings
    6. Interpolate water level

    Args:
        image_path: Path to the image file
        verbose: If True, save debug visualization images

    Returns:
        WaterLevelReading with results (may have error set if processing failed)
    """
    reading = WaterLevelReading(source_image=image_path)

    # Step 1: Load image
    image = cv2.imread(str(image_path))
    if image is None:
        reading.error = f"Could not load image: {image_path}"
        return reading

    # Step 2: Extract metadata
    reading.metadata = extract_metadata(image_path)

    # Step 3: Detect gauge region
    gauge = detect_gauge(image)
    if gauge is None:
        reading.error = "Could not detect gauge in image"
        return reading
    reading.gauge_detection = gauge

    # Step 4: Extract ROI
    roi = image[
        gauge.roi_y : gauge.roi_y + gauge.roi_height,
        gauge.roi_x : gauge.roi_x + gauge.roi_width,
    ]

    if roi.size == 0:
        reading.error = "Gauge ROI is empty"
        return reading

    # Step 5: Detect waterline in ROI
    waterline_y = detect_waterline(roi, gauge.roi_y, gauge.roi_height)
    reading.waterline_y = waterline_y

    # Step 6: OCR scale markings
    scale_readings, raw_text = read_scale(roi)
    reading.scale_readings = scale_readings
    reading.raw_ocr_text = raw_text

    # Step 7: Interpolate water level
    if waterline_y is not None and len(scale_readings) >= 2:
        water_level = interpolate_water_level(scale_readings, waterline_y, gauge.roi_height)
        if water_level is not None:
            reading.water_level_cm = round(water_level, 1)
            # Confidence based on detection quality
            reading.confidence = _compute_confidence(gauge, scale_readings, waterline_y)
    elif waterline_y is not None:
        reading.error = "Could not read enough scale markings for interpolation"
        reading.confidence = 0.2
    else:
        reading.error = "Could not detect waterline"
        reading.confidence = 0.1

    # Optional debug visualization
    if verbose:
        _save_debug_image(image, roi, gauge, waterline_y, scale_readings, image_path)

    return reading


def read_directory(directory: Path, verbose: bool = False) -> list[WaterLevelReading]:
    """Process all supported images in a directory."""
    readings = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            reading = read_water_level(path, verbose=verbose)
            readings.append(reading)
    return readings


def _compute_confidence(gauge, scale_readings, waterline_y) -> float:
    """Compute overall confidence score for a reading."""
    # Base confidence from gauge detection
    conf = gauge.confidence

    # Boost if multiple scale readings found
    if len(scale_readings) >= 3:
        conf += 0.2
    elif len(scale_readings) >= 2:
        conf += 0.1

    # Check if scale readings are consistent (monotonically increasing/decreasing with y)
    if len(scale_readings) >= 2:
        values = [r.value_cm for r in scale_readings]
        if values == sorted(values) or values == sorted(values, reverse=True):
            conf += 0.1

    return min(conf, 1.0)


def _save_debug_image(
    original: np.ndarray,
    roi: np.ndarray,
    gauge,
    waterline_y: int | None,
    scale_readings,
    image_path: Path,
):
    """Save a debug visualization showing detection results."""
    debug = original.copy()

    # Draw gauge ROI
    cv2.rectangle(
        debug,
        (gauge.roi_x, gauge.roi_y),
        (gauge.roi_x + gauge.roi_width, gauge.roi_y + gauge.roi_height),
        (0, 255, 0),
        2,
    )

    # Draw waterline
    if waterline_y is not None:
        y_in_image = gauge.roi_y + waterline_y
        cv2.line(
            debug,
            (gauge.roi_x, y_in_image),
            (gauge.roi_x + gauge.roi_width, y_in_image),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            debug,
            "WATERLINE",
            (gauge.roi_x + gauge.roi_width + 5, y_in_image),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    # Draw scale readings
    for sr in scale_readings:
        y_in_image = gauge.roi_y + sr.y_position
        cv2.circle(debug, (gauge.roi_x + gauge.roi_width // 2, y_in_image), 4, (0, 0, 255), -1)
        cv2.putText(
            debug,
            f"{sr.value_cm}cm",
            (gauge.roi_x + gauge.roi_width + 5, y_in_image),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    # Save debug image
    debug_path = image_path.parent / f"{image_path.stem}_debug{image_path.suffix}"
    cv2.imwrite(str(debug_path), debug)

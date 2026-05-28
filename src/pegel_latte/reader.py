"""Main reader orchestrator — processes images and produces water level readings."""

from pathlib import Path

import cv2
import numpy as np

from pegel_latte.detection.gauge import detect_gauge
from pegel_latte.detection.waterline import detect_waterline
from pegel_latte.detection.scale import read_scale_full_image, interpolate_water_level
from pegel_latte.metadata import extract_metadata
from pegel_latte.models import WaterLevelReading, GaugeDetection, ScaleReading


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def read_water_level(image_path: Path, verbose: bool = False) -> WaterLevelReading:
    """
    Process a single image and return the water level reading.

    Pipeline:
    1. Load image and extract EXIF metadata
    2. Run OCR on full image to find numbers
    3. Cluster numbers vertically to locate the gauge
    4. Detect waterline in the gauge region
    5. Interpolate water level from scale readings + waterline

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

    height, width = image.shape[:2]

    # Step 2: Extract metadata
    reading.metadata = extract_metadata(image_path)

    # Step 3: Run OCR on full image — this finds numbers AND tells us where the gauge is
    scale_readings, raw_text, gauge_x = read_scale_full_image(image)
    reading.scale_readings = scale_readings
    reading.raw_ocr_text = raw_text

    # Step 4: Determine gauge region
    if gauge_x is not None and len(scale_readings) >= 2:
        # Use OCR-detected positions to define gauge region
        gauge_width = max(int(width * 0.1), 100)
        roi_x = max(0, gauge_x - gauge_width // 2)
        roi_w = min(width - roi_x, gauge_width)
        roi_y = max(0, min(r.y_position for r in scale_readings) - 50)

        # Extend ROI below lowest reading to capture waterline
        # For multi-meter gauges (large value ranges), extend further
        max_reading_y = max(r.y_position for r in scale_readings)
        val_range = max(r.value_cm for r in scale_readings) - min(r.value_cm for r in scale_readings)
        # Multi-meter: extend by estimated 1 meter band below lowest reading
        if val_range > 50 and len(scale_readings) >= 3:
            px_per_cm = (max_reading_y - min(r.y_position for r in scale_readings)) / max(val_range, 1)
            extend_below = int(px_per_cm * 100)  # 1 full meter band
        else:
            extend_below = 300
        roi_h = min(height - roi_y, max_reading_y - roi_y + extend_below)

        gauge = GaugeDetection(
            roi_x=roi_x, roi_y=roi_y, roi_width=roi_w, roi_height=roi_h, confidence=0.7
        )
    else:
        # Fallback to CV-based gauge detection
        gauge = detect_gauge(image)

    if gauge is None:
        reading.error = "Could not detect gauge in image"
        return reading
    reading.gauge_detection = gauge

    # Step 5: Extract ROI and detect waterline
    roi = image[
        gauge.roi_y : gauge.roi_y + gauge.roi_height,
        gauge.roi_x : gauge.roi_x + gauge.roi_width,
    ]

    if roi.size == 0:
        reading.error = "Gauge ROI is empty"
        return reading

    waterline_y, waterline_conf = detect_waterline(roi, gauge.roi_y, gauge.roi_height)

    # For multi-meter gauges, waterline MUST be well below the lowest reading.
    # Re-run detection on a sub-ROI below the lowest reading to avoid
    # picking up meter marker edges or text boundaries.
    if len(scale_readings) >= 2:
        val_range = max(r.value_cm for r in scale_readings) - min(r.value_cm for r in scale_readings)
        lowest_reading_y = max(r.y_position for r in scale_readings) - gauge.roi_y

        if val_range > 50:
            # Multi-meter gauge: search only below lowest reading
            # Use dark-valley detection (dirty gauge pattern: bright→dark→bright)
            sub_roi_start = lowest_reading_y
            if sub_roi_start < gauge.roi_height - 50:
                sub_roi = roi[sub_roi_start:, :]
                wl_sub, conf_sub = detect_waterline(
                    sub_roi, 0, sub_roi.shape[0], prefer_dark_valley=True
                )
                if wl_sub is not None:
                    waterline_y = wl_sub + sub_roi_start
                    waterline_conf = conf_sub
        elif waterline_y is not None and waterline_y < lowest_reading_y:
            # Single-band: waterline above readings is suspicious
            waterline_y = lowest_reading_y + int(gauge.roi_height * 0.1)
            waterline_conf *= 0.3

    reading.waterline_y = waterline_y

    # Step 6: Interpolate water level
    if waterline_y is not None and len(scale_readings) >= 2:
        # Adjust scale reading y-positions to be relative to gauge ROI
        adjusted_readings = [
            ScaleReading(value_cm=r.value_cm, y_position=r.y_position - gauge.roi_y)
            for r in scale_readings
        ]
        water_level = interpolate_water_level(adjusted_readings, waterline_y, gauge.roi_height)
        if water_level is not None:
            reading.water_level_cm = round(water_level, 1)
            reading.confidence = _compute_confidence(gauge, scale_readings, waterline_y, waterline_conf)
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


def _compute_confidence(gauge, scale_readings, waterline_y, waterline_conf: float = 0.5) -> float:
    """Compute overall confidence score for a reading, incorporating waterline confidence."""
    conf = float(gauge.confidence) * 0.3  # base from gauge detection

    # Scale reading quality (up to 0.3)
    if len(scale_readings) >= 3:
        conf += 0.3
    elif len(scale_readings) >= 2:
        conf += 0.2

    # Monotonicity bonus (up to 0.1)
    if len(scale_readings) >= 2:
        values = [r.value_cm for r in scale_readings]
        if values == sorted(values) or values == sorted(values, reverse=True):
            conf += 0.1

    # Waterline detection confidence (up to 0.3)
    conf += float(waterline_conf) * 0.3

    return float(min(conf, 1.0))


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
            3,
        )
        cv2.putText(
            debug, "WATERLINE",
            (gauge.roi_x + gauge.roi_width + 10, y_in_image),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
        )

    # Draw scale readings
    for sr in scale_readings:
        cv2.circle(debug, (gauge.roi_x + gauge.roi_width // 2, sr.y_position), 6, (0, 0, 255), -1)
        cv2.putText(
            debug, f"{sr.value_cm}",
            (gauge.roi_x + gauge.roi_width + 10, sr.y_position),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    # Save debug image
    debug_path = image_path.parent / f"{image_path.stem}_debug{image_path.suffix}"
    cv2.imwrite(str(debug_path), debug)

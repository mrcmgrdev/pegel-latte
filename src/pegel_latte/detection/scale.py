"""Read scale markings from gauge using OCR."""

import re

import cv2
import numpy as np
import pytesseract

from pegel_latte.models import ScaleReading


def read_scale(image: np.ndarray) -> tuple[list[ScaleReading], str]:
    """
    Read numerical scale markings from a gauge ROI image using OCR.

    Austrian Pegellatten typically have numbers every 10cm or 50cm,
    with smaller tick marks in between.

    Args:
        image: The gauge ROI image (already cropped to gauge region)

    Returns:
        Tuple of (list of ScaleReadings sorted by y-position, raw OCR text)
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return [], ""

    # Pre-process for OCR
    processed = _preprocess_for_ocr(image)

    # Run OCR with digit-optimized config
    ocr_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
    raw_text = pytesseract.image_to_string(processed, config=ocr_config)

    # Also get bounding box data for spatial mapping
    ocr_data = pytesseract.image_to_data(processed, config=ocr_config, output_type=pytesseract.Output.DICT)

    readings = _parse_ocr_data(ocr_data, height)

    # If standard OCR didn't find enough, try with different preprocessing
    if len(readings) < 2:
        processed_alt = _preprocess_alternative(image)
        ocr_data_alt = pytesseract.image_to_data(
            processed_alt, config=ocr_config, output_type=pytesseract.Output.DICT
        )
        readings_alt = _parse_ocr_data(ocr_data_alt, height)
        if len(readings_alt) > len(readings):
            readings = readings_alt
            raw_text = pytesseract.image_to_string(processed_alt, config=ocr_config)

    # Sort by y-position (top to bottom)
    readings.sort(key=lambda r: r.y_position)

    return readings, raw_text.strip()


def interpolate_water_level(
    readings: list[ScaleReading], waterline_y: int, roi_height: int
) -> float | None:
    """
    Interpolate the water level based on scale readings and waterline position.

    On a Pegellatte, numbers increase from bottom to top (higher water = higher number).
    In image coordinates, y increases downward, so higher numbers have lower y values.

    Args:
        readings: Sorted list of ScaleReadings (by y_position, top to bottom)
        waterline_y: Y-coordinate of the detected waterline in the ROI
        roi_height: Height of the ROI image

    Returns:
        Interpolated water level in cm, or None if cannot be determined.
    """
    if len(readings) < 2:
        return None

    # Find the two readings bracketing the waterline
    above = None  # reading above waterline (lower y, higher value)
    below = None  # reading below waterline (higher y, lower value)

    for reading in readings:
        if reading.y_position <= waterline_y:
            above = reading
        elif reading.y_position > waterline_y and below is None:
            below = reading

    if above is None and below is None:
        return None

    # If waterline is above all readings, extrapolate from top two
    if above is None and len(readings) >= 2:
        r1, r2 = readings[0], readings[1]
        if r2.y_position == r1.y_position:
            return None
        cm_per_pixel = (r1.value_cm - r2.value_cm) / (r2.y_position - r1.y_position)
        return r1.value_cm + cm_per_pixel * (r1.y_position - waterline_y)

    # If waterline is below all readings, extrapolate from bottom two
    if below is None and len(readings) >= 2:
        r1, r2 = readings[-2], readings[-1]
        if r2.y_position == r1.y_position:
            return None
        cm_per_pixel = (r1.value_cm - r2.value_cm) / (r2.y_position - r1.y_position)
        return r2.value_cm + cm_per_pixel * (r2.y_position - waterline_y)

    # Interpolate between the two bracketing readings
    if above is not None and below is not None:
        if below.y_position == above.y_position:
            return None
        fraction = (waterline_y - above.y_position) / (below.y_position - above.y_position)
        return above.value_cm + fraction * (below.value_cm - above.value_cm)

    return None


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Pre-process image for optimal OCR of gauge numbers."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # Adaptive threshold for clear text
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Scale up for better OCR (tesseract works best at ~300 DPI)
    height, width = binary.shape[:2]
    if width < 300:
        scale = 300 / width
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return binary


def _preprocess_alternative(image: np.ndarray) -> np.ndarray:
    """Alternative preprocessing: inverse binary with morphological cleanup."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Scale up
    height, width = binary.shape[:2]
    if width < 300:
        scale = 300 / width
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Invert back (tesseract expects dark text on light background)
    return cv2.bitwise_not(binary)


def _parse_ocr_data(ocr_data: dict, img_height: int) -> list[ScaleReading]:
    """Parse pytesseract output data into ScaleReadings."""
    readings = []
    n_boxes = len(ocr_data["text"])

    for i in range(n_boxes):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])

        # Skip low confidence or empty results
        if conf < 30 or not text:
            continue

        # Extract numbers (gauge markings are typically integers)
        numbers = re.findall(r"\d+", text)
        if not numbers:
            continue

        for num_str in numbers:
            try:
                value = int(num_str)
                # Gauge values are typically 0-1000 cm
                if 0 <= value <= 1000:
                    y_pos = ocr_data["top"][i] + ocr_data["height"][i] // 2
                    readings.append(ScaleReading(value_cm=value, y_position=y_pos))
            except ValueError:
                continue

    # Deduplicate readings at similar y-positions
    readings = _deduplicate_readings(readings, img_height)

    return readings


def _deduplicate_readings(readings: list[ScaleReading], img_height: int) -> list[ScaleReading]:
    """Remove duplicate readings at similar y-positions."""
    if not readings:
        return readings

    threshold = img_height * 0.03  # within 3% of image height
    readings.sort(key=lambda r: r.y_position)

    deduped = [readings[0]]
    for r in readings[1:]:
        if abs(r.y_position - deduped[-1].y_position) > threshold:
            deduped.append(r)
        else:
            # Keep the one with a more "round" number (likely a real gauge marking)
            if _is_rounder(r.value_cm, deduped[-1].value_cm):
                deduped[-1] = r

    return deduped


def _is_rounder(a: int, b: int) -> bool:
    """Check if a is a 'rounder' number than b (divisible by larger powers of 10)."""
    if a == 0:
        return True
    if b == 0:
        return False
    # Count trailing zeros
    a_zeros = len(str(a)) - len(str(a).rstrip("0")) if a != 0 else 0
    b_zeros = len(str(b)) - len(str(b).rstrip("0")) if b != 0 else 0
    return a_zeros > b_zeros

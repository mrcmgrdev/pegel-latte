"""Read scale markings from gauge using OCR (EasyOCR for scene text).

Key domain knowledge for Pegellatten (gauge staffs):
- Numbers are typically single-digit DECIMETERS (1=10cm, 2=20cm, etc.) or
  multi-digit CM values (39=39cm, 595=595cm on flood gauges)
- E-shaped and reverse-E-shaped subdivision marks look like "3" to OCR
- Numbers DECREASE going DOWN toward water (higher y = lower value)
- The LOWEST visible number (highest y, nearest waterline) ≈ the water level
- Numbers below the waterline are submerged and unreadable
"""

import cv2
import numpy as np
import easyocr

from pegel_latte.models import ScaleReading


_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    """Get or initialize the EasyOCR reader (singleton)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def read_scale_full_image(image: np.ndarray) -> tuple[list[ScaleReading], str, int | None]:
    """
    Run OCR on the full image, find gauge numbers, filter E-marks, and locate the gauge.

    Handles both single-band gauges and multi-meter gauges with Roman numerals.

    Returns:
        Tuple of (filtered ScaleReadings in cm, raw OCR text, gauge x-center or None)
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return [], "", None

    # Upscale small images for better OCR (EasyOCR struggles below ~1000px)
    scale_factor = 1
    if max(height, width) < 1200:
        scale_factor = 2
        image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]

    reader = _get_reader()

    # Run OCR with digits + Roman numeral characters
    results = reader.readtext(
        image,
        allowlist="0123456789IVXivx",
        detail=1,
        paragraph=False,
    )

    # Collect all number detections and Roman numeral markers
    detections = []  # (text, value, x_center, y_center, confidence)
    roman_markers = []  # (meter_value, y_center, confidence)
    raw_texts = []

    for bbox, text, conf in results:
        text = text.strip()
        if not text or conf < 0.10:
            continue

        raw_texts.append(f"{text}({conf:.0%})")

        y_center = int((bbox[0][1] + bbox[2][1]) / 2) // scale_factor
        x_center = int((bbox[0][0] + bbox[2][0]) / 2) // scale_factor

        # Check for Roman numerals
        roman_val = _parse_roman(text)
        if roman_val is not None and conf >= 0.4:
            roman_markers.append((roman_val, y_center, conf))
            continue

        # Only digits
        digits_only = ''.join(c for c in text if c.isdigit())
        if not digits_only:
            continue

        value = int(digits_only)
        if value > 1000:
            continue

        detections.append((digits_only, value, x_center, y_center, conf))

    raw_text = " | ".join(raw_texts)

    if not detections:
        return [], raw_text, None

    # Use original dimensions for clustering/sequencing
    orig_height = height // scale_factor
    orig_width = width // scale_factor

    # Cluster by x to find gauge column
    gauge_x, gauge_dets = _cluster_by_x(detections, orig_width)

    if not gauge_dets:
        return [], raw_text, None

    # If Roman numerals found, try multi-meter approach
    if roman_markers:
        readings = _multi_meter_reading(gauge_dets, roman_markers, orig_height)
        if readings:
            readings.sort(key=lambda r: r.y_position)
            return readings, raw_text, gauge_x

    # Single-band: try to find a valid decimeter sequence
    readings = _find_decimeter_sequence(gauge_dets, orig_height)

    if not readings:
        # Fallback: use lowest confident number as direct reading
        readings = _lowest_number_fallback(gauge_dets, orig_height)

    readings.sort(key=lambda r: r.y_position)
    return readings, raw_text, gauge_x


# Roman numeral patterns
_ROMAN_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12,
}


def _parse_roman(text: str) -> int | None:
    """Parse a Roman numeral string. Returns meter value or None."""
    text_upper = text.upper().strip()
    return _ROMAN_MAP.get(text_upper)


def _multi_meter_reading(
    detections: list[tuple[str, int, int, int, float]],
    roman_markers: list[tuple[int, int, float]],
    img_height: int,
) -> list[ScaleReading]:
    """
    Handle multi-meter gauges with Roman numeral band markers.

    Roman numerals mark meter boundaries (VII=700cm, VIII=800cm).
    Arabic digits within each band are decimeters (1=10cm above lower boundary).

    Strategy:
    - Use Roman markers to define meter band boundaries
    - Assign digits to their correct band
    - Convert to absolute cm values
    - Return readings closest to the waterline (bottom of image)
    """
    if not roman_markers:
        return []

    # Sort Roman markers by y (top to bottom in image = higher meter values first)
    roman_markers.sort(key=lambda m: m[1])

    # Only consider single-digit detections for band assignment
    singles = [(t, v, x, y, c) for t, v, x, y, c in detections
               if len(t) == 1 and 1 <= v <= 9 and c >= 0.25]

    if not singles:
        return []

    # For each pair of adjacent Roman markers, calculate px_per_decimeter
    # A full meter band spans from one Roman marker to the next
    band_info = []  # (lower_meter_val, upper_meter_val, lower_y, upper_y, px_per_dm)
    for i in range(len(roman_markers) - 1):
        upper_val, upper_y, _ = roman_markers[i]    # higher in image
        lower_val, lower_y, _ = roman_markers[i + 1]  # lower in image

        # On a gauge, higher y = lower value. Roman markers go:
        # VIII (y=617) then VII (y=1401) — VIII is higher value, higher in image
        if upper_val <= lower_val:
            continue  # unexpected ordering

        band_span_px = lower_y - upper_y
        band_span_meters = upper_val - lower_val  # should be 1 usually
        if band_span_px <= 0 or band_span_meters <= 0:
            continue

        px_per_dm = band_span_px / (10 * band_span_meters)
        band_info.append((lower_val, upper_val, lower_y, upper_y, px_per_dm))

    if not band_info:
        # Single Roman marker — use digit spacing to estimate
        marker_val, marker_y, _ = roman_markers[0]
        # Find digits near this marker to compute spacing
        nearby = [(t, v, x, y, c) for t, v, x, y, c in singles
                  if abs(y - marker_y) < img_height * 0.5]
        if len(nearby) >= 2:
            nearby.sort(key=lambda d: d[3])
            # Compute average spacing between adjacent digits
            gaps = []
            for i in range(len(nearby) - 1):
                val_diff = nearby[i][1] - nearby[i + 1][1]
                y_diff = nearby[i + 1][3] - nearby[i][3]
                if val_diff > 0 and y_diff > 0:
                    gaps.append(y_diff / val_diff)
            if gaps:
                px_per_dm = sum(gaps) / len(gaps)
                # Determine which side of the marker these digits are on
                avg_digit_y = sum(d[3] for d in nearby) / len(nearby)
                if avg_digit_y < marker_y:
                    # Digits are above marker → they're in this band
                    band_info.append((marker_val - 1, marker_val, marker_y, marker_y - int(px_per_dm * 10), px_per_dm))
                else:
                    # Digits are below marker → they're in the band below
                    band_info.append((marker_val - 2, marker_val - 1, marker_y + int(px_per_dm * 10), marker_y, px_per_dm))

    if not band_info:
        return []

    # Assign each digit to a band and compute absolute cm
    readings = []
    for text, val, x, y, conf in singles:
        for lower_meter, upper_meter, lower_y, upper_y, px_per_dm in band_info:
            # Check if digit falls within this band (with some margin)
            margin = px_per_dm * 1.5
            if upper_y - margin <= y <= lower_y + margin:
                # Digit val in this band: absolute_cm = lower_meter * 100 + val * 10
                # (lower_meter is the Roman numeral at the bottom of this band)
                absolute_cm = lower_meter * 100 + val * 10
                readings.append(ScaleReading(value_cm=absolute_cm, y_position=y))
                break

    # Also add readings from the band below the lowest Roman marker
    # (extrapolate using known px_per_dm)
    lowest_roman_val, lowest_roman_y, _ = roman_markers[-1]
    best_px_per_dm = band_info[0][4] if band_info else None

    if best_px_per_dm:
        below_band_meter = lowest_roman_val - 1  # e.g., VII → VI
        for text, val, x, y, conf in singles:
            if y > lowest_roman_y:
                # This digit is below the lowest marker
                absolute_cm = below_band_meter * 100 + val * 10
                # Verify position is reasonable
                expected_y = lowest_roman_y + (10 - val) * best_px_per_dm
                if abs(y - expected_y) < best_px_per_dm * 2:
                    readings.append(ScaleReading(value_cm=absolute_cm, y_position=y))

    if len(readings) < 2:
        return []

    # Add Roman markers as readings too — they ARE scale readings (exact meter boundaries)
    for meter_val, y_pos, conf in roman_markers:
        if conf >= 0.5:
            marker_cm = meter_val * 100
            # Only add if within reasonable range of detected digit readings
            if readings:
                min_val = min(r.value_cm for r in readings)
                max_val = max(r.value_cm for r in readings)
                # Marker should be within 200cm of detected readings
                if abs(marker_cm - min_val) <= 200 or abs(marker_cm - max_val) <= 200:
                    readings.append(ScaleReading(value_cm=marker_cm, y_position=y_pos))

    # Deduplicate (same absolute value from multiple band assignments)
    seen = set()
    unique_readings = []
    for r in sorted(readings, key=lambda r: r.y_position):
        if r.value_cm not in seen:
            seen.add(r.value_cm)
            unique_readings.append(r)

    return unique_readings


def _cluster_by_x(
    detections: list[tuple[str, int, int, int, float]], img_width: int
) -> tuple[int | None, list]:
    """Find the x-column with the most number detections = the gauge."""
    if not detections:
        return None, []

    bin_width = max(int(img_width * 0.08), 50)
    bins: dict[int, list] = {}

    for det in detections:
        _, _, x, _, _ = det
        bin_idx = x // bin_width
        bins.setdefault(bin_idx, []).append(det)

    best_bin = max(bins.keys(), key=lambda k: len(bins[k]))
    best_detections = list(bins[best_bin])

    for adj in [best_bin - 1, best_bin + 1]:
        if adj in bins:
            best_detections.extend(bins[adj])

    gauge_x = int(np.mean([d[2] for d in best_detections]))
    return gauge_x, best_detections


def _find_decimeter_sequence(
    detections: list[tuple[str, int, int, int, float]], img_height: int
) -> list[ScaleReading]:
    """
    Find a sequence of single-digit numbers with consistent pixel spacing.
    These represent decimeter marks (×10 = cm value).

    On a gauge: 5, 4, 3, 2, 1 going downward with ~equal pixel gaps.
    Tolerates missing values (gaps in the sequence).
    Prefers the sequence CLOSEST TO THE BOTTOM (nearest to waterline).
    """
    # Only consider single-digit detections with decent confidence
    singles = [(t, v, x, y, c) for t, v, x, y, c in detections
               if len(t) == 1 and 0 <= v <= 9 and c >= 0.25]

    if len(singles) < 2:
        return []

    # Sort by y-position (top to bottom)
    singles.sort(key=lambda d: d[3])

    # Deduplicate at similar y-positions
    # When two detections are at the same y, prefer non-3 values (3 is likely E-mark)
    threshold = img_height * 0.03
    deduped = [singles[0]]
    for det in singles[1:]:
        if abs(det[3] - deduped[-1][3]) < threshold:
            prev = deduped[-1]
            # Prefer non-3 over 3 (E-mark filtering)
            if prev[1] == 3 and det[1] != 3:
                deduped[-1] = det
            elif det[1] == 3 and prev[1] != 3:
                pass  # keep prev (non-3)
            elif det[4] > prev[4]:
                deduped[-1] = det
        else:
            deduped.append(det)

    if len(deduped) < 2:
        return []

    # Try both strict step=-1 and monotone decreasing with consistent spacing
    step_seq = _find_step_sequence(deduped, step=-1)
    mono_seq = _find_monotone_sequence(deduped)

    # Pick the best sequence: prefer longer, then prefer closer to bottom (higher max y)
    candidates = [s for s in [step_seq, mono_seq] if len(s) >= 3]

    if not candidates:
        # Accept 2-member sequences only if both have high confidence
        candidates = [s for s in [step_seq, mono_seq]
                      if len(s) >= 2 and all(d[4] >= 0.5 for d in s)]

    if not candidates:
        return []

    # Prefer: longest first, then highest max-y (closest to water)
    best_seq = max(candidates, key=lambda s: (len(s), max(d[3] for d in s)))

    # Convert to ScaleReadings (single digit × 10 = cm)
    return [ScaleReading(value_cm=d[1] * 10, y_position=d[3]) for d in best_seq]


def _find_step_sequence(detections: list, step: int) -> list:
    """Find longest subsequence where consecutive values differ by exactly `step`."""
    best = []

    for i, start in enumerate(detections):
        seq = [start]
        expected = start[1] + step

        for j in range(i + 1, len(detections)):
            if detections[j][1] == expected:
                # Check pixel spacing consistency
                y_gap = detections[j][3] - seq[-1][3]
                if len(seq) >= 2:
                    prev_gap = seq[-1][3] - seq[-2][3]
                    if prev_gap > 0 and abs(y_gap - prev_gap) / prev_gap > 0.5:
                        continue
                if y_gap > 0:
                    seq.append(detections[j])
                    expected += step

        if len(seq) > len(best):
            best = seq

    return best


def _find_monotone_sequence(detections: list) -> list:
    """
    Find the longest monotonically decreasing subsequence in values,
    where pixel spacing is roughly consistent. Allows gaps in value.
    Prefers higher-confidence pairs when length is tied.
    """
    best = []
    best_min_conf = -1.0

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            v1, y1 = detections[i][1], detections[i][3]
            v2, y2 = detections[j][1], detections[j][3]

            if v2 >= v1 or y2 <= y1:
                continue  # Must be decreasing value with increasing y

            # This pair defines px_per_unit
            val_diff = v1 - v2
            y_diff = y2 - y1
            px_per_unit = y_diff / val_diff

            # Find all detections that fit this spacing
            seq = [detections[i], detections[j]]
            for k in range(len(detections)):
                if k == i or k == j:
                    continue
                vk, yk = detections[k][1], detections[k][3]
                expected_y = y1 + (v1 - vk) * px_per_unit
                tolerance = px_per_unit * 0.35  # 35% tolerance
                if abs(yk - expected_y) < tolerance and vk < v1 and yk > y1:
                    seq.append(detections[k])

            seq_min_conf = min(d[4] for d in seq)
            # Prefer longer sequences; when tied, prefer higher min confidence
            if len(seq) > len(best) or (len(seq) == len(best) and seq_min_conf > best_min_conf):
                best = sorted(seq, key=lambda d: d[3])
                best_min_conf = seq_min_conf

    return best


def _lowest_number_fallback(
    detections: list[tuple[str, int, int, int, float]], img_height: int
) -> list[ScaleReading]:
    """
    Fallback: when no clean sequence is found, use the lowest confident number.

    For flood gauges, multi-digit numbers (like "39") represent direct cm values.
    The lowest visible number (closest to waterline) IS approximately the reading.
    """
    # Filter E-marks: count "3" frequency
    value_counts: dict[int, int] = {}
    for _, val, _, _, _ in detections:
        value_counts[val] = value_counts.get(val, 0) + 1

    three_count = value_counts.get(3, 0)
    other_singles = [c for v, c in value_counts.items() if 0 <= v <= 9 and v != 3]
    avg_other = sum(other_singles) / max(len(other_singles), 1)
    filter_threes = three_count > max(avg_other * 1.5, 1)

    # Filter and collect valid detections
    valid = []
    for text, val, x, y, conf in detections:
        # Skip low-confidence
        if conf < 0.25:
            continue
        # Skip E-marks (standalone "3")
        if filter_threes and val == 3 and len(text) == 1:
            continue
        # Skip multi-digit numbers starting with "3" (likely E-mark + real number)
        if filter_threes and len(text) >= 2 and text.startswith("3"):
            continue
        # Skip obvious garbage (3+ digits with low confidence)
        if len(text) >= 3 and conf < 0.6:
            continue
        valid.append((text, val, x, y, conf))

    if not valid:
        # If everything was filtered, keep the highest-confidence detection
        if detections:
            best = max(detections, key=lambda d: d[4])
            val = best[1]
            # Single digit → decimeter
            if len(best[0]) == 1 and 0 <= val <= 9:
                val = val * 10
            return [ScaleReading(value_cm=val, y_position=best[3])]
        return []

    # Sort by y-position descending (bottom of image first = closest to water)
    valid.sort(key=lambda d: d[3], reverse=True)

    # The lowest number is our primary reading
    lowest = valid[0]
    val = lowest[1]

    # If it's a single digit, it's a decimeter (×10)
    if len(lowest[0]) == 1 and 0 <= val <= 9:
        val = val * 10

    # Try to find a second number above it for interpolation context
    readings = [ScaleReading(value_cm=val, y_position=lowest[3])]

    if len(valid) >= 2:
        second = valid[1]
        val2 = second[1]
        if len(second[0]) == 1 and 0 <= val2 <= 9:
            val2 = val2 * 10
        # Only include if it makes sense (higher value, lower y)
        if val2 > val and second[3] < lowest[3]:
            readings.append(ScaleReading(value_cm=val2, y_position=second[3]))

    return readings


def interpolate_water_level(
    readings: list[ScaleReading], waterline_y: int, roi_height: int
) -> float | None:
    """
    Interpolate water level from scale readings and waterline position.

    The waterline is typically BELOW (higher y than) the lowest visible number.
    Numbers decrease going down. The lowest number ≈ the current water level.

    Args:
        readings: Sorted by y_position (top to bottom)
        waterline_y: Y-coordinate of waterline in ROI
        roi_height: Height of ROI

    Returns:
        Water level in cm, or None.
    """
    if not readings:
        return None

    readings = sorted(readings, key=lambda r: r.y_position)

    if len(readings) == 1:
        # Only one number found — it IS the reading (lowest visible number)
        return float(readings[0].value_cm)

    if len(readings) == 2:
        r_top = readings[0]
        r_bot = readings[1]

        # Check if these are decimeter-derived readings (multiples of 10)
        # vs direct cm values from flood gauges
        is_decimeter = (r_top.value_cm % 10 == 0 and r_bot.value_cm % 10 == 0
                        and r_top.value_cm != r_bot.value_cm)

        if not is_decimeter:
            # Direct cm values (flood gauge): lowest number IS the reading
            # Don't trust waterline detection for these
            return float(r_bot.value_cm)

        # Decimeter gauge: interpolate or extrapolate
        if r_top.y_position <= waterline_y <= r_bot.y_position:
            return _interpolate_between(readings, waterline_y)

        # Extrapolate one step below lowest readable number
        n_marks = round((r_top.value_cm - r_bot.value_cm) / 10)
        if n_marks <= 0:
            return float(r_bot.value_cm)

        per_mark_step = 10  # cm between marks
        px_per_mark = (r_bot.y_position - r_top.y_position) / n_marks

        if px_per_mark > 0 and waterline_y > r_bot.y_position:
            # Extrapolate below lowest reading using waterline position
            y_below = waterline_y - r_bot.y_position
            cm_below = (y_below / px_per_mark) * per_mark_step
            # Cap at 1 step — with only 2 readings, can't extrapolate further reliably
            cm_below = min(cm_below, per_mark_step)
            return max(r_bot.value_cm - cm_below, 0.0)

        # Default: one step below lowest (blocked number = waterline)
        return max(r_bot.value_cm - per_mark_step, 0.0)

    # With 3+ readings (clean sequence), we can do pixel interpolation
    r_top = readings[0]
    r_bot = readings[-1]

    y_span = r_bot.y_position - r_top.y_position
    val_span = r_bot.value_cm - r_top.value_cm  # typically negative

    if y_span == 0 or val_span == 0:
        return float(r_bot.value_cm)

    cm_per_px = val_span / y_span  # negative (more y = less cm)

    # Extrapolate from lowest reading to waterline
    y_below_lowest = waterline_y - r_bot.y_position

    if y_below_lowest < 0:
        # Waterline is ABOVE the lowest reading — interpolate between readings
        return _interpolate_between(readings, waterline_y)

    # Extrapolate below the lowest reading
    water_level = r_bot.value_cm + y_below_lowest * cm_per_px

    # Sanity: limit extrapolation to prevent wild results
    step_size = abs(val_span) / max(len(readings) - 1, 1)
    # Multi-meter gauges (step_size ~10-20cm per mark) can extrapolate further
    max_steps = 2.0 if step_size <= 15 else 1.5
    min_level = r_bot.value_cm - step_size * max_steps

    if water_level < min_level:
        # Waterline detection is probably wrong, just use lowest reading
        return float(r_bot.value_cm)

    return max(water_level, 0.0)


def _interpolate_between(readings: list[ScaleReading], waterline_y: int) -> float | None:
    """Interpolate waterline between two adjacent readings."""
    for i in range(len(readings) - 1):
        r1 = readings[i]
        r2 = readings[i + 1]
        if r1.y_position <= waterline_y <= r2.y_position:
            if r2.y_position == r1.y_position:
                continue
            fraction = (waterline_y - r1.y_position) / (r2.y_position - r1.y_position)
            return r1.value_cm + fraction * (r2.value_cm - r1.value_cm)

    # Waterline above all readings — extrapolate from top
    if waterline_y < readings[0].y_position and len(readings) >= 2:
        r1, r2 = readings[0], readings[1]
        if r2.y_position != r1.y_position:
            cm_per_px = (r2.value_cm - r1.value_cm) / (r2.y_position - r1.y_position)
            return r1.value_cm + (waterline_y - r1.y_position) * cm_per_px

    return float(readings[-1].value_cm) if readings else None

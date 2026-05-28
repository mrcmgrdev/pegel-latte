"""Detect the water surface line on the gauge using multi-pass median approach."""

import cv2
import numpy as np
from statistics import median


def detect_waterline(image: np.ndarray, roi_y: int = 0, roi_height: int = 0,
                     prefer_dark_valley: bool = False) -> tuple[int | None, float]:
    """
    Detect the water surface line in a gauge ROI image.

    Uses a multi-pass approach: runs detection with multiple preprocessing
    variants (contrast, blur, CLAHE) and takes a confidence-weighted median
    of all candidates to filter out outliers.

    Args:
        image: The ROI image containing the gauge
        roi_y: Y offset of ROI in original image (for reference)
        roi_height: Height of the full ROI
        prefer_dark_valley: If True, prioritize dark-valley detection (for dirty gauges)

    Returns:
        Tuple of (y_coordinate, confidence):
        - Y-coordinate of the waterline within the ROI, or None if not detected.
        - Confidence 0.0-1.0 based on agreement between passes.
    """
    height, width = image.shape[:2]

    if height == 0 or width == 0:
        return None, 0.0

    # Generate multiple preprocessing variants
    variants = _generate_variants(image)

    # Run detection on each variant and collect all candidates with confidence
    # Each candidate is (y_position, confidence_weight)
    all_candidates: list[tuple[int, float]] = []
    for variant_bgr in variants:
        gray = cv2.cvtColor(variant_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(variant_bgr, cv2.COLOR_BGR2HSV)

        wl_edge, conf_edge = _detect_by_horizontal_edge(gray, width, height)
        wl_color, conf_color = _detect_by_color_transition(hsv, gray, height)
        wl_texture, conf_texture = _detect_by_texture_change(gray, height)

        if wl_edge is not None:
            all_candidates.append((wl_edge, conf_edge * 0.40))
        if wl_color is not None:
            all_candidates.append((wl_color, conf_color * 0.30))
        if wl_texture is not None:
            all_candidates.append((wl_texture, conf_texture * 0.15))

    # Dark-valley strategy: for dirty gauges where pattern is bright→dark→bright
    # Only run once on original (it's robust across variants)
    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wl_valley, conf_valley = _detect_by_dark_valley(gray_orig, height)
    if wl_valley is not None:
        # For multi-meter sub-ROIs with clear dark valley, trust it directly
        if prefer_dark_valley and conf_valley >= 0.6:
            return int(wl_valley), round(conf_valley, 3)
        all_candidates.append((wl_valley, conf_valley * 0.50))

    if not all_candidates:
        return None, 0.0

    # Filter outliers: remove candidates far from the weighted median
    if len(all_candidates) >= 3:
        # Compute initial weighted median
        initial_med = _weighted_median(all_candidates)
        # Keep only candidates within 12% of image height from median
        threshold = height * 0.12
        filtered = [(y, w) for y, w in all_candidates if abs(y - initial_med) <= threshold]
        if len(filtered) >= 2:
            final_y = _weighted_median(filtered)
            # Confidence based on: agreement ratio and total weight
            agreement = len(filtered) / len(all_candidates)
            total_weight = sum(w for _, w in filtered)
            max_possible_weight = len(variants) * (0.45 + 0.35 + 0.20)
            weight_ratio = min(total_weight / max_possible_weight, 1.0)
            confidence = agreement * 0.6 + weight_ratio * 0.4
            return int(final_y), round(confidence, 3)

    # Few candidates — use simple weighted median, lower confidence
    final_y = _weighted_median(all_candidates)
    total_weight = sum(w for _, w in all_candidates)
    confidence = min(total_weight / 1.0, 0.5)  # cap at 0.5 for sparse results
    return int(final_y), round(confidence, 3)


def _weighted_median(candidates: list[tuple[int, float]]) -> float:
    """Compute weighted median of (value, weight) pairs."""
    sorted_cands = sorted(candidates, key=lambda x: x[0])
    total_weight = sum(w for _, w in sorted_cands)
    if total_weight == 0:
        return sorted_cands[len(sorted_cands) // 2][0]

    cumulative = 0.0
    for y, w in sorted_cands:
        cumulative += w
        if cumulative >= total_weight / 2:
            return float(y)
    return float(sorted_cands[-1][0])


def _generate_variants(image: np.ndarray) -> list[np.ndarray]:
    """Generate multiple preprocessed variants of the image for robust detection."""
    variants = [image]  # original

    # Variant: increased contrast
    contrast_high = cv2.convertScaleAbs(image, alpha=1.4, beta=-20)
    variants.append(contrast_high)

    # Variant: decreased contrast (helps with washed-out images)
    contrast_low = cv2.convertScaleAbs(image, alpha=0.8, beta=30)
    variants.append(contrast_low)

    # Variant: CLAHE (adaptive histogram equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    variants.append(clahe_img)

    # Variant: slight Gaussian blur (smooths noise)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    variants.append(blurred)

    # Variant: stronger CLAHE
    lab2 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    lab2[:, :, 0] = clahe2.apply(lab2[:, :, 0])
    clahe_img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    variants.append(clahe_img2)

    # Variant: bilateral filter (preserves edges, smooths flat areas)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    variants.append(bilateral)

    return variants


def _detect_by_dark_valley(gray: np.ndarray, height: int) -> tuple[int | None, float]:
    """
    Detect waterline by finding bright→dark→bright pattern (dirty gauge above water).

    On dirty gauges: clean gauge is moderately bright, dirty/algae area is dark,
    water surface reflects light and is bright again. The waterline is at the
    dark→bright transition (end of the dark valley).
    """
    if height < 50:
        return None, 0.0

    row_brightness = np.mean(gray, axis=1).astype(np.float32)

    # Smooth heavily to find the macro pattern
    kernel_size = max(5, height // 15)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth = cv2.GaussianBlur(
        row_brightness.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    # Find the darkest region — start search early (skip only 10% for edge artifacts)
    search_start = height // 10
    lower_region = smooth[search_start:]
    if len(lower_region) < 20:
        return None, 0.0

    # Find the minimum brightness in the lower region
    min_idx = np.argmin(lower_region)
    min_brightness = lower_region[min_idx]
    max_brightness = np.max(smooth[:search_start]) if search_start > 0 else np.max(smooth)

    # Must have a significant dark valley (at least 30% darker than upper region)
    if max_brightness == 0 or min_brightness > max_brightness * 0.7:
        return None, 0.0

    # Find where brightness rises back to 40% of the way from min to max after the minimum
    threshold = min_brightness + (max_brightness - min_brightness) * 0.4
    abs_min_idx = search_start + min_idx

    # Search for the rising edge after the minimum
    for i in range(abs_min_idx, height):
        if smooth[i] >= threshold:
            # Found the dark→bright transition
            valley_depth = (max_brightness - min_brightness) / max_brightness
            confidence = min(valley_depth, 1.0)
            return i, confidence

    return None, 0.0


def _detect_by_horizontal_edge(gray: np.ndarray, width: int, height: int) -> tuple[int | None, float]:
    """Detect waterline by finding strong horizontal edges. Returns (y, confidence)."""
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    row_strength = np.mean(np.abs(sobel_y), axis=1)

    kernel_size = max(3, height // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    row_strength_smooth = cv2.GaussianBlur(
        row_strength.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    search_start = int(height * 0.1)
    search_end = int(height * 0.9)

    if search_start >= search_end:
        return None, 0.0

    search_region = row_strength_smooth[search_start:search_end]
    if len(search_region) == 0:
        return None, 0.0

    peak_idx = np.argmax(search_region)
    peak_value = search_region[peak_idx]
    mean_value = np.mean(search_region)

    if mean_value > 0 and peak_value > mean_value * 1.5:
        # Confidence proportional to how much stronger the peak is vs average
        ratio = peak_value / mean_value
        confidence = min((ratio - 1.5) / 3.0, 1.0)  # scale 1.5-4.5x → 0-1
        return search_start + peak_idx, max(confidence, 0.2)

    return None, 0.0


def _detect_by_color_transition(
    hsv: np.ndarray, gray: np.ndarray, height: int
) -> tuple[int | None, float]:
    """Detect waterline by color transition from air to water. Returns (y, confidence)."""
    value = hsv[:, :, 2]

    row_brightness = np.mean(value, axis=1)

    kernel_size = max(3, height // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1

    bright_smooth = cv2.GaussianBlur(
        row_brightness.reshape(-1, 1).astype(np.float32), (1, kernel_size), 0
    ).flatten()

    bright_diff = np.diff(bright_smooth)

    search_start = int(height * 0.1)
    search_end = int(height * 0.9)

    if search_start >= search_end:
        return None, 0.0

    search_region = bright_diff[search_start:search_end]
    if len(search_region) == 0:
        return None, 0.0

    min_idx = np.argmin(search_region)
    min_value = search_region[min_idx]
    std_value = np.std(search_region)

    if std_value > 0 and min_value < -std_value * 1.0:
        # Confidence proportional to how many std devs below mean
        ratio = abs(min_value) / std_value
        confidence = min((ratio - 1.0) / 3.0, 1.0)  # scale 1-4 std → 0-1
        return search_start + min_idx, max(confidence, 0.2)

    return None, 0.0


def _detect_by_texture_change(gray: np.ndarray, height: int) -> tuple[int | None, float]:
    """Detect waterline by texture transition (water has different texture). Returns (y, confidence)."""
    row_variance = np.zeros(height)
    for y in range(height):
        row_variance[y] = np.var(gray[y, :].astype(np.float32))

    kernel_size = max(3, height // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    var_smooth = cv2.GaussianBlur(
        row_variance.reshape(-1, 1).astype(np.float32), (1, kernel_size), 0
    ).flatten()

    var_diff = np.abs(np.diff(var_smooth))

    search_start = int(height * 0.15)
    search_end = int(height * 0.85)

    if search_start >= search_end:
        return None, 0.0

    search_region = var_diff[search_start:search_end]
    if len(search_region) == 0:
        return None, 0.0

    peak_idx = np.argmax(search_region)
    peak_value = search_region[peak_idx]
    mean_value = np.mean(search_region)

    if mean_value > 0 and peak_value > mean_value * 2.0:
        ratio = peak_value / mean_value
        confidence = min((ratio - 2.0) / 4.0, 1.0)  # scale 2-6x → 0-1
        return search_start + peak_idx, max(confidence, 0.15)

    return None, 0.0

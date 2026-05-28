"""Detect the water surface line on the gauge."""

import cv2
import numpy as np


def detect_waterline(image: np.ndarray, roi_y: int = 0, roi_height: int = 0) -> int | None:
    """
    Detect the water surface line in a gauge ROI image.

    The water surface is typically identified by:
    - A horizontal boundary between water (darker, blue/green) and air (lighter)
    - A change in texture/reflection patterns
    - Color transition from gauge above water to partially submerged gauge below

    Args:
        image: The ROI image containing the gauge
        roi_y: Y offset of ROI in original image (for reference)
        roi_height: Height of the full ROI

    Returns:
        Y-coordinate of the waterline within the ROI, or None if not detected.
    """
    height, width = image.shape[:2]

    if height == 0 or width == 0:
        return None

    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Strategy 1: Find horizontal edge that spans significant width
    waterline_by_edge = _detect_by_horizontal_edge(gray, width, height)

    # Strategy 2: Find color transition (water is usually darker/bluer)
    waterline_by_color = _detect_by_color_transition(hsv, gray, height)

    # Strategy 3: Find texture change (water has reflections/ripples)
    waterline_by_texture = _detect_by_texture_change(gray, height)

    # Combine strategies with weighted voting
    candidates = []
    if waterline_by_edge is not None:
        candidates.append((waterline_by_edge, 0.4))
    if waterline_by_color is not None:
        candidates.append((waterline_by_color, 0.35))
    if waterline_by_texture is not None:
        candidates.append((waterline_by_texture, 0.25))

    if not candidates:
        return None

    # If candidates are close together, average them
    if len(candidates) >= 2:
        positions = [c[0] for c in candidates]
        # Check if at least two are within 10% of image height
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if abs(positions[i] - positions[j]) < height * 0.1:
                    # Weighted average of close candidates
                    total_weight = sum(w for _, w in candidates)
                    weighted_y = sum(y * w for y, w in candidates) / total_weight
                    return int(weighted_y)

    # Return the highest-confidence single candidate
    best = max(candidates, key=lambda x: x[1])
    return best[0]


def _detect_by_horizontal_edge(gray: np.ndarray, width: int, height: int) -> int | None:
    """Detect waterline by finding strong horizontal edges."""
    # Sobel in Y direction to find horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Look for rows with strong consistent horizontal edges
    row_strength = np.mean(np.abs(sobel_y), axis=1)

    # Smooth to avoid noise
    kernel_size = max(3, height // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    row_strength_smooth = cv2.GaussianBlur(
        row_strength.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    # Find peaks in edge strength (potential waterlines)
    # Water line is typically in the middle 80% of the image
    search_start = int(height * 0.1)
    search_end = int(height * 0.9)

    if search_start >= search_end:
        return None

    search_region = row_strength_smooth[search_start:search_end]
    if len(search_region) == 0:
        return None

    # Find the strongest horizontal edge
    peak_idx = np.argmax(search_region)
    peak_value = search_region[peak_idx]

    # Threshold: edge must be significantly stronger than average
    if peak_value > np.mean(search_region) * 1.5:
        return search_start + peak_idx

    return None


def _detect_by_color_transition(
    hsv: np.ndarray, gray: np.ndarray, height: int
) -> int | None:
    """Detect waterline by finding where color transitions from air to water."""
    # Water typically has higher saturation and lower value (darker)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Compute mean saturation and brightness per row
    row_saturation = np.mean(saturation, axis=1)
    row_brightness = np.mean(value, axis=1)

    # Smooth the profiles
    kernel_size = max(3, height // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1

    sat_smooth = cv2.GaussianBlur(
        row_saturation.reshape(-1, 1).astype(np.float32), (1, kernel_size), 0
    ).flatten()
    bright_smooth = cv2.GaussianBlur(
        row_brightness.reshape(-1, 1).astype(np.float32), (1, kernel_size), 0
    ).flatten()

    # Look for the biggest drop in brightness (transition to water)
    bright_diff = np.diff(bright_smooth)

    search_start = int(height * 0.1)
    search_end = int(height * 0.9)

    if search_start >= search_end:
        return None

    search_region = bright_diff[search_start:search_end]
    if len(search_region) == 0:
        return None

    # Positive diff means getting brighter going down — we want negative (getting darker = water)
    # Actually, water is BELOW air, and y increases downward
    # So we look for the biggest increase in "darkness" going down
    min_idx = np.argmin(search_region)
    min_value = search_region[min_idx]

    if min_value < -np.std(search_region) * 1.0:
        return search_start + min_idx

    return None


def _detect_by_texture_change(gray: np.ndarray, height: int) -> int | None:
    """Detect waterline by finding texture transition (water has different texture)."""
    # Compute local variance as a measure of texture
    # Use a sliding window approach
    window_size = max(5, height // 40)

    # Compute variance in horizontal strips
    row_variance = np.zeros(height)
    for y in range(height):
        row_variance[y] = np.var(gray[y, :].astype(np.float32))

    # Smooth
    kernel_size = max(3, height // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    var_smooth = cv2.GaussianBlur(
        row_variance.reshape(-1, 1).astype(np.float32), (1, kernel_size), 0
    ).flatten()

    # Look for sudden change in texture (variance)
    var_diff = np.abs(np.diff(var_smooth))

    search_start = int(height * 0.15)
    search_end = int(height * 0.85)

    if search_start >= search_end:
        return None

    search_region = var_diff[search_start:search_end]
    if len(search_region) == 0:
        return None

    peak_idx = np.argmax(search_region)
    peak_value = search_region[peak_idx]

    if peak_value > np.mean(search_region) * 2.0:
        return search_start + peak_idx

    return None

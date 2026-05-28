"""Detect the gauge (Pegellatte) region in an image."""

import cv2
import numpy as np

from pegel_latte.models import GaugeDetection


def detect_gauge(image: np.ndarray) -> GaugeDetection | None:
    """
    Detect the vertical gauge (Pegellatte) region in the image.

    Austrian Pegellatten typically have:
    - Red/white or black/white alternating color bands
    - Vertical orientation
    - Regular numerical markings

    Returns a GaugeDetection with the ROI coordinates, or None if not found.
    """
    height, width = image.shape[:2]

    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    candidates = []

    # Strategy 1: Look for vertical structure with alternating color bands
    gauge_mask = _find_gauge_by_contrast(gray, hsv, image)
    if gauge_mask is not None:
        detection = _extract_roi_from_mask(gauge_mask, height, width)
        if detection is not None and detection.confidence > 0.3:
            candidates.append(detection)

    # Strategy 2: Look for striped pattern (alternating dark/light bands)
    detection = _find_gauge_by_stripes(gray, height, width)
    if detection is not None:
        candidates.append(detection)

    # Strategy 3: Use edge detection + Hough lines to find vertical lines
    detection = _find_gauge_by_lines(gray, height, width)
    if detection is not None:
        candidates.append(detection)

    # Return best candidate
    if candidates:
        best = max(candidates, key=lambda d: d.confidence)
        # Ensure minimum ROI width for OCR (at least 5% of image width)
        min_width = max(int(width * 0.08), 80)
        if best.roi_width < min_width:
            center_x = best.roi_x + best.roi_width // 2
            best.roi_x = max(0, center_x - min_width // 2)
            best.roi_width = min(width - best.roi_x, min_width)
        return best

    # Fallback: use center strip of image as ROI
    return GaugeDetection(
        roi_x=width // 3,
        roi_y=0,
        roi_width=width // 3,
        roi_height=height,
        confidence=0.1,
    )


def _find_gauge_by_contrast(
    gray: np.ndarray, hsv: np.ndarray, bgr: np.ndarray
) -> np.ndarray | None:
    """Find gauge by looking for high-contrast vertical striped regions."""
    height, width = gray.shape[:2]

    # Detect red regions (common in Austrian Pegellatten)
    red_mask = _detect_red_regions(hsv)

    # Also detect high-saturation colored regions (yellow, blue markers)
    high_sat = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([180, 255, 255]))

    # Detect high-contrast vertical edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # Vertical structures have strong horizontal gradients
    vertical_edges = np.abs(sobel_x).astype(np.uint8)
    _, vertical_mask = cv2.threshold(vertical_edges, 30, 255, cv2.THRESH_BINARY)

    # Combine red, saturated, and vertical edge regions
    combined = cv2.bitwise_or(red_mask, vertical_mask)
    combined = cv2.bitwise_or(combined, high_sat)

    # Morphological operations to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Filter for tall, narrow contours (gauge-like aspect ratio)
    best_contour = None
    best_score = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / max(w, 1)
        area_ratio = cv2.contourArea(contour) / max(w * h, 1)

        # Gauge should be tall and narrow (aspect ratio > 2)
        if aspect_ratio < 1.5 or h < height * 0.15:
            continue

        # Score based on height, aspect ratio, and area coverage
        score = (h / height) * min(aspect_ratio / 4.0, 1.0) * max(area_ratio, 0.3)

        # Bonus for red content in this region
        roi_red = red_mask[y:y+h, x:x+w]
        red_ratio = np.sum(roi_red > 0) / max(w * h, 1)
        if red_ratio > 0.05:
            score *= 1.5

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None

    # Create mask from best contour
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, -1)
    return mask


def _detect_red_regions(hsv: np.ndarray) -> np.ndarray:
    """Detect red-colored regions (common in Austrian Pegellatten)."""
    # Red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)


def _extract_roi_from_mask(
    mask: np.ndarray, img_height: int, img_width: int
) -> GaugeDetection | None:
    """Extract ROI bounding box from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get bounding rect of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add padding
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.05)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_width - x, w + 2 * pad_x)
    h = min(img_height - y, h + 2 * pad_y)

    return GaugeDetection(
        roi_x=x,
        roi_y=y,
        roi_width=w,
        roi_height=h,
        confidence=0.6,
    )


def _find_gauge_by_lines(
    gray: np.ndarray, img_height: int, img_width: int
) -> GaugeDetection | None:
    """Find gauge using Hough line detection for vertical lines."""
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=img_height // 5, maxLineGap=30
    )

    if lines is None:
        return None

    # Filter for near-vertical lines (within 15 degrees of vertical)
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1:
            angle = 90.0
        else:
            angle = abs(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))

        if angle > 75:  # near-vertical
            vertical_lines.append((x1, y1, x2, y2, angle))

    if not vertical_lines:
        return None

    # Cluster vertical lines by x-position to find gauge region
    x_positions = sorted([(l[0] + l[2]) / 2 for l in vertical_lines])

    if not x_positions:
        return None

    # Find the densest cluster using a sliding window
    best_cluster_x = _find_densest_cluster(x_positions, img_width * 0.1)

    # Define ROI around the cluster
    cluster_lines = [
        l for l in vertical_lines
        if abs((l[0] + l[2]) / 2 - best_cluster_x) < img_width * 0.1
    ]

    if not cluster_lines:
        return None

    min_x = min(min(l[0], l[2]) for l in cluster_lines)
    max_x = max(max(l[0], l[2]) for l in cluster_lines)
    min_y = min(min(l[1], l[3]) for l in cluster_lines)
    max_y = max(max(l[1], l[3]) for l in cluster_lines)

    # Ensure minimum width — gauge has numbers next to the lines
    roi_width = max_x - min_x
    min_roi_width = max(int(img_width * 0.08), 100)
    if roi_width < min_roi_width:
        center_x = (min_x + max_x) // 2
        min_x = center_x - min_roi_width // 2
        max_x = center_x + min_roi_width // 2

    # Add padding
    pad_x = max(30, int((max_x - min_x) * 0.3))
    pad_y = 20
    x = max(0, min_x - pad_x)
    y = max(0, min_y - pad_y)
    w = min(img_width - x, (max_x - min_x) + 2 * pad_x)
    h = min(img_height - y, (max_y - min_y) + 2 * pad_y)

    return GaugeDetection(
        roi_x=x,
        roi_y=y,
        roi_width=w,
        roi_height=h,
        confidence=0.5,
    )


def _find_gauge_by_stripes(
    gray: np.ndarray, img_height: int, img_width: int
) -> GaugeDetection | None:
    """Find gauge by detecting alternating light/dark horizontal bands (stripe pattern)."""
    # Compute vertical profile variance in sliding columns
    # A gauge has high horizontal variance within its column due to alternating bands
    col_width = max(20, img_width // 40)
    best_score = 0
    best_x = 0

    for x in range(0, img_width - col_width, col_width // 2):
        col = gray[:, x:x + col_width]
        # Compute row-wise mean
        row_means = np.mean(col, axis=1)
        # Look for oscillation (high frequency changes) = stripes
        diff = np.abs(np.diff(row_means.astype(np.float32)))
        # Score = variance of the difference (more oscillation = more stripes)
        score = np.std(diff) * (np.mean(diff) + 1)
        if score > best_score:
            best_score = score
            best_x = x

    if best_score < 5:  # threshold for "stripe-like" pattern
        return None

    # Expand around the best column to find full gauge width
    # Check neighbors for similar stripe pattern
    threshold = best_score * 0.4
    left = best_x
    right = best_x + col_width

    while left > 0:
        col = gray[:, max(0, left - col_width):left]
        row_means = np.mean(col, axis=1)
        diff = np.abs(np.diff(row_means.astype(np.float32)))
        if np.std(diff) * (np.mean(diff) + 1) < threshold:
            break
        left -= col_width // 2

    while right < img_width:
        col = gray[:, right:min(img_width, right + col_width)]
        row_means = np.mean(col, axis=1)
        diff = np.abs(np.diff(row_means.astype(np.float32)))
        if np.std(diff) * (np.mean(diff) + 1) < threshold:
            break
        right += col_width // 2

    # The stripe region defines the gauge
    gauge_width = right - left
    if gauge_width < img_width * 0.02:
        return None

    # Add padding for numbers that may be beside the stripes
    pad_x = max(40, int(gauge_width * 0.5))
    x = max(0, left - pad_x)
    w = min(img_width - x, gauge_width + 2 * pad_x)

    return GaugeDetection(
        roi_x=x,
        roi_y=0,
        roi_width=w,
        roi_height=img_height,
        confidence=0.45,
    )


def _find_densest_cluster(positions: list[float], window: float) -> float:
    """Find the center of the densest cluster in a sorted list of positions."""
    best_count = 0
    best_center = positions[len(positions) // 2]

    for i, pos in enumerate(positions):
        count = sum(1 for p in positions if abs(p - pos) <= window)
        if count > best_count:
            best_count = count
            best_center = pos

    return best_center

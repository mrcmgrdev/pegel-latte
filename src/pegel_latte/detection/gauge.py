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

    # Strategy 1: Look for vertical structure with high contrast bands
    gauge_mask = _find_gauge_by_contrast(gray, hsv, image)

    if gauge_mask is not None:
        detection = _extract_roi_from_mask(gauge_mask, height, width)
        if detection is not None:
            return detection

    # Strategy 2: Use edge detection + Hough lines to find vertical lines
    detection = _find_gauge_by_lines(gray, height, width)
    if detection is not None:
        return detection

    # Fallback: use center strip of image as ROI
    return GaugeDetection(
        roi_x=width // 4,
        roi_y=0,
        roi_width=width // 2,
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

    # Detect high-contrast vertical edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Vertical structures have strong horizontal gradients
    vertical_edges = np.abs(sobel_x).astype(np.uint8)
    _, vertical_mask = cv2.threshold(vertical_edges, 30, 255, cv2.THRESH_BINARY)

    # Look for columns with many vertical edge pixels
    col_density = np.sum(vertical_mask, axis=0) / height

    # Find regions with both red and vertical edges, or just strong vertical structure
    combined = cv2.bitwise_or(red_mask, vertical_mask)

    # Morphological operations to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
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

        # Gauge should be tall and narrow (aspect ratio > 3)
        if aspect_ratio < 2.0 or h < height * 0.2:
            continue

        # Score based on height, aspect ratio, and area coverage
        score = (h / height) * min(aspect_ratio / 5.0, 1.0) * area_ratio
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
        edges, 1, np.pi / 180, threshold=100, minLineLength=img_height // 4, maxLineGap=20
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
    x_positions = [(l[0] + l[2]) / 2 for l in vertical_lines]

    if not x_positions:
        return None

    # Find the densest cluster of vertical lines
    x_positions.sort()
    best_cluster_x = x_positions[len(x_positions) // 2]

    # Define ROI around the cluster
    cluster_lines = [
        l for l, x in zip(vertical_lines, [(l[0] + l[2]) / 2 for l in vertical_lines])
        if abs(x - best_cluster_x) < img_width * 0.15
    ]

    if not cluster_lines:
        return None

    min_x = min(l[0] for l in cluster_lines)
    max_x = max(l[2] for l in cluster_lines)
    min_y = min(min(l[1], l[3]) for l in cluster_lines)
    max_y = max(max(l[1], l[3]) for l in cluster_lines)

    # Add padding
    pad = 20
    x = max(0, min_x - pad)
    y = max(0, min_y - pad)
    w = min(img_width - x, (max_x - min_x) + 2 * pad)
    h = min(img_height - y, (max_y - min_y) + 2 * pad)

    return GaugeDetection(
        roi_x=x,
        roi_y=y,
        roi_width=w,
        roi_height=h,
        confidence=0.4,
    )

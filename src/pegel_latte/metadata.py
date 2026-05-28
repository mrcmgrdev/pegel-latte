"""Extract metadata (timestamp, GPS) from image EXIF data."""

from datetime import datetime
from pathlib import Path

import exifread
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from pegel_latte.models import ImageMetadata


def extract_metadata(image_path: Path) -> ImageMetadata:
    """Extract EXIF metadata from an image file."""
    metadata = ImageMetadata()

    # Try Pillow first for basic EXIF
    try:
        metadata = _extract_with_pillow(image_path, metadata)
    except Exception:
        pass

    # Try exifread for more robust GPS extraction
    try:
        metadata = _extract_with_exifread(image_path, metadata)
    except Exception:
        pass

    return metadata


def _extract_with_pillow(image_path: Path, metadata: ImageMetadata) -> ImageMetadata:
    """Extract EXIF data using Pillow."""
    with Image.open(image_path) as img:
        exif_data = img._getexif()
        if exif_data is None:
            return metadata

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)

            if tag == "DateTimeOriginal" or (tag == "DateTime" and metadata.timestamp is None):
                metadata.timestamp = _parse_exif_datetime(value)
            elif tag == "Make":
                metadata.camera_make = str(value).strip()
            elif tag == "Model":
                metadata.camera_model = str(value).strip()
            elif tag == "GPSInfo":
                gps = _parse_gps_info(value)
                if gps:
                    metadata.latitude = gps[0]
                    metadata.longitude = gps[1]
                    if len(gps) > 2:
                        metadata.altitude = gps[2]

    return metadata


def _extract_with_exifread(image_path: Path, metadata: ImageMetadata) -> ImageMetadata:
    """Extract EXIF data using exifread (better GPS support)."""
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    # Timestamp
    if metadata.timestamp is None:
        for key in ("EXIF DateTimeOriginal", "EXIF DateTime", "Image DateTime"):
            if key in tags:
                metadata.timestamp = _parse_exif_datetime(str(tags[key]))
                break

    # GPS
    if metadata.latitude is None:
        lat = _get_exifread_gps_coord(tags, "GPS GPSLatitude", "GPS GPSLatitudeRef")
        lon = _get_exifread_gps_coord(tags, "GPS GPSLongitude", "GPS GPSLongitudeRef")
        if lat is not None and lon is not None:
            metadata.latitude = lat
            metadata.longitude = lon

    # Altitude
    if metadata.altitude is None and "GPS GPSAltitude" in tags:
        try:
            alt_val = tags["GPS GPSAltitude"].values[0]
            metadata.altitude = float(alt_val.num) / float(alt_val.den)
            if "GPS GPSAltitudeRef" in tags and str(tags["GPS GPSAltitudeRef"]) == "1":
                metadata.altitude = -metadata.altitude
        except (AttributeError, ZeroDivisionError, IndexError):
            pass

    return metadata


def _parse_exif_datetime(dt_str: str) -> datetime | None:
    """Parse EXIF datetime string (format: 'YYYY:MM:DD HH:MM:SS')."""
    if not dt_str:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_gps_info(gps_info: dict) -> tuple[float, ...] | None:
    """Parse GPS info dict from Pillow EXIF."""
    gps_tags = {}
    for key, val in gps_info.items():
        tag = GPSTAGS.get(key, key)
        gps_tags[tag] = val

    if "GPSLatitude" not in gps_tags or "GPSLongitude" not in gps_tags:
        return None

    lat = _convert_to_degrees(gps_tags["GPSLatitude"])
    lon = _convert_to_degrees(gps_tags["GPSLongitude"])

    if gps_tags.get("GPSLatitudeRef", "N") == "S":
        lat = -lat
    if gps_tags.get("GPSLongitudeRef", "E") == "W":
        lon = -lon

    result = (lat, lon)
    if "GPSAltitude" in gps_tags:
        try:
            alt = float(gps_tags["GPSAltitude"])
            if gps_tags.get("GPSAltitudeRef", 0) == 1:
                alt = -alt
            result = (lat, lon, alt)
        except (TypeError, ValueError):
            pass

    return result


def _convert_to_degrees(value) -> float:
    """Convert GPS coordinate from degrees/minutes/seconds to decimal degrees."""
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except (TypeError, IndexError, ValueError):
        return 0.0


def _get_exifread_gps_coord(tags: dict, coord_key: str, ref_key: str) -> float | None:
    """Extract a GPS coordinate from exifread tags."""
    if coord_key not in tags:
        return None
    try:
        values = tags[coord_key].values
        d = float(values[0].num) / float(values[0].den)
        m = float(values[1].num) / float(values[1].den)
        s = float(values[2].num) / float(values[2].den)
        coord = d + (m / 60.0) + (s / 3600.0)

        if ref_key in tags and str(tags[ref_key]) in ("S", "W"):
            coord = -coord
        return coord
    except (AttributeError, ZeroDivisionError, IndexError):
        return None

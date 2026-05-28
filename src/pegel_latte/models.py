"""Data models for water level readings."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ImageMetadata:
    """Metadata extracted from image EXIF data."""

    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None


@dataclass
class GaugeDetection:
    """Intermediate result from gauge detection in an image."""

    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 0
    roi_height: int = 0
    angle: float = 0.0  # rotation angle of the gauge in degrees
    confidence: float = 0.0


@dataclass
class ScaleReading:
    """A single scale marking detected via OCR."""

    value_cm: int  # the number printed on the gauge
    y_position: int  # pixel y-position in the ROI


@dataclass
class WaterLevelReading:
    """Final result: a water level reading from an image."""

    water_level_cm: float | None = None
    confidence: float = 0.0
    source_image: Path = field(default_factory=lambda: Path("."))
    metadata: ImageMetadata = field(default_factory=ImageMetadata)
    gauge_detection: GaugeDetection | None = None
    scale_readings: list[ScaleReading] = field(default_factory=list)
    waterline_y: int | None = None  # pixel y of detected waterline in ROI
    raw_ocr_text: str | None = None
    error: str | None = None  # set if processing failed

    @property
    def timestamp(self) -> datetime | None:
        return self.metadata.timestamp

    @property
    def latitude(self) -> float | None:
        return self.metadata.latitude

    @property
    def longitude(self) -> float | None:
        return self.metadata.longitude

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary."""
        return {
            "water_level_cm": self.water_level_cm,
            "confidence": round(self.confidence, 3),
            "source_image": str(self.source_image),
            "timestamp": self.metadata.timestamp.isoformat() if self.metadata.timestamp else None,
            "latitude": self.metadata.latitude,
            "longitude": self.metadata.longitude,
            "raw_ocr_text": self.raw_ocr_text,
            "error": self.error,
        }

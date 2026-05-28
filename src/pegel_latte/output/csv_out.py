"""Output formatters for water level readings — CSV."""

import csv
from pathlib import Path

from pegel_latte.models import WaterLevelReading


CSV_FIELDS = [
    "source_image",
    "water_level_cm",
    "confidence",
    "timestamp",
    "latitude",
    "longitude",
    "raw_ocr_text",
    "error",
]


def write_csv(readings: list[WaterLevelReading], output_path: Path) -> None:
    """Write readings to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for reading in readings:
            writer.writerow(_reading_to_row(reading))


def format_csv(readings: list[WaterLevelReading]) -> str:
    """Format readings as a CSV string."""
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for reading in readings:
        writer.writerow(_reading_to_row(reading))
    return output.getvalue()


def _reading_to_row(reading: WaterLevelReading) -> dict:
    """Convert a reading to a CSV row dict."""
    return {
        "source_image": str(reading.source_image),
        "water_level_cm": reading.water_level_cm,
        "confidence": round(reading.confidence, 3) if reading.confidence else 0,
        "timestamp": reading.metadata.timestamp.isoformat() if reading.metadata.timestamp else "",
        "latitude": reading.metadata.latitude or "",
        "longitude": reading.metadata.longitude or "",
        "raw_ocr_text": reading.raw_ocr_text or "",
        "error": reading.error or "",
    }

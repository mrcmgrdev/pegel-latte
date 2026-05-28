"""Output formatters for water level readings — SQLite."""

import sqlite3
from pathlib import Path

from pegel_latte.models import WaterLevelReading


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_image TEXT NOT NULL,
    water_level_cm REAL,
    confidence REAL,
    timestamp TEXT,
    latitude REAL,
    longitude REAL,
    raw_ocr_text TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_SQL = """
INSERT INTO readings (source_image, water_level_cm, confidence, timestamp, latitude, longitude, raw_ocr_text, error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""


def write_sqlite(readings: list[WaterLevelReading], output_path: Path) -> None:
    """Write readings to a SQLite database."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(output_path))
    try:
        conn.execute(CREATE_TABLE_SQL)
        for reading in readings:
            conn.execute(
                INSERT_SQL,
                (
                    str(reading.source_image),
                    reading.water_level_cm,
                    round(reading.confidence, 3) if reading.confidence else 0,
                    reading.metadata.timestamp.isoformat() if reading.metadata.timestamp else None,
                    reading.metadata.latitude,
                    reading.metadata.longitude,
                    reading.raw_ocr_text,
                    reading.error,
                ),
            )
        conn.commit()
    finally:
        conn.close()

"""Output formatters for water level readings — JSON."""

import json
from pathlib import Path

from pegel_latte.models import WaterLevelReading


def write_json(readings: list[WaterLevelReading], output_path: Path) -> None:
    """Write readings to a JSON file."""
    data = [r.to_dict() for r in readings]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_json(readings: list[WaterLevelReading]) -> str:
    """Format readings as a JSON string."""
    data = [r.to_dict() for r in readings]
    return json.dumps(data, indent=2, ensure_ascii=False)

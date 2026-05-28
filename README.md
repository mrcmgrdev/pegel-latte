# Pegel-Latte

Read water levels from Austrian Pegellatte (water gauge) photos using computer vision and OCR.

## Features

- **Gauge detection** — finds the vertical gauge in an image using edge detection and color analysis
- **Water line detection** — detects where the water surface meets the gauge
- **Scale OCR** — reads numerical markings on the gauge using Tesseract
- **Metadata extraction** — reads EXIF timestamp and GPS coordinates from photos
- **Multiple output formats** — JSON, CSV, and SQLite

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Tesseract OCR: `brew install tesseract tesseract-lang`

## Installation

```bash
uv sync
```

## Usage

```bash
# Read a single image
uv run pegel-latte read path/to/image.jpg

# Process all images in a directory
uv run pegel-latte read data/

# Save results as CSV
uv run pegel-latte read data/ -f csv -o results.csv

# Save to SQLite database
uv run pegel-latte read data/ -f sqlite -o readings.db

# Verbose mode (saves annotated debug images)
uv run pegel-latte read data/ --verbose

# Show EXIF metadata for an image
uv run pegel-latte info path/to/image.jpg
```

## Project Structure

```
src/pegel_latte/
├── cli.py              # CLI entry point (Click)
├── reader.py           # Main orchestrator pipeline
├── models.py           # Data models (WaterLevelReading, etc.)
├── metadata.py         # EXIF extraction (timestamp, GPS)
├── detection/
│   ├── gauge.py        # Gauge region detection
│   ├── waterline.py    # Water surface detection
│   └── scale.py        # Scale OCR and interpolation
└── output/
    ├── json_out.py     # JSON output
    ├── csv_out.py      # CSV output
    └── sqlite_out.py   # SQLite output
```

## How It Works

1. **Load image** and extract EXIF metadata (timestamp, GPS)
2. **Detect gauge region** using vertical edge detection, Hough lines, and red/white color segmentation
3. **Detect water surface** by analyzing horizontal edges, color transitions, and texture changes
4. **OCR scale markings** using Tesseract, then parse and deduplicate number positions
5. **Interpolate water level** by mapping the waterline's pixel position between known scale readings

## Future Plans

- Video support (extract frames, read timestamps)
- RNW (Regulierungsniederwasser) calculation for Austrian Danube gauges
- Integration with DoRIS data (doris.bmimi.gv.at)

## Running Tests

```bash
uv run pytest tests/ -v
```

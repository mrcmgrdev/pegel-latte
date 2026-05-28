# Pegel-Latte

Read water levels from Pegellatte (water gauge staff) photos using computer vision and OCR.

Built for Austrian gauges but works with any gauge that uses numbered decimeter markings.

## Features

- **EasyOCR scene text recognition** — reads numbers directly from outdoor gauge photos
- **Multi-meter band support** — detects Roman numeral meter boundaries (VII, VIII, etc.) for absolute readings
- **Multi-pass waterline detection** — 7 preprocessing variants × 4 strategies with confidence-weighted median
- **Dirty gauge handling** — dark-valley detection for gauges with algae/grime above waterline
- **E-mark filtering** — suppresses OCR false positives from E-shaped subdivision marks
- **EXIF metadata extraction** — timestamp, GPS coordinates from photos
- **Multiple output formats** — JSON, CSV, SQLite

## Tech Stack

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — deep learning scene text detection (replaces Tesseract which failed on outdoor photos)
- **[OpenCV](https://opencv.org/)** — image processing, edge detection, color analysis
- **[Click](https://click.palletsprojects.com/)** — CLI framework
- **[ExifRead](https://github.com/ianare/exif-py)** — EXIF metadata parsing

## Installation

```bash
# Install dependencies (uv must be installed: https://docs.astral.sh/uv/getting-started/installation/)
uv sync
```

## Usage

```bash
# Read a single image
uv run pegel-latte read path/to/image.jpg

# Process all images in a directory
uv run pegel-latte read data/real/

# Output as CSV
uv run pegel-latte read data/real/ -f csv -o results.csv

# Save to SQLite database
uv run pegel-latte read data/real/ -f sqlite -o readings.db

# Verbose mode (saves annotated debug images)
uv run pegel-latte read data/real/ --verbose

# Show EXIF metadata for an image
uv run pegel-latte info path/to/image.jpg
```

### Example Output

```
Processing directory: data/real

Processed 6 image(s), 4 successful reading(s).
  ✓ 5.jpg: 700.0 cm (conf: 86.0%)
  ✓ csm_pegellatte_8d0c928fdd.webp: 10 cm (conf: 74.1%)
  ✓ eilenburg_hochwasser_pegel.jpg: 39.0 cm (conf: 72.2%)
  ✓ image.png: 10 cm (conf: 63.1%)
```

## How It Works

### Pipeline

1. **Load image** and extract EXIF metadata (timestamp, GPS)
2. **OCR full image** — EasyOCR detects all numbers + Roman numerals with their pixel positions
3. **Cluster detections** — groups numbers by x-coordinate to locate the gauge column
4. **Sequence detection** — finds monotonically decreasing digit sequences (5→4→3→2 = decimeters)
5. **Multi-meter detection** — if Roman numerals found (VII, VIII), assigns digits to meter bands and computes absolute cm values
6. **Waterline detection** — multi-pass with 7 preprocessing variants, 4 strategies (edge, color, texture, dark-valley), confidence-weighted median
7. **Interpolation** — maps waterline pixel position to cm using detected scale readings

### Gauge Types Supported

| Type | Example | How it reads |
|------|---------|-------------|
| Decimeter gauge | Digits 1-9, each = 10cm | Sequence detection, extrapolate below lowest |
| Flood gauge | Multi-digit (39, 595) | Lowest visible number = direct reading |
| Multi-meter gauge | Roman numerals + digits | Band assignment + pixel interpolation |

### Key Algorithms

- **E-mark filtering**: Austrian gauges have E-shaped (Ⅲ) subdivision marks that OCR reads as "3". Detected by frequency analysis and y-position deduplication.
- **Dark-valley detection**: For dirty/algae-covered gauges, finds the bright→dark→bright brightness pattern where dirty gauge meets reflective water surface.
- **Confidence scoring**: Combines gauge detection quality (30%), scale reading count + monotonicity (40%), and waterline detection agreement (30%).

## Project Structure

```
src/pegel_latte/
├── cli.py              # CLI entry point (Click)
├── reader.py           # Main orchestrator pipeline
├── models.py           # Data models (WaterLevelReading, ScaleReading, etc.)
├── metadata.py         # EXIF extraction (timestamp, GPS)
├── detection/
│   ├── gauge.py        # CV-based gauge region detection (fallback)
│   ├── waterline.py    # Multi-pass waterline detection (4 strategies)
│   └── scale.py        # OCR, sequence detection, interpolation
└── output/
    ├── json_out.py     # JSON output
    ├── csv_out.py      # CSV output
    └── sqlite_out.py   # SQLite output
```

## Running Tests

```bash
# Unit tests (fast, no images needed)
uv run pytest tests/ -v

# Integration tests run automatically when data/real/ contains images
```

## Data Organization

```
data/
├── real/       # Real gauge photos for testing
├── ai/         # AI-generated test images
└── reference/  # Reference images (clear numbers, no water)
```

## Limitations

- Small/low-resolution images (< 1200px) are upscaled 2× but may still fail
- Very dirty or algae-covered digit areas cannot be read by OCR
- Waterline detection depends on visible color/texture transition at water surface
- Video support not yet implemented (planned)

## Future Plans

- **Video frame extraction** — read water levels from video with timestamp sync
- **RNW calculation** — Regulierungsniederwasser and other Austrian hydrological metrics
- **DoRIS integration** — cross-reference with official Austrian waterway data
- **Improved OCR** — potentially fine-tune EasyOCR model on gauge-specific fonts


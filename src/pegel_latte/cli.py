"""CLI interface for pegel-latte."""

import sys
from pathlib import Path

import click

# Windows consoles default to cp1252, which can't encode characters that may
# appear in OCR output or error messages. Force UTF-8 so echo never crashes.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from pegel_latte.reader import read_water_level, read_directory, SUPPORTED_EXTENSIONS
from pegel_latte.output.json_out import write_json, format_json
from pegel_latte.output.csv_out import write_csv, format_csv
from pegel_latte.output.sqlite_out import write_sqlite


@click.group()
@click.version_option()
def main():
    """Pegel-Latte: Read water levels from gauge photos using CV and OCR."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["json", "csv", "sqlite"]),
    default="json",
    help="Output format for results.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path. If not specified, prints to stdout (json/csv) or writes to readings.db (sqlite).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Save debug visualization images alongside originals.",
)
def read(path: str, output_format: str, output_path: str | None, verbose: bool):
    """Read water level from an image or all images in a directory."""
    target = Path(path)

    # Collect readings
    if target.is_dir():
        click.echo(f"Processing directory: {target}")
        readings = read_directory(target, verbose=verbose)
    elif target.is_file() and target.suffix.lower() in SUPPORTED_EXTENSIONS:
        click.echo(f"Processing image: {target}")
        readings = [read_water_level(target, verbose=verbose)]
    else:
        click.echo(f"Error: Unsupported file type: {target.suffix}", err=True)
        raise SystemExit(1)

    if not readings:
        click.echo("No images found to process.", err=True)
        raise SystemExit(1)

    # Report results
    success_count = sum(1 for r in readings if r.water_level_cm is not None)
    click.echo(f"\nProcessed {len(readings)} image(s), {success_count} successful reading(s).")

    for reading in readings:
        status = "[OK]" if reading.water_level_cm is not None else "[--]"
        level = f"{reading.water_level_cm} cm" if reading.water_level_cm is not None else "N/A"
        conf = f"(conf: {reading.confidence:.1%})" if reading.confidence > 0 else ""
        error = f" - {reading.error}" if reading.error else ""
        click.echo(f"  {status} {reading.source_image.name}: {level} {conf}{error}")

    # Output
    if output_format == "json":
        if output_path:
            write_json(readings, Path(output_path))
            click.echo(f"\nResults written to: {output_path}")
        else:
            click.echo(f"\n{format_json(readings)}")
    elif output_format == "csv":
        if output_path:
            write_csv(readings, Path(output_path))
            click.echo(f"\nResults written to: {output_path}")
        else:
            click.echo(f"\n{format_csv(readings)}")
    elif output_format == "sqlite":
        db_path = Path(output_path) if output_path else Path("readings.db")
        write_sqlite(readings, db_path)
        click.echo(f"\nResults written to SQLite: {db_path}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str):
    """Show metadata (EXIF) information for an image."""
    from pegel_latte.metadata import extract_metadata

    target = Path(path)
    metadata = extract_metadata(target)

    click.echo(f"Image: {target.name}")
    click.echo(f"  Timestamp:  {metadata.timestamp or 'Not available'}")
    click.echo(f"  Latitude:   {metadata.latitude or 'Not available'}")
    click.echo(f"  Longitude:  {metadata.longitude or 'Not available'}")
    click.echo(f"  Altitude:   {metadata.altitude or 'Not available'}")
    click.echo(f"  Camera:     {metadata.camera_make or '?'} {metadata.camera_model or '?'}")


if __name__ == "__main__":
    main()

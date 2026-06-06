"""
Link road files, weather stations and best CMIP6 models.

This script reads station/model metadata from CSV files and matches them to
road-station files based on a shared station identifier extracted from file
names. It is designed for workflows where each road file has already been
linked to a corresponding weather station, and the station metadata contains
the best-performing CMIP6 model.

Default matching rule
---------------------
The station identifier is extracted as the last N characters before "_TMY".
By default, N = 6.

Example
-------
python link_roads_stations_models.py \
    --metadata-dir data/best_model_and_station \
    --road-file-dir data/roads_and_corresponding_stations \
    --output-csv outputs/destination.csv \
    --unmatched-log outputs/unmatched.txt \
    --name-column Name \
    --key-length 6
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


def extract_station_key(name: str, key_length: int = 6, delimiter: str = "_TMY") -> str:
    """
    Extract station key from a station name or filename.

    The default rule takes the last `key_length` characters before "_TMY".
    """
    name = str(name)
    key_base = name.split(delimiter)[0]
    return key_base[-key_length:]


def read_metadata_mapping(
    metadata_dir: Path,
    name_column: str = "Name",
    key_length: int = 6,
    delimiter: str = "_TMY",
) -> dict[str, dict]:
    """
    Read all metadata CSV files and build a station-key to metadata mapping.
    """
    mapping: dict[str, dict] = {}
    csv_files = sorted(metadata_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {metadata_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        if name_column not in df.columns:
            logging.warning("Column '%s' not found in %s", name_column, csv_path)
            continue

        for _, row in df.iterrows():
            key = extract_station_key(
                row[name_column],
                key_length=key_length,
                delimiter=delimiter,
            )

            if key in mapping:
                logging.warning("Duplicate station key '%s' found in %s", key, csv_path)

            mapping[key] = row.to_dict()

    return mapping


def find_road_station_files(road_file_dir: Path, extension: str = ".xlsx") -> list[Path]:
    """Recursively find road-station files."""
    files = sorted(road_file_dir.rglob(f"*{extension}"))

    if not files:
        raise FileNotFoundError(f"No '{extension}' files found in {road_file_dir}")

    return files


def match_files_to_metadata(
    metadata_mapping: dict[str, dict],
    road_station_files: list[Path],
    key_length: int = 6,
    delimiter: str = "_TMY",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Match road-station files to metadata records.
    """
    matched_records = []
    unmatched_files = []

    for file_path in road_station_files:
        file_stem = file_path.stem
        key = extract_station_key(
            file_stem,
            key_length=key_length,
            delimiter=delimiter,
        )

        if key in metadata_mapping:
            record = dict(metadata_mapping[key])
            record["matched_file"] = file_path.name
            record["matched_key"] = key
            matched_records.append(record)
        else:
            unmatched_files.append(file_stem)

    return pd.DataFrame(matched_records), unmatched_files


def write_unmatched_log(unmatched_files: list[str], log_path: Path) -> None:
    """Write unmatched filenames to a text file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as file:
        for item in unmatched_files:
            file.write(f"{item}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link road files, weather stations and best CMIP6 model metadata."
    )

    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--road-file-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--unmatched-log", required=True, type=Path)
    parser.add_argument("--name-column", default="Name")
    parser.add_argument("--key-length", default=6, type=int)
    parser.add_argument("--delimiter", default="_TMY")
    parser.add_argument("--road-file-extension", default=".xlsx")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    metadata_mapping = read_metadata_mapping(
        metadata_dir=args.metadata_dir,
        name_column=args.name_column,
        key_length=args.key_length,
        delimiter=args.delimiter,
    )

    road_station_files = find_road_station_files(
        road_file_dir=args.road_file_dir,
        extension=args.road_file_extension,
    )

    matched_df, unmatched_files = match_files_to_metadata(
        metadata_mapping=metadata_mapping,
        road_station_files=road_station_files,
        key_length=args.key_length,
        delimiter=args.delimiter,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(args.output_csv, index=False)
    write_unmatched_log(unmatched_files, args.unmatched_log)

    logging.info("Matched records: %d", len(matched_df))
    logging.info("Unmatched files: %d", len(unmatched_files))
    logging.info("Saved matched table: %s", args.output_csv)
    logging.info("Saved unmatched log: %s", args.unmatched_log)


if __name__ == "__main__":
    main()

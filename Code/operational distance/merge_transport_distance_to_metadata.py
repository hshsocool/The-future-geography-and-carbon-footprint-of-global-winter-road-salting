"""
Merge per-road transport distances back to road metadata.

This script matches parquet transport-distance files to same-named road
shapefiles, merges the per-osm_id distance columns into each shapefile, and
summarizes full-load and empty-load distances by representative road ID
(`repr_road`). The summary is then written back to a metadata Excel table.

Expected shapefile columns
--------------------------
- osm_id
- repr_road

Expected parquet columns
------------------------
- osm_id
- straight_distance_km
- Full_load_travel_distance_km
- Empty_load_travel_distance_km

Expected metadata Excel columns
-------------------------------
- osm_id

Example
-------
python merge_transport_distance_to_metadata.py \
    --road-root data/roads \
    --parquet-dir data/transport_distance/distance \
    --metadata-excel data/REP_Area_and_Length_filtered_with_transport.xlsx \
    --output-excel outputs/REP_Area_and_Length_with_transport.xlsx \
    --log-dir outputs/logs
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.errors import GEOSException
from tqdm import tqdm


def build_shp_index(root_dir: Path) -> dict[str, Path]:
    """Build {filename.lower(): full_path} index for shapefiles."""
    index = {}

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".shp"):
                index[filename.lower()] = Path(root) / filename

    return index


def find_shp_for_parquet(shp_index: dict[str, Path], parquet_path: Path) -> Path | None:
    """Find same-named shapefile for a parquet file."""
    shp_name = parquet_path.name.replace(".parquet", ".shp").lower()
    return shp_index.get(shp_name)


def to_str(series: pd.Series) -> pd.Series:
    """Convert series to stripped string."""
    return series.astype(str).str.strip()


def to_num(series: pd.Series) -> pd.Series:
    """Convert series to numeric."""
    return pd.to_numeric(series, errors="coerce")


def merge_and_summarize_distances(
    road_root: Path,
    parquet_dir: Path,
    write_shapefiles: bool,
) -> tuple[dict[str, float], dict[str, float], list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Merge distance columns to shapefiles and summarize distances by repr_road.
    """
    shp_index = build_shp_index(road_root)
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))

    full_sum_all = defaultdict(float)
    empty_sum_all = defaultdict(float)

    skipped = []
    errors = []

    logging.info("Indexed %d shapefiles", len(shp_index))
    logging.info("Found %d parquet files", len(parquet_files))

    for parquet_path in tqdm(parquet_files, desc="Merging transport distances", unit="file"):
        shp_path = find_shp_for_parquet(shp_index, parquet_path)

        if shp_path is None:
            skipped.append((parquet_path.name, "same-named shapefile not found"))
            continue

        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:
            errors.append((parquet_path.name, f"failed to read parquet: {exc}"))
            continue

        required_parquet_cols = {
            "osm_id",
            "straight_distance_km",
            "Full_load_travel_distance_km",
            "Empty_load_travel_distance_km",
        }
        missing = required_parquet_cols - set(df.columns)

        if missing:
            skipped.append((parquet_path.name, f"missing columns: {sorted(missing)}"))
            continue

        df["osm_id"] = to_str(df["osm_id"])
        for col in [
            "straight_distance_km",
            "Full_load_travel_distance_km",
            "Empty_load_travel_distance_km",
        ]:
            df[col] = to_num(df[col])

        try:
            gdf = gpd.read_file(shp_path)
        except (GEOSException, Exception) as exc:
            errors.append((parquet_path.name, f"failed to read shapefile: {exc}"))
            continue

        required_shp_cols = {"osm_id", "repr_road"}
        missing = required_shp_cols - set(gdf.columns)

        if missing:
            skipped.append((parquet_path.name, f"shapefile missing columns: {sorted(missing)}"))
            continue

        gdf["osm_id"] = to_str(gdf["osm_id"])

        distance_cols = [
            "osm_id",
            "straight_distance_km",
            "Full_load_travel_distance_km",
            "Empty_load_travel_distance_km",
        ]

        gdf = gdf.merge(df[distance_cols], on="osm_id", how="left")
        gdf["Full_load_travel_distance_km"] = to_num(
            gdf["Full_load_travel_distance_km"]
        ).fillna(0.0)
        gdf["Empty_load_travel_distance_km"] = to_num(
            gdf["Empty_load_travel_distance_km"]
        ).fillna(0.0)

        if write_shapefiles:
            try:
                gdf.to_file(shp_path, driver="ESRI Shapefile")
            except Exception as exc:
                errors.append((parquet_path.name, f"failed to write shapefile: {exc}"))
                continue

        gdf_unique = gdf.drop_duplicates(["repr_road", "osm_id"])
        grouped = (
            gdf_unique.groupby("repr_road", dropna=True)[
                ["Full_load_travel_distance_km", "Empty_load_travel_distance_km"]
            ]
            .sum()
            .reset_index()
        )

        for _, row in grouped.iterrows():
            repr_road = str(row["repr_road"])
            full_sum_all[repr_road] += float(row["Full_load_travel_distance_km"])
            empty_sum_all[repr_road] += float(row["Empty_load_travel_distance_km"])

    return full_sum_all, empty_sum_all, skipped, errors


def update_metadata_excel(
    metadata_excel: Path,
    output_excel: Path,
    full_distance_map: dict[str, float],
    empty_distance_map: dict[str, float],
) -> None:
    """Write transport-distance columns to metadata Excel."""
    metadata = pd.read_excel(metadata_excel)

    if "osm_id" not in metadata.columns:
        raise KeyError("Metadata Excel must contain an 'osm_id' column.")

    metadata["osm_id"] = metadata["osm_id"].astype(str).str.strip()
    metadata["Full_load_travel_distance_km"] = metadata["osm_id"].map(full_distance_map).astype(float)
    metadata["Empty_load_travel_distance_km"] = metadata["osm_id"].map(empty_distance_map).astype(float)

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_excel(output_excel, index=False)


def write_logs(
    log_dir: Path,
    skipped: list[tuple[str, str]],
    errors: list[tuple[str, str]],
) -> None:
    """Write processing logs."""
    log_dir.mkdir(parents=True, exist_ok=True)

    skipped_path = log_dir / "skipped_parquet_files.csv"
    errors_path = log_dir / "transport_distance_errors.csv"

    pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(skipped_path, index=False)
    pd.DataFrame(errors, columns=["file", "reason"]).to_csv(errors_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge transport distances into road metadata."
    )

    parser.add_argument("--road-root", required=True, type=Path)
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--metadata-excel", required=True, type=Path)
    parser.add_argument("--output-excel", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--write-shapefiles", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    full_map, empty_map, skipped, errors = merge_and_summarize_distances(
        road_root=args.road_root,
        parquet_dir=args.parquet_dir,
        write_shapefiles=args.write_shapefiles,
    )

    update_metadata_excel(
        metadata_excel=args.metadata_excel,
        output_excel=args.output_excel,
        full_distance_map=full_map,
        empty_distance_map=empty_map,
    )

    write_logs(args.log_dir, skipped, errors)

    logging.info("Updated metadata saved to %s", args.output_excel)
    logging.info("Skipped files: %d; errors: %d", len(skipped), len(errors))


if __name__ == "__main__":
    main()

"""
Estimate full-load and empty-load transport distances for road segments.

This script updates per-road transport-distance parquet files using:
- straight-line distance from station/depot to road segment;
- road segment length;
- country-level circuitity factor.

Formula
-------
Full-load travel distance:
    straight_distance_km * circuitity_factor + Length_km

Empty-load travel distance:
    straight_distance_km * circuitity_factor

Expected road vector files
--------------------------
- osm_id
- Length_km

Expected parquet files
----------------------
- osm_id
- straight_distance_km

Example
-------
python estimate_transport_distance.py \
    --road-root data/roads \
    --parquet-root data/transport_distance/parquet_by_shp \
    --default-circuitity 1.405
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd


DEFAULT_COUNTRY_CIRCUITY = {
    "Argentina": 1.22,
    "Australia": 1.28,
    "Belarus": 1.12,
    "Brazil": 1.23,
    "Canada": 1.30,
    "China": 1.33,
    "Egypt": 2.10,
    "Europe": 1.46,
    "England": 1.40,
    "France": 1.65,
    "Germany": 1.32,
    "Italy": 1.18,
    "Spain": 1.58,
    "Hungary": 1.35,
    "India": 1.31,
    "Indonesia": 1.43,
    "Japan": 1.41,
    "Mexico": 1.46,
    "New Zealand": 2.05,
    "Poland": 1.21,
    "Russia": 1.37,
    "Saudi Arabia": 1.34,
    "South Africa": 1.23,
    "Thailand": 1.42,
    "Turkey": 1.36,
    "Ukraine": 1.29,
    "United States": 1.20,
    "Alaska": 1.79,
    "US East": 1.20,
    "US West": 1.21,
}


SPECIAL_MAPPING = {
    "Africa": "South Africa",
    "China": "China",
    "India": "India",
    "Germany": "Germany",
    "North America": "United States",
    "UK": "England",
    "Belgium": "Europe",
    "Oceania": "Oceania",
}


def infer_circuitity_factor(
    filename: str,
    country_circuitity: dict[str, float],
    default_circuitity: float,
) -> float:
    """Infer circuitity factor from file name."""
    oceania_factor = (
        country_circuitity["Australia"] + country_circuitity["New Zealand"]
    ) / 2.0

    for prefix, mapped_country in SPECIAL_MAPPING.items():
        if filename.startswith(prefix):
            if mapped_country == "Oceania":
                return oceania_factor
            return country_circuitity.get(mapped_country, default_circuitity)

    for country, factor in country_circuitity.items():
        if country.lower() in filename.lower():
            return factor

    return default_circuitity


def update_one_transport_file(
    shp_path: Path,
    parquet_root: Path,
    country_circuitity: dict[str, float],
    default_circuitity: float,
    overwrite: bool = True,
) -> None:
    """Update one parquet file with full-load and empty-load distances."""
    filename = shp_path.stem
    parquet_path = parquet_root / f"{filename}.parquet"

    if not parquet_path.exists():
        logging.warning("Parquet not found for %s", filename)
        return

    gdf = gpd.read_file(shp_path)

    required_road_cols = {"osm_id", "Length_km"}
    missing_road_cols = required_road_cols - set(gdf.columns)
    if missing_road_cols:
        logging.warning("Skipping %s, missing columns: %s", shp_path, sorted(missing_road_cols))
        return

    gdf_subset = gdf[["osm_id", "Length_km"]].copy()
    gdf_subset["osm_id"] = gdf_subset["osm_id"].astype(str).str.strip()
    gdf_subset["Length_km"] = pd.to_numeric(gdf_subset["Length_km"], errors="coerce")

    df = pd.read_parquet(parquet_path)

    if "straight_distance_km" not in df.columns:
        logging.warning("Skipping %s, no straight_distance_km column", parquet_path)
        return

    if "osm_id" not in df.columns:
        logging.warning("Skipping %s, no osm_id column", parquet_path)
        return

    df["osm_id"] = df["osm_id"].astype(str).str.strip()
    df["straight_distance_km"] = pd.to_numeric(df["straight_distance_km"], errors="coerce")

    df = df.merge(gdf_subset, on="osm_id", how="left")

    factor = infer_circuitity_factor(
        filename=filename,
        country_circuitity=country_circuitity,
        default_circuitity=default_circuitity,
    )

    df["Full_load_travel_distance_km"] = (
        df["straight_distance_km"] * factor + df["Length_km"]
    )
    df["Empty_load_travel_distance_km"] = df["straight_distance_km"] * factor

    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

    logging.info("Updated %s with circuitity factor %.3f", parquet_path.name, factor)


def update_transport_distances(
    road_root: Path,
    parquet_root: Path,
    default_circuitity: float,
) -> None:
    """Update all parquet files corresponding to shapefiles under road_root."""
    shp_files = [
        Path(path)
        for path in glob.glob(str(road_root / "**" / "*.shp"), recursive=True)
    ]

    if not shp_files:
        raise FileNotFoundError(f"No shapefiles found in {road_root}")

    logging.info("Found %d shapefiles", len(shp_files))

    for shp_path in shp_files:
        try:
            update_one_transport_file(
                shp_path=shp_path,
                parquet_root=parquet_root,
                country_circuitity=DEFAULT_COUNTRY_CIRCUITY,
                default_circuitity=default_circuitity,
            )
        except Exception as exc:
            logging.error("Failed to process %s: %s", shp_path, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate full-load and empty-load transport distances."
    )

    parser.add_argument("--road-root", required=True, type=Path)
    parser.add_argument("--parquet-root", required=True, type=Path)
    parser.add_argument("--default-circuitity", default=1.405, type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    update_transport_distances(
        road_root=args.road_root,
        parquet_root=args.parquet_root,
        default_circuitity=args.default_circuitity,
    )

    logging.info("Transport-distance estimation completed.")


if __name__ == "__main__":
    main()

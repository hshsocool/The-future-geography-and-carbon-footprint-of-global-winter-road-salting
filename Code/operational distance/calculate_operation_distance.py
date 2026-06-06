"""
Calculate annual operational distance from salting times and transport distances.

This script multiplies annual salting frequency by full-load and empty-load
travel distances for each road segment, then summarizes results by country.

Salt frequency is calculated as the sum of selected weather-class columns.
By default, N2, NH and NC are excluded, following the original workflow.

Expected salting-times CSV columns
----------------------------------
- osm_id
- country
- year
- N1, N2, ..., N9, NH, NC or a subset of these

Expected transport metadata columns
-----------------------------------
- osm_id
- Full_load_travel_distance_km
- Empty_load_travel_distance_km

Output workbook sheets
----------------------
- summary_full_by_country
- summary_empty_by_country
- summary_total_by_country
- road_details, optional

Example
-------
python calculate_operation_distance.py \
    --salting-times data/salting_time_ssp585.csv \
    --transport-metadata data/REP_Area_and_Length_filtered_with_transport.xlsx \
    --output outputs/Operation_distance_ssp585.xlsx \
    --include-road-details
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def first_nonnull(series: pd.Series):
    """Return first non-null value in a series."""
    clean = series.dropna()
    return clean.iloc[0] if not clean.empty else np.nan


def standardize_transport_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize full-load and empty-load distance column names."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "osm_id" not in df.columns:
        raise KeyError("Transport metadata must contain 'osm_id' column.")

    df["osm_id"] = df["osm_id"].astype(str).str.strip()

    column_map = {
        "Full_load_travel_distance_km": None,
        "Empty_load_travel_distance_km": None,
    }

    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")

        if "full" in col_lower and "travel" in col_lower and "km" in col_lower:
            column_map["Full_load_travel_distance_km"] = col

        if "empty" in col_lower and "travel" in col_lower and "km" in col_lower:
            column_map["Empty_load_travel_distance_km"] = col

    if (
        column_map["Full_load_travel_distance_km"] is None
        or column_map["Empty_load_travel_distance_km"] is None
    ):
        raise KeyError(
            "Could not identify full/empty travel distance columns in transport metadata."
        )

    df = df.rename(columns={
        column_map["Full_load_travel_distance_km"]: "Full_load_travel_distance_km",
        column_map["Empty_load_travel_distance_km"]: "Empty_load_travel_distance_km",
    })

    df["Full_load_travel_distance_km"] = pd.to_numeric(
        df["Full_load_travel_distance_km"], errors="coerce"
    )
    df["Empty_load_travel_distance_km"] = pd.to_numeric(
        df["Empty_load_travel_distance_km"], errors="coerce"
    )

    deduplicated = (
        df.groupby("osm_id", as_index=False)
        .agg(
            Full_load_travel_distance_km=("Full_load_travel_distance_km", first_nonnull),
            Empty_load_travel_distance_km=("Empty_load_travel_distance_km", first_nonnull),
        )
    )

    return deduplicated


def infer_weather_columns(
    df: pd.DataFrame,
    exclude_columns: set[str],
) -> list[str]:
    """Infer weather-class columns used to count salting operations."""
    weather_cols = [
        col for col in df.columns
        if col.startswith("N") or col in {"NH", "NC"}
    ]
    use_cols = [col for col in weather_cols if col not in exclude_columns]

    if not use_cols:
        raise ValueError("No weather-class columns found. Expected N1-N9/NH/NC columns.")

    return use_cols


def calculate_operation_distance(
    salting_times_csv: Path,
    transport_metadata: Path,
    output: Path,
    excluded_weather_classes: set[str],
    include_road_details: bool,
) -> None:
    """Calculate operational distance and save country-level summaries."""
    df = pd.read_csv(salting_times_csv)
    df.columns = df.columns.str.strip()

    required = {"osm_id", "country", "year"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Salting-times CSV missing columns: {sorted(missing)}")

    df["osm_id"] = df["osm_id"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    transport = pd.read_excel(transport_metadata)
    transport = standardize_transport_columns(transport)

    use_cols = infer_weather_columns(df, excluded_weather_classes)

    for col in use_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_need = df[["osm_id", "country", "year"] + use_cols].dropna(subset=["year"]).copy()

    summary_full_list = []
    summary_empty_list = []
    summary_total_list = []
    road_details_list = []

    for year, group in df_need.groupby("year", sort=True):
        group = group.copy()
        group["salt_times"] = group[use_cols].sum(axis=1, skipna=True)
        group = group.merge(transport, on="osm_id", how="left")

        group["salt_times"] = pd.to_numeric(group["salt_times"], errors="coerce").fillna(0)
        group["Full_load_travel_distance_km"] = pd.to_numeric(
            group["Full_load_travel_distance_km"], errors="coerce"
        ).fillna(0)
        group["Empty_load_travel_distance_km"] = pd.to_numeric(
            group["Empty_load_travel_distance_km"], errors="coerce"
        ).fillna(0)

        group["operate_full_km"] = (
            group["salt_times"] * group["Full_load_travel_distance_km"]
        )
        group["operate_empty_km"] = (
            group["salt_times"] * group["Empty_load_travel_distance_km"]
        )
        group["operate_total_km"] = group["operate_full_km"] + group["operate_empty_km"]

        full_by_country = group.groupby("country", as_index=False)["operate_full_km"].sum()
        empty_by_country = group.groupby("country", as_index=False)["operate_empty_km"].sum()
        total_by_country = group.groupby("country", as_index=False)["operate_total_km"].sum()

        full_by_country["year"] = year
        empty_by_country["year"] = year
        total_by_country["year"] = year

        summary_full_list.append(full_by_country)
        summary_empty_list.append(empty_by_country)
        summary_total_list.append(total_by_country)

        if include_road_details:
            road_details_list.append(
                group[
                    [
                        "osm_id",
                        "country",
                        "year",
                        "salt_times",
                        "Full_load_travel_distance_km",
                        "Empty_load_travel_distance_km",
                        "operate_full_km",
                        "operate_empty_km",
                        "operate_total_km",
                    ]
                ]
            )

    if not summary_full_list:
        raise ValueError("No operation-distance records were generated.")

    full_long = pd.concat(summary_full_list, ignore_index=True)
    empty_long = pd.concat(summary_empty_list, ignore_index=True)
    total_long = pd.concat(summary_total_list, ignore_index=True)

    summary_full_pivot = (
        full_long.pivot(index="country", columns="year", values="operate_full_km")
        .sort_index()
        .fillna(0.0)
    )
    summary_empty_pivot = (
        empty_long.pivot(index="country", columns="year", values="operate_empty_km")
        .sort_index()
        .fillna(0.0)
    )
    summary_total_pivot = (
        total_long.pivot(index="country", columns="year", values="operate_total_km")
        .sort_index()
        .fillna(0.0)
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_full_pivot.to_excel(writer, sheet_name="summary_full_by_country")
        summary_empty_pivot.to_excel(writer, sheet_name="summary_empty_by_country")
        summary_total_pivot.to_excel(writer, sheet_name="summary_total_by_country")

        if include_road_details and road_details_list:
            road_details = pd.concat(road_details_list, ignore_index=True)
            road_details.to_excel(writer, sheet_name="road_details", index=False)

    logging.info("Saved operation-distance workbook: %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate annual operational distance from salting times and transport distances."
    )

    parser.add_argument("--salting-times", required=True, type=Path)
    parser.add_argument("--transport-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--exclude-weather-classes",
        nargs="+",
        default=["N2", "NH", "NC"],
    )
    parser.add_argument("--include-road-details", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    calculate_operation_distance(
        salting_times_csv=args.salting_times,
        transport_metadata=args.transport_metadata,
        output=args.output,
        excluded_weather_classes=set(args.exclude_weather_classes),
        include_road_details=args.include_road_details,
    )


if __name__ == "__main__":
    main()

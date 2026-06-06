"""
Calculate winter road salt demand from pavement temperature and climate data.

This script estimates daily road salt use for each road segment using threshold
rules based on pavement surface temperature, relative humidity and precipitation.

Core rule
---------
Daily precipitation is converted from kg m-2 s-1 to mm day-1 by multiplying by
86400. Salt demand is calculated as:

    daily_salt = salt_rate * road_area

where salt_rate is selected from meteorological thresholds and road_area is
provided in the road metadata table.

Expected metadata columns
-------------------------
- osm_id
- Assigned station
- Best Model
- highway
- lat
- Area
- country

Expected pavement temperature files
-----------------------------------
{temperature-root}/{highway}/{country}/{year}/osm_id_{osm_id}_{year}.parquet

Each file should contain:
- datetime
- Surface_temperature

Expected climate files
----------------------
{climate-root}/{best_model}/hurs/{station}.parquet
{climate-root}/{best_model}/pr/{station}.parquet

Each file should contain:
- time
- hurs or pr

Example
-------
python calculate_salt_demand.py \
    --metadata data/REP_Area_and_Length.xlsx \
    --temperature-root data/pavement_temperature/ssp585 \
    --climate-root data/dscale_input/ssp585 \
    --output-root outputs/salt_demand/ssp585 \
    --start-year 2020 \
    --end-year 2100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_salt_rate(
    surface_temperature: float,
    relative_humidity: float,
    precipitation_flux: float,
) -> float:
    """
    Calculate salt application rate per unit road area.

    Parameters
    ----------
    surface_temperature
        Daily mean pavement surface temperature in degrees Celsius.
    relative_humidity
        Daily relative humidity in percent.
    precipitation_flux
        Daily precipitation flux in kg m-2 s-1.

    Returns
    -------
    float
        Salt application rate per unit road area.
    """
    if pd.isna(surface_temperature) or pd.isna(relative_humidity) or pd.isna(precipitation_flux):
        return 0.0

    precipitation = precipitation_flux * 86400.0  # kg m-2 s-1 to mm day-1

    if surface_temperature > 2:
        if precipitation >= 2.0:
            return 0.0033
        if relative_humidity > 80 or 0 <= precipitation <= 1.9:
            return 0.0
        if relative_humidity <= 80:
            return 0.0003

    elif -6 <= surface_temperature <= 2:
        if precipitation >= 2.0:
            return 0.0445
        if relative_humidity > 80 or 0 <= precipitation <= 1.9:
            return 0.0133
        if relative_humidity <= 80:
            return 0.0032

    elif surface_temperature < -6:
        if precipitation >= 2.0:
            return 0.0624
        if relative_humidity > 80 or 0 <= precipitation <= 1.9:
            return 0.0116
        if relative_humidity <= 80:
            return 0.0074

    return 0.0


def calculate_salt_rates_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized implementation of the salt-demand threshold rules."""
    temp = df["Surface_temperature"]
    humidity = df["hurs"]
    precipitation = df["pr"] * 86400.0

    rate = pd.Series(0.0, index=df.index)

    valid = temp.notna() & humidity.notna() & precipitation.notna()

    mask = valid & (temp > 2) & (precipitation >= 2.0)
    rate.loc[mask] = 0.0033

    mask = valid & (temp > 2) & (humidity <= 80) & ~(precipitation.between(0, 1.9)) & (precipitation < 2.0)
    rate.loc[mask] = 0.0003

    mask = valid & (temp >= -6) & (temp <= 2) & (precipitation >= 2.0)
    rate.loc[mask] = 0.0445

    mask = valid & (temp >= -6) & (temp <= 2) & ((humidity > 80) | precipitation.between(0, 1.9))
    rate.loc[mask] = 0.0133

    mask = valid & (temp >= -6) & (temp <= 2) & (humidity <= 80) & ~(precipitation.between(0, 1.9)) & (precipitation < 2.0)
    rate.loc[mask] = 0.0032

    mask = valid & (temp < -6) & (precipitation >= 2.0)
    rate.loc[mask] = 0.0624

    mask = valid & (temp < -6) & ((humidity > 80) | precipitation.between(0, 1.9))
    rate.loc[mask] = 0.0116

    mask = valid & (temp < -6) & (humidity <= 80) & ~(precipitation.between(0, 1.9)) & (precipitation < 2.0)
    rate.loc[mask] = 0.0074

    return rate


def winter_months_for_latitude(latitude: float) -> list[int]:
    """Return winter maintenance months based on hemisphere."""
    if latitude >= 0:
        return [10, 11, 12, 1, 2, 3]
    return [4, 5, 6, 7, 8, 9]


def load_daily_surface_temperature(
    temperature_root: Path,
    osm_id: str,
    highway: str,
    country: str,
    latitude: float,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Load hourly pavement temperature and aggregate to daily winter mean."""
    frames = []
    winter_months = winter_months_for_latitude(latitude)

    for year in range(start_year, end_year + 1):
        temp_path = (
            temperature_root
            / str(highway)
            / country.lower()
            / str(year)
            / f"osm_id_{osm_id}_{year}.parquet"
        )

        if not temp_path.exists():
            continue

        df = pd.read_parquet(temp_path)

        if "datetime" not in df.columns or "Surface_temperature" not in df.columns:
            logging.warning("Missing required columns in %s", temp_path)
            continue

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).copy()
        df["date"] = df["datetime"].dt.date
        df["month"] = df["datetime"].dt.month

        df = df[df["month"].isin(winter_months)]

        daily = (
            df.groupby("date", as_index=False)["Surface_temperature"]
            .mean()
        )
        daily["year"] = year
        frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=["date", "Surface_temperature", "year"])

    return pd.concat(frames, ignore_index=True)


def load_daily_climate(
    climate_root: Path,
    best_model: str,
    station: str,
) -> pd.DataFrame:
    """Load daily relative humidity and precipitation for one station/model."""
    hurs_path = climate_root / best_model / "hurs" / f"{station}.parquet"
    pr_path = climate_root / best_model / "pr" / f"{station}.parquet"

    if not hurs_path.exists() or not pr_path.exists():
        raise FileNotFoundError(f"Missing hurs/pr files: {hurs_path} | {pr_path}")

    hurs = pd.read_parquet(hurs_path)[["time", "hurs"]]
    pr = pd.read_parquet(pr_path)[["time", "pr"]]

    hurs["date"] = pd.to_datetime(hurs["time"], errors="coerce").dt.date
    pr["date"] = pd.to_datetime(pr["time"], errors="coerce").dt.date

    hurs = hurs.dropna(subset=["date"]).drop(columns=["time"])
    pr = pr.dropna(subset=["date"]).drop(columns=["time"])

    # If the climate files are already daily, this keeps one record per date.
    hurs = hurs.groupby("date", as_index=False)["hurs"].mean()
    pr = pr.groupby("date", as_index=False)["pr"].mean()

    return hurs.merge(pr, on="date", how="outer")


def process_road_segment(
    osm_id: str,
    row: pd.Series,
    temperature_root: Path,
    climate_root: Path,
    output_root: Path,
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> None:
    """Calculate salt demand for one OSM road segment."""
    station = str(row["Assigned station"])
    best_model = str(row["Best Model"])
    highway = str(row["highway"])
    latitude = float(row["lat"])
    area = float(row["Area"])
    country = str(row["country"]).lower()

    output_path = output_root / country / highway / f"osm_id_{osm_id}.csv"

    if output_path.exists() and not overwrite:
        return

    df_temp = load_daily_surface_temperature(
        temperature_root=temperature_root,
        osm_id=osm_id,
        highway=highway,
        country=country,
        latitude=latitude,
        start_year=start_year,
        end_year=end_year,
    )

    if df_temp.empty:
        logging.warning("No pavement temperature data for osm_id=%s", osm_id)
        return

    try:
        df_climate = load_daily_climate(
            climate_root=climate_root,
            best_model=best_model,
            station=station,
        )
    except FileNotFoundError as exc:
        logging.warning("Missing climate data for osm_id=%s: %s", osm_id, exc)
        return

    df = df_temp.merge(df_climate, on="date", how="left")

    df["Salt_amount"] = calculate_salt_rates_vectorized(df) * area
    df = df[["date", "Salt_amount"]].round(4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_metadata(metadata_path: Path, model_filter: list[str] | None = None) -> pd.DataFrame:
    """Load road metadata and optionally filter by best CMIP6 model."""
    df = pd.read_excel(metadata_path)
    df["osm_id"] = df["osm_id"].astype(str)

    required = [
        "osm_id",
        "Assigned station",
        "Best Model",
        "highway",
        "lat",
        "Area",
        "country",
    ]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    if model_filter:
        df = df[df["Best Model"].isin(model_filter)]

    df = df.drop_duplicates(subset="osm_id").set_index("osm_id")
    return df


def process_all_segments(
    metadata_path: Path,
    temperature_root: Path,
    climate_root: Path,
    output_root: Path,
    start_year: int,
    end_year: int,
    model_filter: list[str] | None,
    overwrite: bool,
) -> None:
    """Process all road segments in the metadata file."""
    metadata = load_metadata(metadata_path, model_filter=model_filter)

    for osm_id, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Calculating salt demand"):
        try:
            process_road_segment(
                osm_id=osm_id,
                row=row,
                temperature_root=temperature_root,
                climate_root=climate_root,
                output_root=output_root,
                start_year=start_year,
                end_year=end_year,
                overwrite=overwrite,
            )
        except Exception as exc:
            logging.error("Failed osm_id=%s: %s", osm_id, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate winter road salt demand from pavement temperature and climate data."
    )

    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--climate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)

    parser.add_argument("--start-year", default=2020, type=int)
    parser.add_argument("--end-year", default=2100, type=int)
    parser.add_argument("--model-filter", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    process_all_segments(
        metadata_path=args.metadata,
        temperature_root=args.temperature_root,
        climate_root=args.climate_root,
        output_root=args.output_root,
        start_year=args.start_year,
        end_year=args.end_year,
        model_filter=args.model_filter,
        overwrite=args.overwrite,
    )

    logging.info("Salt-demand calculation completed.")


if __name__ == "__main__":
    main()

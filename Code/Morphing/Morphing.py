"""
Unified morphing-based downscaling script for EPW hourly weather variables.

This script generates future hourly weather series by morphing baseline EPW
weather data with monthly CMIP6 anomalies or scaling factors.

Supported variables
-------------------
- tas      -> EPW drybulb_C
- hurs     -> EPW relhum_percent
- rsds     -> EPW glohorrad_Whm2
- sfcWind  -> EPW windspd_ms

Temperature downscaling uses tas, tasmax and tasmin jointly.

Examples
--------
# Downscale all supported variables
python downscale_epw_morphing.py \
    --epw-dir data/epwfiles \
    --daily-input-dir data/dscale_input/ssp245 \
    --monthly-mean-dir data/dscale_input/monthly_mean/ssp245 \
    --output-dir outputs/dscale_output/ssp245 \
    --index-file data/time_index_2015_2100.csv \
    --variables tas hurs rsds sfcWind \
    --start-year 2015 \
    --end-year 2100

# Downscale only temperature
python downscale_epw_morphing.py \
    --epw-dir data/epwfiles \
    --daily-input-dir data/dscale_input/ssp585 \
    --monthly-mean-dir data/dscale_input/monthly_mean/ssp585 \
    --output-dir outputs/dscale_output/ssp585 \
    --index-file data/time_index_2015_2100.csv \
    --variables tas
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


EPW_COLUMNS = [
    "month", "day", "hour", "minute", "datasource", "drybulb_C", "dewpoint_C",
    "relhum_percent", "atmos_Pa", "exthorrad_Whm2", "extdirrad_Whm2",
    "horirsky_Whm2", "glohorrad_Whm2", "dirnorrad_Whm2", "difhorrad_Whm2",
    "glohorillum_lux", "dirnorillum_lux", "difhorillum_lux", "zenlum_lux",
    "winddir_deg", "windspd_ms", "totskycvr_tenths", "opaqskycvr_tenths",
    "visibility_km", "ceiling_hgt_m", "presweathobs", "presweathcodes",
    "precip_wtr_mm", "aerosol_opt_thousandths", "snowdepth_cm",
    "days_last_snow", "albedo", "liq_precip_depth_mm", "liq_precip_rate_hour",
]


DEFAULT_MODELS = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "CESM2",
    "CMCC-ESM2",
    "EC-Earth3",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "NorESM2-MM",
]


VARIABLE_TO_EPW_COLUMN = {
    "tas": "drybulb_C",
    "hurs": "relhum_percent",
    "rsds": "glohorrad_Whm2",
    "sfcWind": "windspd_ms",
}


def read_epw(epw_path: Path) -> pd.DataFrame:
    """Read an EPW file and return hourly weather data with a synthetic datetime index."""
    df = pd.read_csv(
        epw_path,
        skiprows=8,
        header=None,
        names=EPW_COLUMNS,
        encoding="utf-8",
        encoding_errors="ignore",
    ).drop(columns=["datasource"])

    # EPW hour is 1--24. Convert to 0--23 for pandas datetime.
    df["datetime"] = pd.to_datetime(
        {
            "year": 2023,
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": df["hour"].astype(int) - 1,
            "minute": df["minute"].astype(int),
        },
        errors="coerce",
    )

    df = df.set_index("datetime")
    return df


def monthly_mean(series: pd.Series) -> pd.Series:
    """Daily mean followed by monthly mean."""
    return series.resample("D").mean().resample("ME").mean().reset_index(drop=True)


def monthly_min_mean_max(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Monthly means of daily maximum, daily minimum and daily mean."""
    monthly_max = series.resample("D").max().resample("ME").mean().reset_index(drop=True)
    monthly_min = series.resample("D").min().resample("ME").mean().reset_index(drop=True)
    monthly_avg = series.resample("D").mean().resample("ME").mean().reset_index(drop=True)
    return monthly_max, monthly_min, monthly_avg


def read_monthly_csv(path: Path, variable: str) -> pd.DataFrame:
    """Read a monthly-mean CSV file and return a DateTime-indexed dataframe."""
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError(f"'time' column not found in {path}")

    if variable not in df.columns:
        raise ValueError(f"'{variable}' column not found in {path}")

    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def future_monthly_for_year(
    future_df: pd.DataFrame,
    variable: str,
    year: int,
    convert_kelvin_to_celsius: bool = False,
) -> pd.Series:
    """Extract future monthly means for one year."""
    year_df = future_df[future_df.index.year == year]

    if year_df.empty:
        raise ValueError(f"No monthly data found for year {year}")

    values = year_df[variable]

    if convert_kelvin_to_celsius:
        values = values - 273.15

    return monthly_mean(values)


def morph_temperature(
    epw_df: pd.DataFrame,
    fut_tas: pd.Series,
    fut_tasmax: pd.Series,
    fut_tasmin: pd.Series,
) -> pd.Series:
    """
    Morph dry-bulb temperature using monthly mean, maximum and minimum changes.

    Formula follows the common morphing structure:
    T_future = T_EPW + delta_T_mean
               + scale_T_range * (T_EPW - T_EPW_monthly_mean)
    """
    epw = epw_df.copy()

    hist_tmax, hist_tmin, hist_tmean = monthly_min_mean_max(epw["drybulb_C"])

    delta_mean = fut_tas.reset_index(drop=True) - hist_tmean
    delta_max = fut_tasmax.reset_index(drop=True) - hist_tmax
    delta_min = fut_tasmin.reset_index(drop=True) - hist_tmin

    historical_range = hist_tmax - hist_tmin
    future_range_change = delta_max - delta_min

    scale = future_range_change / historical_range.replace(0, np.nan)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0)

    months = np.arange(1, 13)
    delta_mean_map = dict(zip(months, delta_mean))
    scale_map = dict(zip(months, scale))
    hist_mean_map = dict(zip(months, hist_tmean))

    result = (
        epw["drybulb_C"]
        + epw["month"].map(delta_mean_map)
        + epw["month"].map(scale_map)
        * (epw["drybulb_C"] - epw["month"].map(hist_mean_map))
    )

    return result.round(1).rename("drybulb_C")


def morph_relative_humidity(
    epw_df: pd.DataFrame,
    fut_hurs: pd.Series,
) -> pd.Series:
    """Morph relative humidity using monthly multiplicative scaling."""
    epw = epw_df.copy()
    hist_hurs = monthly_mean(epw["relhum_percent"])

    scale = fut_hurs.reset_index(drop=True) / hist_hurs.replace(0, np.nan)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    scale_map = dict(zip(np.arange(1, 13), scale))
    result = epw["relhum_percent"] * epw["month"].map(scale_map)
    result = np.clip(result, 1, 100)

    return pd.Series(result, index=epw.index).round(1).rename("relhum_percent")


def morph_solar_radiation(
    epw_df: pd.DataFrame,
    fut_rsds: pd.Series,
) -> pd.Series:
    """
    Morph global horizontal radiation using monthly additive change converted
    to a multiplicative hourly scaling factor.
    """
    epw = epw_df.copy()

    hist_rsds = monthly_mean(epw["glohorrad_Whm2"])
    delta = fut_rsds.reset_index(drop=True) - hist_rsds

    scale = 1.0 + delta / hist_rsds.replace(0, np.nan)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    scale_map = dict(zip(np.arange(1, 13), scale))
    result = epw["glohorrad_Whm2"] * epw["month"].map(scale_map)
    result = np.clip(result, 0, None)

    return pd.Series(result, index=epw.index).round(0).astype(int).rename("glohorrad_Whm2")


def morph_wind_speed(
    epw_df: pd.DataFrame,
    fut_sfcwind: pd.Series,
) -> pd.Series:
    """Morph wind speed using monthly multiplicative scaling."""
    epw = epw_df.copy()

    hist_sfcwind = monthly_mean(epw["windspd_ms"])

    scale = fut_sfcwind.reset_index(drop=True) / hist_sfcwind.replace(0, np.nan)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    scale_map = dict(zip(np.arange(1, 13), scale))
    result = epw["windspd_ms"] * epw["month"].map(scale_map)
    result = np.clip(result, 0, None)

    return pd.Series(result, index=epw.index).round(1).rename("windspd_ms")


def downscale_one_location_variable(
    epw_df: pd.DataFrame,
    monthly_mean_dir: Path,
    model: str,
    location_name: str,
    variable: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Downscale one variable for one location and one model."""
    output_column = VARIABLE_TO_EPW_COLUMN[variable]
    annual_results = []

    if variable == "tas":
        tas_df = read_monthly_csv(monthly_mean_dir / model / "tas" / f"{location_name}.csv", "tas")
        tasmax_df = read_monthly_csv(monthly_mean_dir / model / "tasmax" / f"{location_name}.csv", "tasmax")
        tasmin_df = read_monthly_csv(monthly_mean_dir / model / "tasmin" / f"{location_name}.csv", "tasmin")

        for year in range(start_year, end_year + 1):
            fut_tas = future_monthly_for_year(tas_df, "tas", year, convert_kelvin_to_celsius=True)
            fut_tasmax = future_monthly_for_year(tasmax_df, "tasmax", year, convert_kelvin_to_celsius=True)
            fut_tasmin = future_monthly_for_year(tasmin_df, "tasmin", year, convert_kelvin_to_celsius=True)

            morphed = morph_temperature(
                epw_df=epw_df,
                fut_tas=fut_tas,
                fut_tasmax=fut_tasmax,
                fut_tasmin=fut_tasmin,
            )
            annual_results.append(pd.DataFrame({output_column: morphed.values}))

    else:
        future_df = read_monthly_csv(
            monthly_mean_dir / model / variable / f"{location_name}.csv",
            variable,
        )

        for year in range(start_year, end_year + 1):
            fut_monthly = future_monthly_for_year(
                future_df,
                variable,
                year,
                convert_kelvin_to_celsius=False,
            )

            if variable == "hurs":
                morphed = morph_relative_humidity(epw_df, fut_monthly)
            elif variable == "rsds":
                morphed = morph_solar_radiation(epw_df, fut_monthly)
            elif variable == "sfcWind":
                morphed = morph_wind_speed(epw_df, fut_monthly)
            else:
                raise ValueError(f"Unsupported variable: {variable}")

            annual_results.append(pd.DataFrame({output_column: morphed.values}))

    return pd.concat(annual_results, ignore_index=True)


def get_locations_from_daily_input(
    daily_input_dir: Path,
    model: str,
    variable: str,
) -> list[str]:
    """
    Infer location names from daily input CSV files.

    The daily files are used only to identify which locations should be processed.
    Monthly mean files are used for the actual morphing calculation.
    """
    location_dir = daily_input_dir / model / variable

    if not location_dir.exists():
        logging.warning("Missing input directory: %s", location_dir)
        return []

    return sorted(path.stem for path in location_dir.glob("*.csv"))


def process_all(
    epw_dir: Path,
    daily_input_dir: Path,
    monthly_mean_dir: Path,
    output_dir: Path,
    index_file: Path,
    models: list[str],
    variables: list[str],
    start_year: int,
    end_year: int,
    overwrite: bool,
) -> None:
    """Process all requested variables, models and locations."""
    index_df = pd.read_csv(index_file)

    if "datetime" not in index_df.columns:
        raise ValueError(f"'datetime' column not found in {index_file}")

    time_index = index_df["datetime"]

    epw_cache: dict[str, pd.DataFrame] = {}

    for model in models:
        logging.info("Processing model: %s", model)

        for variable in variables:
            if variable not in VARIABLE_TO_EPW_COLUMN:
                raise ValueError(
                    f"Unsupported variable '{variable}'. "
                    f"Supported variables are: {list(VARIABLE_TO_EPW_COLUMN)}"
                )

            location_variable = "tas" if variable == "tas" else variable
            location_names = get_locations_from_daily_input(
                daily_input_dir=daily_input_dir,
                model=model,
                variable=location_variable,
            )

            if not location_names:
                continue

            for location_name in location_names:
                epw_path = epw_dir / f"{location_name}.epw"

                if not epw_path.exists():
                    logging.warning("EPW file not found: %s", epw_path)
                    continue

                output_path = output_dir / model / variable / f"{location_name}.csv"

                if output_path.exists() and not overwrite:
                    logging.info("Skipping existing file: %s", output_path)
                    continue

                if location_name not in epw_cache:
                    epw_cache[location_name] = read_epw(epw_path)

                try:
                    output_df = downscale_one_location_variable(
                        epw_df=epw_cache[location_name],
                        monthly_mean_dir=monthly_mean_dir,
                        model=model,
                        location_name=location_name,
                        variable=variable,
                        start_year=start_year,
                        end_year=end_year,
                    )

                    if len(output_df) != len(time_index):
                        raise ValueError(
                            f"Length mismatch for {location_name}, {model}, {variable}: "
                            f"{len(output_df)} rows generated; {len(time_index)} rows in index."
                        )

                    output_df.insert(0, "datetime", time_index.values)

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_df.to_csv(output_path, index=False)

                    logging.info("Saved: %s", output_path)

                except Exception as exc:
                    logging.error(
                        "Failed: model=%s, variable=%s, location=%s | %s",
                        model,
                        variable,
                        location_name,
                        exc,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified EPW morphing-based downscaling for CMIP6 variables."
    )

    parser.add_argument("--epw-dir", required=True, type=Path)
    parser.add_argument("--daily-input-dir", required=True, type=Path)
    parser.add_argument("--monthly-mean-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index-file", required=True, type=Path)

    parser.add_argument(
        "--variables",
        nargs="+",
        default=["tas", "hurs", "rsds", "sfcWind"],
        choices=["tas", "hurs", "rsds", "sfcWind"],
    )

    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--start-year", default=2015, type=int)
    parser.add_argument("--end-year", default=2100, type=int)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    process_all(
        epw_dir=args.epw_dir,
        daily_input_dir=args.daily_input_dir,
        monthly_mean_dir=args.monthly_mean_dir,
        output_dir=args.output_dir,
        index_file=args.index_file,
        models=args.models,
        variables=args.variables,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )

    logging.info("Downscaling completed.")


if __name__ == "__main__":
    main()

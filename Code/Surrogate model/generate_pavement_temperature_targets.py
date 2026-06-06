"""
Generate pavement temperature training targets from weather forcing.

This script implements the physics-based pavement temperature calculation used
to generate target variables for surrogate-model training. It reads hourly
weather CSV files, computes pavement surface and asphalt-layer temperatures,
and writes updated CSV files.

Expected input columns
----------------------
- datetime
- drybulb_C
- relhum_percent
- glohorrad_Whm2
- windspd_ms
- D0
- D1
- D2
- Country_Cluster

Expected vapour pressure table
------------------------------
An Excel file with two columns:
- air temperature
- saturated vapour pressure

Example
-------
python generate_pavement_temperature_targets.py \
    --input-dir data/physics_input \
    --output-dir outputs/physics_targets \
    --vapour-pressure-table data/pvap.xlsx \
    --vapour-pressure-sheet pvap
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy import interpolate


TARGET_COLUMNS = [
    "Surface temperature",
    "Layer1 temperature",
    "Layer2 temperature",
    "Layer3 temperature",
]


def sin18(hour: float, phase_shift: float) -> float:
    """Sinusoidal correction used in asphalt layer temperature estimation."""
    x = hour

    if phase_shift == 15.5:
        if 0 <= x <= 5:
            x += 24
        elif 5 < x < 11:
            x = 11

    elif phase_shift == 13.5:
        if 0 < x <= 3:
            x += 24
        elif 3 < x < 9:
            x = 9

    angle = 2 * np.pi * (x - phase_shift) / 18
    return float(np.sin(angle))


def load_vapour_pressure_interpolator(
    table_path: Path,
    sheet_name: str = "pvap",
) -> interpolate.interp1d:
    """Load saturated vapour pressure table and return interpolation function."""
    table = pd.read_excel(table_path, sheet_name=sheet_name, usecols="A:B")
    return interpolate.interp1d(
        table.iloc[:, 0],
        table.iloc[:, 1],
        fill_value="extrapolate",
    )


def validate_input_columns(df: pd.DataFrame, file_path: Path) -> None:
    """Check that all required input columns exist."""
    required = [
        "datetime",
        "drybulb_C",
        "relhum_percent",
        "glohorrad_Whm2",
        "windspd_ms",
        "D0",
        "D1",
        "D2",
    ]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")


def calculate_pavement_temperatures(
    df: pd.DataFrame,
    vapour_pressure_fn,
    asphalt_absorptivity: float = 0.93,
    asphalt_emissivity: float = 0.93,
    asphalt_thermal_conductivity: float = 1.0811,
    tolerance: float = 0.03,
    progress_interval: int = 1000,
) -> pd.DataFrame:
    """
    Calculate pavement surface and asphalt-layer temperatures.

    The surface temperature is solved using a Newton-Raphson energy-balance
    iteration. Layer temperatures are estimated using empirical asphalt
    temperature equations.
    """
    df = df.copy()

    datetime = pd.to_datetime(df["datetime"])
    temp_air = df["drybulb_C"].astype(float).reset_index(drop=True)
    rh = df["relhum_percent"].astype(float).reset_index(drop=True)
    solar = df["glohorrad_Whm2"].astype(float).reset_index(drop=True)
    wind_speed = df["windspd_ms"].astype(float).reset_index(drop=True)

    n = len(df)
    hour = datetime.dt.hour.reset_index(drop=True)

    temp_prev_day = pd.concat(
        [temp_air.iloc[:24], temp_air.iloc[: max(n - 24, 0)]],
        ignore_index=True,
    )

    if len(temp_prev_day) < n:
        temp_prev_day = temp_prev_day.reindex(range(n)).ffill().bfill()

    shortwave = asphalt_absorptivity * solar
    sigma = 5.68e-8

    pvap_max = vapour_pressure_fn(temp_air)
    pvap = rh / 100.0 * pvap_max

    emissivity_air = 0.77 - 0.28 * (10 ** (-0.074 * pvap))
    longwave = emissivity_air * sigma * ((temp_air + 273.15) ** 4)

    pavement_temp = pd.DataFrame(
        np.zeros((n, len(TARGET_COLUMNS))),
        columns=TARGET_COLUMNS,
    )

    eps = np.finfo(float).eps

    for k in range(n):
        if progress_interval > 0 and (k + 1) % progress_interval == 0:
            logging.info("Completed %.2f%%", (k + 1) / n * 100)

        d0 = float(df.loc[df.index[k], "D0"])
        d1 = float(df.loc[df.index[k], "D1"])
        d2 = float(df.loc[df.index[k], "D2"])

        layer_depths = [d0, d1, d2]
        depth_temp = 0.01 * d0

        if depth_temp == 0:
            depth_temp = eps

        z_temp = list(np.cumsum([0] + layer_depths))

        if k == 0:
            temp_surface = 1.5 * temp_air.iloc[0]
            temp_depth = 0.7 * temp_air.iloc[0]
        else:
            temp_surface = 0.5 * (pavement_temp.iloc[k - 1, 0] + temp_air.iloc[k])
            temp_depth = pavement_temp.iloc[k - 1, 1]

        while True:
            mean_kelvin = mean([temp_surface + 273.15, temp_air.iloc[k] + 273.15])

            convective_coefficient = 698.24 * (
                0.00144 * (mean_kelvin ** 0.3) * (wind_speed.iloc[k] ** 0.7)
                + 0.00097 * abs(temp_surface - temp_air.iloc[k]) ** 0.3
            )

            residual = (
                shortwave.iloc[k]
                + longwave.iloc[k]
                - asphalt_emissivity * sigma * (temp_surface + 273.15) ** 4
                + asphalt_thermal_conductivity * (1 / depth_temp) * (temp_depth - temp_surface)
                - convective_coefficient * (temp_surface - temp_air.iloc[k])
            )

            derivative = (
                -4 * asphalt_emissivity * sigma * (temp_surface + 273.15) ** 3
                - asphalt_thermal_conductivity * (1 / depth_temp)
                - convective_coefficient
                - (temp_surface - temp_air.iloc[k])
                * 698.24
                * (
                    0.00144
                    * 0.3
                    * (mean_kelvin ** -0.3)
                    * 0.5
                    * wind_speed.iloc[k] ** 0.7
                    + 0.00097
                    * 0.3
                    * abs(temp_surface - temp_air.iloc[k] + eps) ** (-0.3)
                )
            )

            if abs(residual) < tolerance:
                break

            temp_surface = temp_surface - residual / derivative

        pavement_temp.iloc[k, 0] = temp_surface

        layer_boundary_temps = [0.0] * (len(layer_depths) + 1)
        layer_boundary_temps[0] = temp_surface

        for z in range(1, len(z_temp)):
            layer_boundary_temps[z] = (
                2.78
                + 0.912 * temp_air.iloc[k]
                + (np.log10(10 * z_temp[z]) - 1.25)
                * (-0.428 * temp_air.iloc[k] + 0.553 * temp_prev_day.iloc[k])
                + 2.63 * sin18(hour.iloc[k], 15.5)
                + 0.027 * temp_air.iloc[k] * sin18(hour.iloc[k], 13.5)
            )

        layer_mean_temps = [
            mean([layer_boundary_temps[t], layer_boundary_temps[t + 1]])
            for t in range(len(layer_depths))
        ]

        pavement_temp.iloc[k, 1:4] = layer_mean_temps

    for col in TARGET_COLUMNS:
        df[col] = pavement_temp[col].values

    return df


def process_file(
    input_file: Path,
    output_file: Path,
    vapour_pressure_fn,
    overwrite: bool,
    progress_interval: int,
) -> None:
    """Process one CSV file."""
    if output_file.exists() and not overwrite:
        logging.info("Skipping existing file: %s", output_file)
        return

    df = pd.read_csv(input_file)
    validate_input_columns(df, input_file)

    output_df = calculate_pavement_temperatures(
        df,
        vapour_pressure_fn=vapour_pressure_fn,
        progress_interval=progress_interval,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    logging.info("Saved: %s", output_file)


def process_folder(
    input_dir: Path,
    output_dir: Path,
    vapour_pressure_table: Path,
    vapour_pressure_sheet: str,
    overwrite: bool,
    progress_interval: int,
) -> None:
    """Process all CSV files in all subdirectories."""
    vapour_pressure_fn = load_vapour_pressure_interpolator(
        vapour_pressure_table,
        sheet_name=vapour_pressure_sheet,
    )

    csv_files = sorted(input_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for input_file in csv_files:
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path

        try:
            process_file(
                input_file=input_file,
                output_file=output_file,
                vapour_pressure_fn=vapour_pressure_fn,
                overwrite=overwrite,
                progress_interval=progress_interval,
            )
        except Exception as exc:
            logging.error("Failed to process %s: %s", input_file, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pavement temperature targets from hourly weather inputs."
    )

    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vapour-pressure-table", required=True, type=Path)
    parser.add_argument("--vapour-pressure-sheet", default="pvap")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", default=1000, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vapour_pressure_table=args.vapour_pressure_table,
        vapour_pressure_sheet=args.vapour_pressure_sheet,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
    )

    logging.info("Pavement temperature target generation completed.")


if __name__ == "__main__":
    main()

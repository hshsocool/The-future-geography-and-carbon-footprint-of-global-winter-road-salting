"""
Plot global operation distance projections.

This script generates a 3 x 2 global map figure showing:
- absolute national winter road maintenance operation distance in 2020
- changes in 2050 and 2100 relative to 2020

The 2020 baseline is taken from SSP2-4.5 by default.

Expected input
--------------
Excel files with a sheet named "summary_total_by_country".
Rows should be ISO3 country codes and columns should include 2020, 2050 and 2100.

Example
-------
python plot_global_operation_distance.py \
    --shapefile data/ne_110m_admin_0_countries.shp \
    --ssp126-xlsx data/Operation_distance_ssp126.xlsx \
    --ssp245-xlsx data/Operation_distance_ssp245.xlsx \
    --china-province-geojson data/china_provinces.geojson \
    --iso-fixer fix_country_iso_codes.py \
    --output outputs/fig4_operation_distance.png \
    --output-svg outputs/fig4_operation_distance.svg
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.titleweight"] = "normal"
matplotlib.rcParams["axes.labelweight"] = "normal"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def load_optional_iso_fixer(path: Path | None):
    """Load an optional ISO-code correction function."""
    if path is None:
        return None

    spec = importlib.util.spec_from_file_location("fix_country_iso_codes", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ISO fixer from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "fix_map_iso_codes"):
        raise AttributeError(f"'fix_map_iso_codes' not found in {path}")

    return module.fix_map_iso_codes


def load_operation_distance(filepath: Path, sheet_name: str = "summary_total_by_country") -> pd.DataFrame:
    """Read country-level operation distance and convert to million km."""
    df = pd.read_excel(filepath, index_col=0, sheet_name=sheet_name)
    df.index = df.index.astype(str).str.upper().str.strip()
    df.columns = df.columns.astype(str).str.strip()

    if "TCD" in df.index and "NER" not in df.index:
        df.loc["NER"] = df.loc["TCD"]

    return df / 1e6


def add_china_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign mainland China values to Taiwan, Hong Kong and Macao if needed."""
    df = df.copy()

    if "CHN" in df.index:
        for region in ["TWN", "HKG", "MAC"]:
            df.loc[region] = df.loc["CHN"]

    return df


def get_data_driven_bounds(series: pd.Series, num_bins: int = 8) -> list[float]:
    """Generate percentile-based bounds for absolute values."""
    clean = series.dropna()

    if clean.empty:
        raise ValueError("Cannot generate bounds from an empty series.")

    percentiles = np.percentile(clean, np.linspace(0, 100, num_bins + 1))
    bounds = sorted(set(np.round(percentiles, 2)))

    if len(bounds) < 2:
        bounds = [float(clean.min()), float(clean.max())]

    return bounds


def add_latitude_labels(ax) -> None:
    """Add latitude labels on the left side of a Robinson map."""
    for lat in [-60, -30, 0, 30, 60]:
        label = f"{abs(lat)}°S" if lat < 0 else ("0°" if lat == 0 else f"{lat}°N")

        ax.annotate(
            label,
            xy=(-180, lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=18,
            color="dimgray",
            clip_on=False,
        )


def get_layout():
    """Return manual axes layout parameters for the 3 x 2 map panel."""
    left_margin = 0.035
    right_margin = 0.035
    top_edge = 0.90
    bottom_edge = 0.10

    hgap = -0.11
    vgap_top = 0.045
    vgap_bottom = 0.045

    cbar_gap = 0.003
    cbar_w = 0.018
    shift_col2 = 0.015
    shift_cbar = -0.012

    n_cols = 2
    n_rows = 3

    total_w = 1.0 - left_margin - right_margin
    col_w = (total_w - cbar_w - (n_cols - 1) * hgap) / n_cols

    lefts = [
        left_margin,
        left_margin + col_w + hgap + shift_col2,
    ]

    cbar_left = left_margin + col_w + hgap + col_w + cbar_gap + shift_cbar

    total_h = top_edge - bottom_edge
    row_h = (total_h - vgap_top - vgap_bottom) / n_rows

    bottoms = [
        top_edge - row_h,
        top_edge - row_h - (row_h + vgap_top),
        top_edge - row_h - (row_h + vgap_top) - (row_h + vgap_bottom),
    ]

    return lefts, bottoms, col_w, row_h, cbar_left, cbar_w


def build_colormaps(all_2020_values: pd.Series):
    """Build absolute and delta colour maps."""
    delta_bounds = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
    delta_colors = [
        "#01665e",
        "#35978f",
        "#80cdc1",
        "#c7eae5",
        "#f5f5f5",
        "#dfc27d",
        "#bf812d",
        "#8c510a",
    ]
    delta_cmap = ListedColormap(delta_colors)
    delta_norm = BoundaryNorm(delta_bounds, len(delta_colors))

    value_bounds = get_data_driven_bounds(all_2020_values, num_bins=8)
    value_cmap = ListedColormap(
        [
            "#c7eae5",
            "#a9ddd7",
            "#8dcfc9",
            "#80cdc1",
            "#5bb8ac",
            "#3fa094",
            "#2a7f74",
            "#01665e",
        ]
    )
    value_norm = BoundaryNorm(value_bounds, len(value_bounds) - 1)

    return value_bounds, value_cmap, value_norm, delta_bounds, delta_cmap, delta_norm


def plot_operation_distance_map(
    shapefile: Path,
    scenario_files: list[Path],
    labels: list[str],
    output: Path,
    output_svg: Path | None = None,
    china_province_geojson: Path | None = None,
    iso_fixer: Path | None = None,
    baseline_label: str = "SSP2-4.5",
    dpi: int = 600,
) -> None:
    """Create global maps of operation distance and future changes."""
    if len(scenario_files) != len(labels):
        raise ValueError("scenario_files and labels must have the same length.")

    if len(labels) != 2:
        raise ValueError("This layout expects exactly two scenario labels.")

    world = gpd.read_file(shapefile)

    fix_map_iso_codes = load_optional_iso_fixer(iso_fixer)
    if fix_map_iso_codes is not None:
        world = fix_map_iso_codes(world)

    if "ISO_A3" not in world.columns:
        raise ValueError("World shapefile must contain an 'ISO_A3' column or use --iso-fixer.")

    world = world[["ISO_A3", "geometry"]].copy()

    china_admin = None
    if china_province_geojson is not None:
        china_admin = gpd.read_file(china_province_geojson)

    data = {
        label: add_china_regions(load_operation_distance(path))
        for label, path in zip(labels, scenario_files)
    }

    if baseline_label not in data:
        raise ValueError(f"Baseline label '{baseline_label}' not found.")

    baseline = data[baseline_label]["2020"]
    baseline.index = baseline.index.astype(str).str.upper()

    all_2020_values = pd.concat([data[label]["2020"] for label in labels])

    (
        value_bounds,
        value_cmap,
        value_norm,
        delta_bounds,
        delta_cmap,
        delta_norm,
    ) = build_colormaps(all_2020_values)

    sm_abs = plt.cm.ScalarMappable(cmap=value_cmap, norm=value_norm)
    sm_delta = plt.cm.ScalarMappable(cmap=delta_cmap, norm=delta_norm)

    fig = plt.figure(figsize=(22, 18), facecolor="white")

    rows = [
        ("2020", "abs"),
        ("2050", "delta"),
        ("2100", "delta"),
    ]
    row_labels = [
        "2020",
        "2050 (relative to 2020)",
        "2100 (relative to 2020)",
    ]
    sub_labels = ["a", "b", "c", "d", "e", "f"]

    lefts, bottoms, col_w, row_h, cbar_left, cbar_w = get_layout()

    for col_idx, label in enumerate(labels):
        df = data[label].copy()
        df.index = df.index.astype(str).str.upper()

        for row_idx, (year, plot_type) in enumerate(rows):
            ax = fig.add_axes(
                [lefts[col_idx], bottoms[row_idx], col_w, row_h],
                projection=ccrs.Robinson(),
            )

            ax.set_global()
            ax.set_facecolor("#e6f2fa")
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="#e6f2fa")
            ax.coastlines(linewidth=0.4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

            gridlines = ax.gridlines(
                draw_labels=False,
                linewidth=0.6,
                color="gray",
                alpha=0.7,
                linestyle="--",
            )
            gridlines.xlocator = plt.FixedLocator(range(-180, 181, 60))
            gridlines.ylocator = plt.FixedLocator([-60, -30, 0, 30, 60])

            add_latitude_labels(ax)

            if plot_type == "abs":
                values = df["2020"]
                cmap, norm = value_cmap, value_norm
            else:
                values = (df[year] - baseline).fillna(0)
                values = values.clip(np.min(delta_bounds), np.max(delta_bounds))
                cmap, norm = delta_cmap, delta_norm

            merged = world.merge(
                values.rename("value"),
                left_on="ISO_A3",
                right_index=True,
                how="left",
            )

            merged.plot(
                column="value",
                cmap=cmap,
                norm=norm,
                ax=ax,
                transform=ccrs.PlateCarree(),
                edgecolor="black",
                linewidth=0.2,
                legend=False,
                missing_kwds={
                    "color": "#e0e0e0",
                    "edgecolor": "lightgray",
                },
            )

            if china_admin is not None:
                china_admin.to_crs(ccrs.PlateCarree().proj4_init).plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.4,
                    linestyle="--",
                )

            if row_idx == 0:
                ax.set_title(label, fontsize=26, fontweight="normal", pad=26)

            if col_idx == 0:
                ax.text(
                    -0.11,
                    0.50,
                    row_labels[row_idx],
                    va="center",
                    ha="center",
                    fontsize=26,
                    fontweight="normal",
                    rotation="vertical",
                    transform=ax.transAxes,
                    clip_on=False,
                )

            label_idx = col_idx * len(rows) + row_idx
            ax.text(
                0.01,
                1.03,
                sub_labels[label_idx],
                transform=ax.transAxes,
                fontsize=28,
                fontweight="normal",
                ha="left",
                va="top",
                clip_on=False,
            )

    # Absolute colour bar
    cbar1_height_factor = 0.88
    cbar1_height = row_h * cbar1_height_factor
    cbar1_bottom = bottoms[0] + (row_h - cbar1_height) / 2

    cbar_ax1 = fig.add_axes([cbar_left, cbar1_bottom, cbar_w, cbar1_height])
    cb1 = fig.colorbar(
        sm_abs,
        cax=cbar_ax1,
        orientation="vertical",
        boundaries=value_bounds,
        ticks=value_bounds,
        spacing="uniform",
    )
    cb1.ax.set_yticklabels([f"{int(round(x))}" for x in value_bounds])
    cb1.set_label(
        "Operation distance (million km)",
        fontsize=22,
        fontweight="normal",
        labelpad=12,
    )
    cb1.ax.tick_params(labelsize=18)

    # Delta colour bar
    cbar2_full_bottom = bottoms[2]
    cbar2_full_top = bottoms[1] + row_h
    cbar2_full_height = cbar2_full_top - cbar2_full_bottom

    cbar2_height_factor = 0.90
    cbar2_height = cbar2_full_height * cbar2_height_factor
    cbar2_bottom = cbar2_full_bottom + (cbar2_full_height - cbar2_height) / 2

    cbar_ax2 = fig.add_axes([cbar_left, cbar2_bottom, cbar_w, cbar2_height])
    cb2 = fig.colorbar(
        sm_delta,
        cax=cbar_ax2,
        orientation="vertical",
        boundaries=delta_bounds,
        ticks=delta_bounds,
        spacing="uniform",
    )
    cb2.ax.set_yticklabels([f"{x:g}" for x in delta_bounds])
    cb2.set_label(
        "Change in operation distance (million km)",
        fontsize=22,
        fontweight="normal",
        labelpad=12,
    )
    cb2.ax.tick_params(labelsize=18)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.15)

    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_svg,
            format="svg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={"Date": None},
        )

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot global operation distance projections."
    )

    parser.add_argument("--shapefile", required=True, type=Path)
    parser.add_argument("--ssp126-xlsx", required=True, type=Path)
    parser.add_argument("--ssp245-xlsx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-svg", default=None, type=Path)
    parser.add_argument("--china-province-geojson", default=None, type=Path)
    parser.add_argument("--iso-fixer", default=None, type=Path)
    parser.add_argument("--baseline-label", default="SSP2-4.5")
    parser.add_argument("--dpi", default=600, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_operation_distance_map(
        shapefile=args.shapefile,
        scenario_files=[args.ssp126_xlsx, args.ssp245_xlsx],
        labels=["SSP1-2.6", "SSP2-4.5"],
        output=args.output,
        output_svg=args.output_svg,
        china_province_geojson=args.china_province_geojson,
        iso_fixer=args.iso_fixer,
        baseline_label=args.baseline_label,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

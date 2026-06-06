"""
Plot global pavement surface temperature projections.

This script generates a 3 x 2 global map figure showing:
- absolute pavement surface temperature in 2020
- changes in 2050 and 2100 relative to 2020

Supported layout
----------------
Rows:
    2020 absolute value
    2050 minus 2020
    2100 minus 2020

Columns:
    SSP1-2.6
    SSP2-4.5

Example
-------
python plot_global_pavement_temperature.py \
    --ssp126-csv data/surface_temperature_ssp126.csv \
    --ssp245-csv data/surface_temperature_ssp245.csv \
    --road-coordinates data/road_coordinates.csv \
    --world-shapefile data/ne_110m_admin_0_countries.shp \
    --output outputs/fig1_global_pavement_temperature.png \
    --output-svg outputs/fig1_global_pavement_temperature.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap


def configure_matplotlib() -> None:
    """Set global matplotlib style for publication-quality vector output."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def add_latitude_labels(ax, fontsize: int = 18) -> None:
    """Add latitude labels on the left side of a Robinson map."""
    for lat in [-60, -30, 0, 30, 60]:
        label = f"{abs(lat)}°S" if lat < 0 else ("0°" if lat == 0 else f"{lat}°N")

        ax.annotate(
            label,
            xy=(-180, lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(-8, 0),
            textcoords="offset points",
            fontsize=fontsize,
            va="center",
            ha="right",
            color="dimgray",
            clip_on=False,
        )


def build_colormaps():
    """Build absolute-temperature and temperature-change colour maps."""
    levels_abs = np.arange(5, 36, 5)
    cmap_abs = ListedColormap(
        ["#4575b4", "#74add1", "#abd9e9", "#fdae61", "#f46d43", "#d73027"]
    )
    norm_abs = BoundaryNorm(levels_abs, cmap_abs.N)

    levels_delta = np.arange(0, 5.5, 0.5)
    cmap_delta = LinearSegmentedColormap.from_list(
        "delta_cmap",
        ["#fee8c8", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f"],
        N=len(levels_delta) - 1,
    )
    norm_delta = BoundaryNorm(levels_delta, cmap_delta.N)

    return levels_abs, cmap_abs, norm_abs, levels_delta, cmap_delta, norm_delta


def get_layout():
    """Return manual axes layout parameters for the 3 x 2 map panel."""
    left_margin = 0.035
    right_margin = 0.035
    top_edge = 0.90
    bottom_edge = 0.10

    hgap = -0.11
    vgap_top = 0.040
    vgap_bottom = 0.040

    cbar_gap = -0.003
    cbar_w = 0.018

    n_cols = 2
    n_rows = 3

    total_w = 1.0 - left_margin - right_margin
    col_w = (total_w - cbar_w - (n_cols - 1) * hgap) / n_cols

    shift_col2 = 0.015
    lefts = [
        left_margin,
        left_margin + col_w + hgap + shift_col2,
    ]

    shift_cbar = -0.012
    cbar_left = left_margin + col_w + hgap + col_w + cbar_gap + shift_cbar

    total_h = top_edge - bottom_edge
    row_h = (total_h - vgap_top - vgap_bottom) / n_rows

    bottoms = [
        top_edge - row_h,
        top_edge - row_h - (row_h + vgap_top),
        top_edge - row_h - (row_h + vgap_top) - (row_h + vgap_bottom),
    ]

    return lefts, bottoms, col_w, row_h, cbar_left, cbar_w


def plot_global_pavement_temperature(
    ssp126_csv: Path,
    ssp245_csv: Path,
    road_coordinates: Path,
    world_shapefile: Path,
    output: Path,
    output_svg: Path | None = None,
    dpi: int = 600,
) -> None:
    """Create global pavement surface temperature projection maps."""
    configure_matplotlib()

    csv_paths = {
        "ssp126": ssp126_csv,
        "ssp245": ssp245_csv,
    }

    scenarios = ["ssp126", "ssp245"]
    scenario_titles = ["SSP1-2.6", "SSP2-4.5"]

    rows = [
        ("2020", "abs"),
        ("2050", "delta"),
        ("2100", "delta"),
    ]
    row_labels = ["2020", "2050 (relative to 2020)", "2100 (relative to 2020)"]
    sub_labels = ["a", "b", "c", "d", "e", "f"]

    levels_abs, cmap_abs, norm_abs, levels_delta, cmap_delta, norm_delta = build_colormaps()

    roads_coords = (
        pd.read_csv(road_coordinates)[["osm_id", "lon", "lat"]]
        .drop_duplicates()
    )
    world = gpd.read_file(world_shapefile)

    fig = plt.figure(figsize=(22, 18), facecolor="white")
    lefts, bottoms, col_w, row_h, cbar_left, cbar_w = get_layout()

    sc_abs = None
    sc_delta = None

    for col_idx, scenario in enumerate(scenarios):
        df_surface = pd.read_csv(csv_paths[scenario])
        base_2020 = df_surface[["osm_id", "2020"]].rename(columns={"2020": "base_2020"})

        for row_idx, (year, plot_type) in enumerate(rows):
            ax = fig.add_axes(
                [lefts[col_idx], bottoms[row_idx], col_w, row_h],
                projection=ccrs.Robinson(),
            )

            ax.set_global()
            ax.set_facecolor("#e6f2fa")
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="#e6f2fa")

            ax.gridlines(
                draw_labels=False,
                xlocs=np.arange(-180, 181, 60),
                ylocs=[-60, -30, 0, 30, 60],
                linestyle="--",
                linewidth=0.6,
                color="gray",
                alpha=0.7,
            )

            add_latitude_labels(ax)

            if plot_type == "abs":
                data = df_surface[["osm_id", year]].rename(columns={year: "value"})
                merged = roads_coords.merge(data, on="osm_id", how="inner")

                sc = ax.scatter(
                    merged["lon"],
                    merged["lat"],
                    c=merged["value"],
                    cmap=cmap_abs,
                    norm=norm_abs,
                    s=2,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                sc_abs = sc

            else:
                future = df_surface[["osm_id", year]].rename(columns={year: "future"})
                delta = future.merge(base_2020, on="osm_id", how="inner")
                delta["value"] = delta["future"] - delta["base_2020"]

                merged = roads_coords.merge(
                    delta[["osm_id", "value"]],
                    on="osm_id",
                    how="inner",
                )

                sc = ax.scatter(
                    merged["lon"],
                    merged["lat"],
                    c=merged["value"],
                    cmap=cmap_delta,
                    norm=norm_delta,
                    s=2,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                sc_delta = sc

            sc.set_rasterized(True)

            world.boundary.plot(
                ax=ax,
                linewidth=0.6,
                edgecolor="lightgray",
                transform=ccrs.PlateCarree(),
            )

            if row_idx == 0:
                ax.set_title(
                    scenario_titles[col_idx],
                    fontsize=26,
                    fontweight="normal",
                    pad=18,
                )

            if col_idx == 0:
                ax.text(
                    -0.11,
                    0.50,
                    row_labels[row_idx],
                    transform=ax.transAxes,
                    fontsize=26,
                    fontweight="normal",
                    rotation="vertical",
                    va="center",
                    ha="center",
                    clip_on=False,
                )

            label_idx = col_idx * len(rows) + row_idx
            ax.text(
                0.01,
                1.03,
                sub_labels[label_idx],
                transform=ax.transAxes,
                fontsize=30,
                fontweight="normal",
                ha="left",
                va="top",
                clip_on=False,
            )

    if sc_abs is None or sc_delta is None:
        raise RuntimeError("No scatter data were plotted. Check input files.")

    # Absolute-temperature colour bar
    cbar1_height_factor = 0.88
    cbar1_height = row_h * cbar1_height_factor
    cbar1_bottom = bottoms[0] + (row_h - cbar1_height) / 2

    cbar_ax1 = fig.add_axes([cbar_left, cbar1_bottom, cbar_w, cbar1_height])
    cb1 = fig.colorbar(
        sc_abs,
        cax=cbar_ax1,
        orientation="vertical",
        ticks=levels_abs,
        boundaries=levels_abs,
        spacing="proportional",
    )
    cb1.set_label("Surface temperature (°C)", fontsize=22, fontweight="normal", labelpad=12)
    cb1.ax.tick_params(labelsize=18)

    # Delta-temperature colour bar
    cbar2_full_bottom = bottoms[2]
    cbar2_full_top = bottoms[1] + row_h
    cbar2_full_height = cbar2_full_top - cbar2_full_bottom

    cbar2_height_factor = 0.88
    cbar2_height = cbar2_full_height * cbar2_height_factor
    cbar2_bottom = cbar2_full_bottom + (cbar2_full_height - cbar2_height) / 2

    cbar_ax2 = fig.add_axes([cbar_left, cbar2_bottom, cbar_w, cbar2_height])
    cb2 = fig.colorbar(
        sc_delta,
        cax=cbar_ax2,
        orientation="vertical",
        ticks=levels_delta,
        boundaries=levels_delta,
        spacing="proportional",
    )
    cb2.set_label(
        "Change in surface temperature (°C)",
        fontsize=22,
        fontweight="normal",
        labelpad=12,
    )
    cb2.ax.set_yticklabels([f"{x:g}" for x in levels_delta])
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
        description="Plot global pavement surface temperature projections."
    )

    parser.add_argument("--ssp126-csv", required=True, type=Path)
    parser.add_argument("--ssp245-csv", required=True, type=Path)
    parser.add_argument("--road-coordinates", required=True, type=Path)
    parser.add_argument("--world-shapefile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-svg", default=None, type=Path)
    parser.add_argument("--dpi", default=600, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_global_pavement_temperature(
        ssp126_csv=args.ssp126_csv,
        ssp245_csv=args.ssp245_csv,
        road_coordinates=args.road_coordinates,
        world_shapefile=args.world_shapefile,
        output=args.output,
        output_svg=args.output_svg,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

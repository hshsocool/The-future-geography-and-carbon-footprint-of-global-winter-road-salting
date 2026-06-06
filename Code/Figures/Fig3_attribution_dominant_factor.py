"""
Plot dominant attribution factors for changes in road salt demand.

This script generates one map per scenario-year combination showing the
dominant driver of national salt-demand change. The dominant factor is defined
as the variable with the largest absolute contribution among:
- temperature change
- relative humidity change
- precipitation change

Expected input
--------------
Each attribution summary Excel file may contain one or multiple sheets.
Required columns:
- country
- target_year, or sheet names that can be parsed as target years
- temp_change
- hurs_change
- precip_change
- total_change

Example
-------
python plot_attribution_dominant_factor.py \
    --ssp126-xlsx data/Attribution_Summary_ssp126.xlsx \
    --ssp245-xlsx data/Attribution_Summary_ssp245.xlsx \
    --world-shapefile data/ne_110m_admin_0_countries.shp \
    --china-geojson data/china_provinces.geojson \
    --output-dir outputs/fig3_attribution \
    --target-years 2050 2100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy
from shapely.geometry import LineString, MultiLineString


FACTOR_COLORS = {
    "temp": "#D9523A",
    "hurs": "#E4D33C",
    "precip": "#417AB5",
}
MISSING_COLOR = "#e0e0e0"


def configure_matplotlib() -> None:
    """Set publication-style matplotlib defaults."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["font.size"] = 22
    plt.rcParams["axes.titlesize"] = 26
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 22
    plt.rcParams["ytick.labelsize"] = 22
    plt.rcParams["legend.fontsize"] = 22
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def read_summary_all_sheets(xlsx_path: Path) -> pd.DataFrame:
    """Read attribution results from all sheets in an Excel workbook."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    frames = []

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()

        if "target_year" not in tmp.columns:
            try:
                tmp["target_year"] = int(str(sheet_name).strip())
            except ValueError:
                logging.warning(
                    "Sheet name could not be parsed as target_year: %s in %s",
                    sheet_name,
                    xlsx_path,
                )

        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    keep_columns = [
        "country",
        "target_year",
        "temp_change",
        "hurs_change",
        "precip_change",
        "total_change",
    ]
    available_columns = [col for col in keep_columns if col in df_all.columns]

    return df_all[available_columns].copy()


def choose_iso_column(world: gpd.GeoDataFrame) -> str:
    """Choose a usable ISO3 column from a Natural Earth-style country layer."""
    for column in ["ISO_A3_EH", "ISO_A3", "ADM0_A3", "SOV_A3", "WB_A3", "GU_A3"]:
        if column in world.columns:
            return column

    raise KeyError(f"No ISO3-like field found. Columns include: {list(world.columns)[:20]}")


def compute_dominant_factor(row: pd.Series) -> str | None:
    """Return the attribution factor with the largest absolute contribution."""
    values = [
        row.get("temp_change"),
        row.get("hurs_change"),
        row.get("precip_change"),
    ]

    if any(pd.isna(value) for value in values):
        return None

    if all(abs(value) == 0 for value in values):
        return None

    index = int(np.argmax(np.abs(values)))
    return ["temp", "hurs", "precip"][index]


def add_graticules_with_left_latlabels(ax, show_left: bool = True) -> None:
    """Add graticules and optional latitude labels on the left side."""
    xlocs = np.arange(-180, 181, 60)
    ylocs = [-60, -30, 0, 30, 60]

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=xlocs,
        ylocs=ylocs,
        linewidth=0.2,
        color="lightgray",
        alpha=0.6,
        linestyle="--",
    )

    if not show_left:
        return

    lon_min, _, _, _ = ax.get_extent(crs=ccrs.PlateCarree())
    base_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    text_transform = offset_copy(base_transform, x=-4, units="points", fig=ax.figure)

    for lat in [0, 30, 60, -30, -60]:
        label = "0°" if lat == 0 else f"{abs(lat)}°{'N' if lat > 0 else 'S'}"
        ax.text(
            lon_min,
            lat,
            label,
            transform=text_transform,
            ha="right",
            va="center",
            fontsize=20,
            clip_on=False,
            zorder=5,
        )


def draw_nine_dash_line_only(ax, china_gdf: gpd.GeoDataFrame | None) -> None:
    """Draw only the nine-dash-line-like geometries from a China boundary layer."""
    if china_gdf is None or china_gdf.empty:
        return

    lon_min, lon_max = 105, 122
    lat_min, lat_max = 0, 25

    for geom in china_gdf.geometry:
        if geom is None:
            continue

        boundary = geom if isinstance(geom, (LineString, MultiLineString)) else geom.boundary
        lines = [boundary] if isinstance(boundary, LineString) else getattr(boundary, "geoms", [])

        for line in lines:
            representative = line.representative_point()
            cx, cy = float(representative.x), float(representative.y)

            if lon_min <= cx <= lon_max and lat_min <= cy <= lat_max:
                x, y = line.xy
                ax.plot(
                    x,
                    y,
                    transform=ccrs.PlateCarree(),
                    color="lightgray",
                    linewidth=0.1,
                    linestyle=(0, (10, 6)),
                    zorder=4,
                )


def plot_single_panel(
    ax,
    world_gdf: gpd.GeoDataFrame,
    china_gdf: gpd.GeoDataFrame | None,
    colors: pd.Series,
    title: str,
    show_left_labels: bool = True,
) -> None:
    """Plot one attribution map panel."""
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgrey", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", edgecolor="none", zorder=0)

    facecolors = colors.reindex(world_gdf.index).fillna(MISSING_COLOR).values

    world_gdf.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        facecolor=facecolors,
        edgecolor="white",
        linewidth=0.3,
        zorder=2,
    )

    draw_nine_dash_line_only(ax, china_gdf)
    add_graticules_with_left_latlabels(ax, show_left=show_left_labels)

    ax.set_title(title, pad=16)


def sanitize_filename(text: str) -> str:
    """Create a filesystem-friendly filename stem."""
    return text.replace(" ", "")


def load_world_layer(world_shapefile: Path) -> gpd.GeoDataFrame:
    """Load and prepare world boundaries."""
    world = gpd.read_file(world_shapefile)

    if not world.is_valid.all():
        world = world.copy()
        world["geometry"] = world.buffer(0)

    world = world.to_crs("EPSG:4326")

    iso_col = choose_iso_column(world)
    world["iso3"] = world[iso_col].astype(str).str.strip().str.lower()
    world.loc[world["iso3"].isin(["-99", "nan", "none", ""]), "iso3"] = np.nan

    return world


def plot_attribution_maps(
    scenario_files: dict[str, Path],
    world_shapefile: Path,
    china_geojson: Path | None,
    output_dir: Path,
    target_years: list[int],
) -> None:
    """Generate attribution maps for all scenarios and target years."""
    configure_matplotlib()

    output_dir.mkdir(parents=True, exist_ok=True)

    world = load_world_layer(world_shapefile)

    china = None
    if china_geojson is not None:
        china = gpd.read_file(china_geojson).to_crs(world.crs)

    scenario_data = {}
    for scenario_name, xlsx_path in scenario_files.items():
        df = read_summary_all_sheets(xlsx_path)

        if not df.empty:
            df["country"] = df["country"].astype(str).str.strip().str.lower()
            df = df[df["target_year"].isin(target_years)].copy()

        scenario_data[scenario_name] = df

    for scenario_name, df_all in scenario_data.items():
        if df_all is None or df_all.empty:
            logging.warning("Empty attribution data: %s", scenario_name)
            continue

        for year in target_years:
            subset = df_all[df_all["target_year"] == year].copy()

            if subset.empty:
                logging.warning("Empty subset: %s, %s", scenario_name, year)
                continue

            subset["dominant_factor"] = subset.apply(compute_dominant_factor, axis=1)

            merged = world.merge(subset, left_on="iso3", right_on="country", how="left")

            # Assign China's dominant factor to Taiwan when needed.
            china_factor = merged.loc[merged["iso3"] == "chn", "dominant_factor"].dropna()
            if not china_factor.empty:
                merged.loc[merged["iso3"] == "twn", "dominant_factor"] = china_factor.iloc[0]

            colors = merged["dominant_factor"].map(FACTOR_COLORS).fillna(MISSING_COLOR)

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection=ccrs.Robinson())

            title = f"{scenario_name}  {year}-2020"
            plot_single_panel(ax, merged, china, colors, title)

            safe_scenario = sanitize_filename(scenario_name)
            filename_stem = f"{safe_scenario}_{year}-2020"

            png_path = output_dir / f"{filename_stem}.png"
            svg_path = output_dir / f"{filename_stem}.svg"

            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            fig.savefig(svg_path, dpi=600, bbox_inches="tight", metadata={"Date": None})
            plt.close(fig)

            logging.info("Saved: %s", png_path)
            logging.info("Saved: %s", svg_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dominant attribution factor maps."
    )

    parser.add_argument("--ssp126-xlsx", required=True, type=Path)
    parser.add_argument("--ssp245-xlsx", required=True, type=Path)
    parser.add_argument("--world-shapefile", required=True, type=Path)
    parser.add_argument("--china-geojson", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-years", nargs="+", default=[2050, 2100], type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    scenario_files = {
        "SSP1-2.6": args.ssp126_xlsx,
        "SSP2-4.5": args.ssp245_xlsx,
    }

    plot_attribution_maps(
        scenario_files=scenario_files,
        world_shapefile=args.world_shapefile,
        china_geojson=args.china_geojson,
        output_dir=args.output_dir,
        target_years=args.target_years,
    )


if __name__ == "__main__":
    main()

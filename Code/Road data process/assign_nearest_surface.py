"""
Assign missing road surface material information using nearest-neighbour search.

This script fills missing values in the `surface` column of road vector files
by assigning the surface value of the nearest road segment with known surface
information. Nearest-neighbour search is performed on road centroids.

Input
-----
A folder containing road vector files, e.g. Shapefile or GeoPackage files.
Each file should contain:
- geometry
- surface
- highway

Example
-------
python assign_nearest_surface.py \
    --input-dir data/roads/attribute_selected/Russia \
    --output-dir outputs/roads/surface_assigned/Russia \
    --file-extension .shp \
    --highway-column highway \
    --surface-column surface \
    --projected-crs EPSG:3857
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree


def get_centroid_coordinates(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Return centroid coordinates as an array with shape (n, 2)."""
    centroids = gdf.geometry.centroid
    return np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])


def assign_nearest_surface(
    roads: gpd.GeoDataFrame,
    highway_column: str = "highway",
    surface_column: str = "surface",
    categories: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Fill missing road surface information using nearest roads with known surface.

    Parameters
    ----------
    roads
        Road GeoDataFrame.
    highway_column
        Column containing road category/class.
    surface_column
        Column containing surface material information.
    categories
        Optional list of highway categories to process separately. If None,
        all missing surface values are processed together.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with missing surface values filled where possible.
    """
    if surface_column not in roads.columns:
        raise ValueError(f"Column '{surface_column}' not found.")

    if highway_column not in roads.columns:
        raise ValueError(f"Column '{highway_column}' not found.")

    roads = roads.copy()

    roads_with_surface = roads[roads[surface_column].notna()].copy()

    if roads_with_surface.empty:
        logging.warning("No roads with known surface information found.")
        return roads

    tree = BallTree(get_centroid_coordinates(roads_with_surface), metric="euclidean")

    if categories is None:
        target_index = roads.index[roads[surface_column].isna()]
        categories_to_process = [None]
    else:
        categories_to_process = [str(category) for category in categories]

    for category in categories_to_process:
        if category is None:
            missing_roads = roads.loc[target_index]
            logging.info("Processing all categories: %d missing records", len(missing_roads))
        else:
            category_mask = roads[highway_column].astype(str) == str(category)
            missing_roads = roads[category_mask & roads[surface_column].isna()]
            logging.info(
                "Processing category %s: %d missing records",
                category,
                len(missing_roads),
            )

        if missing_roads.empty:
            continue

        missing_coords = get_centroid_coordinates(missing_roads)
        nearest_indices = tree.query(missing_coords, return_distance=False).ravel()

        nearest_surfaces = roads_with_surface.iloc[nearest_indices][surface_column].to_numpy()
        roads.loc[missing_roads.index, surface_column] = nearest_surfaces

    return roads


def process_vector_file(
    input_path: Path,
    output_path: Path,
    highway_column: str,
    surface_column: str,
    categories: list[str] | None,
    projected_crs: str | None,
    overwrite: bool,
) -> None:
    """Process one road vector file."""
    if output_path.exists() and not overwrite:
        logging.info("Skipping existing file: %s", output_path)
        return

    roads = gpd.read_file(input_path)

    if roads.empty:
        logging.warning("Empty file: %s", input_path)
        return

    original_crs = roads.crs

    if projected_crs is not None:
        roads_projected = roads.to_crs(projected_crs)
    else:
        roads_projected = roads

        if roads_projected.crs is not None and roads_projected.crs.is_geographic:
            logging.warning(
                "%s uses a geographic CRS. Nearest-neighbour distances will be based on degrees. "
                "Use --projected-crs for more reliable results.",
                input_path.name,
            )

    assigned = assign_nearest_surface(
        roads=roads_projected,
        highway_column=highway_column,
        surface_column=surface_column,
        categories=categories,
    )

    if projected_crs is not None and original_crs is not None:
        assigned = assigned.to_crs(original_crs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_file(output_path)

    logging.info("Saved: %s", output_path)


def process_folder(
    input_dir: Path,
    output_dir: Path,
    file_extension: str,
    highway_column: str,
    surface_column: str,
    categories: list[str] | None,
    projected_crs: str | None,
    overwrite: bool,
) -> None:
    """Process all vector files in a folder."""
    files = sorted(input_dir.glob(f"*{file_extension}"))

    if not files:
        raise FileNotFoundError(f"No '{file_extension}' files found in {input_dir}")

    for input_path in files:
        output_path = output_dir / input_path.name

        try:
            process_vector_file(
                input_path=input_path,
                output_path=output_path,
                highway_column=highway_column,
                surface_column=surface_column,
                categories=categories,
                projected_crs=projected_crs,
                overwrite=overwrite,
            )
        except Exception as exc:
            logging.error("Failed to process %s: %s", input_path.name, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign missing road surface information using nearest-neighbour search."
    )

    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--file-extension", default=".shp")
    parser.add_argument("--highway-column", default="highway")
    parser.add_argument("--surface-column", default="surface")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="Road categories to process. Use --categories all to process all roads together.",
    )
    parser.add_argument(
        "--projected-crs",
        default=None,
        help="Optional projected CRS for centroid distance calculation, e.g. EPSG:3857.",
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    categories = None if args.categories == ["all"] else args.categories

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_extension=args.file_extension,
        highway_column=args.highway_column,
        surface_column=args.surface_column,
        categories=categories,
        projected_crs=args.projected_crs,
        overwrite=args.overwrite,
    )

    logging.info("Surface assignment completed.")


if __name__ == "__main__":
    main()

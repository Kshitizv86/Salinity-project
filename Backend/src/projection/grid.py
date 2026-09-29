"""Build a projected 1-km India grid and polygon-derived land mask."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def create_india_target_grid(
    boundary_path: Path,
    output_path: Path,
    resolution_m: int = 1000,
    projected_crs: str = "EPSG:6933",
) -> Path:
    """Rasterize an India boundary shapefile into a cell-center land mask."""
    try:
        import geopandas as gpd
        from pyproj import Transformer
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError(
            "Grid generation requires Backend/requirements-geospatial.txt"
        ) from exc

    if resolution_m <= 0:
        raise ValueError("resolution_m must be a positive integer")
    if not boundary_path.is_file():
        raise FileNotFoundError(f"India boundary shapefile not found: {boundary_path}")

    boundaries = gpd.read_file(boundary_path)
    if boundaries.empty or boundaries.crs is None:
        raise ValueError("India boundary shapefile must contain geometry and a defined CRS")
    geometry = unary_union(boundaries.to_crs(projected_crs).geometry)
    if geometry.is_empty:
        raise ValueError("India boundary shapefile has no usable polygon geometry")

    min_x, min_y, max_x, max_y = geometry.bounds
    origin_x = np.floor(min_x / resolution_m) * resolution_m
    origin_y = np.ceil(max_y / resolution_m) * resolution_m
    width = int(np.ceil((max_x - origin_x) / resolution_m))
    height = int(np.ceil((origin_y - min_y) / resolution_m))
    transform = from_origin(origin_x, origin_y, resolution_m, resolution_m)
    mask = rasterize(
        [(geometry, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    if not np.any(mask):
        raise ValueError("India boundary rasterized to an empty target grid")

    x = origin_x + (np.arange(width) + 0.5) * resolution_m
    y = origin_y - (np.arange(height) + 0.5) * resolution_m
    projected_x, projected_y = np.meshgrid(x, y)
    to_geographic = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    longitude, latitude = to_geographic.transform(projected_x, projected_y)
    target = xr.Dataset(
        {"india_mask": (("y", "x"), mask)},
        coords={
            "x": x,
            "y": y,
            "lat": (("y", "x"), latitude.astype(np.float32)),
            "lon": (("y", "x"), longitude.astype(np.float32)),
        },
        attrs={
            "target_resolution": f"{resolution_m / 1000:g}km",
            "target_resolution_m": resolution_m,
            "grid_crs": projected_crs,
            "mask_source": str(boundary_path.resolve()),
            "mask_method": "polygon rasterization; pixel center must be within India boundary",
            "geographic_domain": "India boundary polygon mask",
        },
    )
    target["india_mask"].attrs.update({
        "long_name": "Indian land-cell mask",
        "flag_values": np.array([0, 1], dtype=np.uint8),
        "flag_meanings": "outside_india india_land_cell",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_netcdf(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 1-km India boundary target grid")
    parser.add_argument("--boundary-shapefile", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "india_1km_grid.nc",
    )
    parser.add_argument("--resolution-m", type=int, default=1000)
    parser.add_argument("--projected-crs", default="EPSG:6933")
    args = parser.parse_args()
    print(create_india_target_grid(
        args.boundary_shapefile, args.output, args.resolution_m, args.projected_crs
    ))


if __name__ == "__main__":
    main()
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PRODUCT_DIR = Path(os.environ.get(
    "AQUASHIELD_PRODUCT_DIR",
    str(Path(__file__).resolve().parent / "data" / "processed"),
)).expanduser().resolve()
FUTURE_REQUIRED_LAYERS = {
    "salinity_mean",
    "climate_model_spread_sd",
    "model_uncertainty_sd",
    "data_uncertainty_sd",
    "applicability_mask",
    "covariate_shift_score",
    "applicability_probability",
    "applicability_uncertainty",
}
UNCERTAINTY_LAYERS = {
    "climate_model_spread_sd",
    "model_uncertainty_sd",
    "data_uncertainty_sd",
    "covariate_shift_score",
    "applicability_probability",
    "applicability_uncertainty",
}


class MapResponse(BaseModel):
    scenario: str
    period: str
    layers: dict[str, Any]
    latitude: Any
    longitude: Any
    metadata: dict[str, Any]


class LocationResponse(BaseModel):
    scenario: str
    period: str
    latitude: float
    longitude: float
    values: dict[str, float | None]
    metadata: dict[str, Any]


class TimeSeriesResponse(BaseModel):
    scenario: str
    latitude: float
    longitude: float
    points: list[dict[str, Any]]


class DriversResponse(BaseModel):
    scenario: str
    period: str
    latitude: float
    longitude: float
    drivers: dict[str, float]
    metadata: dict[str, Any]


class UncertaintyResponse(BaseModel):
    scenario: str
    period: str
    latitude: float
    longitude: float
    layers: dict[str, float]
    metadata: dict[str, Any]


app = FastAPI(title="AquaShield Salinity API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _product_path(scenario: str, period: str) -> Path:
    if scenario == "current" and period == "current":
        return PRODUCT_DIR / "current.nc"
    if scenario not in {"ssp245", "ssp585"} or period not in {"2030", "2050", "2080"}:
        raise HTTPException(status_code=422, detail="Use current/current or ssp245|ssp585 with 2030|2050|2080")
    return PRODUCT_DIR / f"future_{scenario}_{period}.nc"


@contextmanager
def _open_product(scenario: str, period: str):
    path = _product_path(scenario, period)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Standardized product is not available: {path.name}")
    try:
        dataset = xr.open_dataset(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read product {path.name}: {exc}") from exc
    try:
        if scenario != "current":
            missing = FUTURE_REQUIRED_LAYERS - set(dataset.data_vars)
            if missing:
                raise HTTPException(
                    status_code=503,
                    detail=f"Future product is not validated for serving; missing required layers: {sorted(missing)}",
                )
        yield dataset
    finally:
        dataset.close()


def _coordinates(dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    if "lat" not in dataset or "lon" not in dataset:
        raise HTTPException(status_code=503, detail="Product must contain standardized lat/lon coordinates")
    return dataset["lat"], dataset["lon"]


def _metadata(dataset: xr.Dataset) -> dict[str, Any]:
    return {str(key): str(value) for key, value in dataset.attrs.items()}


def _json_array(data: xr.DataArray) -> Any:
    values = np.asarray(data.values)
    if np.issubdtype(values.dtype, np.number):
        values = values.astype(object)
        values[~np.isfinite(np.asarray(data.values, dtype=float))] = None
    return values.tolist()


def _subset_map(dataset: xr.Dataset, lat_min: float, lat_max: float,
                lon_min: float, lon_max: float) -> xr.Dataset:
    latitude, longitude = _coordinates(dataset)
    lon_min = (lon_min + 180) % 360 - 180
    lon_max = (lon_max + 180) % 360 - 180
    inside_lon = ((longitude >= lon_min) & (longitude <= lon_max)) if lon_min <= lon_max else (
        (longitude >= lon_min) | (longitude <= lon_max)
    )
    subset = dataset.where(
        (latitude >= lat_min) & (latitude <= lat_max) & inside_lon,
        drop=True,
    )
    spatial_dims = set(latitude.dims) | set(longitude.dims)
    cell_count = int(np.prod([subset.sizes[dim] for dim in spatial_dims])) if spatial_dims else 0
    if cell_count == 0:
        raise HTTPException(status_code=404, detail="No grid cells intersect the requested bounds")
    if cell_count > 100_000:
        raise HTTPException(status_code=413, detail="Map subset exceeds 100,000 cells; request a smaller area")
    return subset


def _point(dataset: xr.Dataset, latitude: float, longitude: float) -> tuple[xr.Dataset, float, float]:
    lat, lon = _coordinates(dataset)
    if lat.ndim == 1 and lon.ndim == 1:
        lon_values = np.asarray(lon.values, dtype=float)
        query_lon = longitude + 360 if np.nanmax(lon_values) > 180 and longitude < 0 else longitude
        point = dataset.sel(lat=latitude, lon=query_lon, method="nearest")
        return point, float(point.lat.values), float(point.lon.values)

    lat_values = np.asarray(lat.values, dtype=float)
    lon_values = np.asarray(lon.values, dtype=float)
    lon_delta = (lon_values - longitude + 180) % 360 - 180
    distance = (lat_values - latitude) ** 2 + (lon_delta * np.cos(np.deg2rad(latitude))) ** 2
    if not np.isfinite(distance).any():
        raise HTTPException(status_code=404, detail="Product has no valid grid coordinates")
    indices = np.unravel_index(np.nanargmin(distance), distance.shape)
    indexers = {dim: index for dim, index in zip(lat.dims, indices)}
    return dataset.isel(indexers), float(lat.values[indices]), float(lon.values[indices])


def _require_layers(dataset: xr.Dataset, names: set[str]) -> None:
    missing = names - set(dataset.data_vars)
    if missing:
        raise HTTPException(status_code=503, detail=f"Product is missing required layers: {sorted(missing)}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/products")
def get_products() -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    for scenario, period in [("current", "current"), ("ssp245", "2030"), ("ssp245", "2050"), ("ssp245", "2080"), ("ssp585", "2030"), ("ssp585", "2050"), ("ssp585", "2080")]:
        key = "current" if (scenario == "current" and period == "current") else f"{scenario}_{period}"
        path = _product_path(scenario, period)
        products[key] = {
            "scenario": scenario,
            "period": period,
            "available": path.is_file(),
            "path": str(path),
        }
    return {"products": products}


@app.get("/api/v1/maps/{scenario}/{period}", response_model=MapResponse)
def get_map(
    scenario: str,
    period: str,
    layer: str = "salinity_mean",
    lat_min: float = 5.0,
    lat_max: float = 38.0,
    lon_min: float = 65.0,
    lon_max: float = 100.0,
):
    with _open_product(scenario, period) as dataset:
        subset = _subset_map(dataset, lat_min, lat_max, lon_min, lon_max)
        if layer not in subset.data_vars:
            raise HTTPException(status_code=404, detail=f"Layer {layer!r} is not available")
        latitude, longitude = _coordinates(subset)
        return MapResponse(
            scenario=scenario,
            period=period,
            layers={layer: _json_array(subset[layer])},
            latitude=_json_array(latitude),
            longitude=_json_array(longitude),
            metadata=_metadata(dataset),
        )


@app.get("/api/v1/location", response_model=LocationResponse)
def get_location(lat: float, lon: float, scenario: str = "current", period: str = "current"):
    with _open_product(scenario, period) as dataset:
        point, actual_lat, actual_lon = _point(dataset, lat, lon)
        names = [name for name in ("salinity_mean", "applicability_mask", *UNCERTAINTY_LAYERS)
                 if name in point.data_vars]
        if scenario != "current":
            _require_layers(dataset, FUTURE_REQUIRED_LAYERS)
        if "salinity_mean" not in names:
            raise HTTPException(status_code=503, detail="Product has no salinity_mean layer")
        values = {}
        for name in names:
            value = float(point[name].values)
            values[name] = value if np.isfinite(value) else None
        return LocationResponse(scenario=scenario, period=period, latitude=actual_lat,
                                longitude=actual_lon, values=values, metadata=_metadata(dataset))


@app.get("/api/v1/drivers", response_model=DriversResponse)
def get_drivers(lat: float, lon: float, scenario: str = "current", period: str = "current"):
    with _open_product(scenario, period) as dataset:
        point, actual_lat, actual_lon = _point(dataset, lat, lon)
        drivers = {name: float(point[name].values) for name in point.data_vars if name.startswith("driver_")}
        if not drivers:
            raise HTTPException(status_code=404, detail="No driver_* layers are available in this product")
        return {"scenario": scenario, "period": period, "latitude": actual_lat,
                "longitude": actual_lon, "drivers": drivers, "metadata": _metadata(dataset)}


@app.get("/api/v1/uncertainty", response_model=UncertaintyResponse)
def get_uncertainty(lat: float, lon: float, scenario: str, period: str):
    with _open_product(scenario, period) as dataset:
        required = UNCERTAINTY_LAYERS | {"applicability_mask"}
        _require_layers(dataset, required)
        point, actual_lat, actual_lon = _point(dataset, lat, lon)
        layers = {name: float(point[name].values) for name in sorted(required)}
        return {"scenario": scenario, "period": period, "latitude": actual_lat,
                "longitude": actual_lon, "layers": layers, "metadata": _metadata(dataset)}


@app.get("/api/v1/timeseries", response_model=TimeSeriesResponse)
def get_timeseries(lat: float, lon: float, scenario: str = "ssp245"):
    if scenario not in {"ssp245", "ssp585"}:
        raise HTTPException(status_code=422, detail="Scenario must be ssp245 or ssp585")
    points = []
    actual_lat = lat
    actual_lon = lon
    for period in ("2030", "2050", "2080"):
        with _open_product(scenario, period) as dataset:
            _require_layers(dataset, FUTURE_REQUIRED_LAYERS)
            point, actual_lat, actual_lon = _point(dataset, lat, lon)
            points.append({
                "period": period,
                "salinity_mean": float(point.salinity_mean.values),
                "climate_model_spread_sd": float(point.climate_model_spread_sd.values),
                "model_uncertainty_sd": float(point.model_uncertainty_sd.values),
                "data_uncertainty_sd": float(point.data_uncertainty_sd.values),
                "applicability_mask": int(point.applicability_mask.values),
                "covariate_shift_score": float(point.covariate_shift_score.values),
                "applicability_probability": float(point.applicability_probability.values),
                "applicability_uncertainty": float(point.applicability_uncertainty.values),
            })
    return TimeSeriesResponse(scenario=scenario, latitude=actual_lat, longitude=actual_lon, points=points)
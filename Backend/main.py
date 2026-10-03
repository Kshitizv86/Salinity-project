import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


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
    status: str
    layers: dict[str, Any]
    latitude: Any
    longitude: Any
    metadata: dict[str, Any]


class LocationResponse(BaseModel):
    scenario: str
    period: str
    status: str
    latitude: float
    longitude: float
    values: dict[str, float | None]
    metadata: dict[str, Any]


class TimeSeriesResponse(BaseModel):
    scenario: str
    status: str
    latitude: float
    longitude: float
    points: list[dict[str, Any]]


class DriversResponse(BaseModel):
    scenario: str
    period: str
    status: str
    latitude: float
    longitude: float
    drivers: dict[str, float]
    metadata: dict[str, Any]


class UncertaintyResponse(BaseModel):
    scenario: str
    period: str
    status: str
    latitude: float
    longitude: float
    layers: dict[str, float]
    metadata: dict[str, Any]


app = FastAPI(title="AquaShield Salinity API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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


class UncertaintyValue(BaseModel):
    value: float
    unit: str
    method: str


class LocationPredictionResponse(BaseModel):
    latitude: float
    longitude: float
    levelM: float | None
    salinityEcUsCm: float | None
    salinityProbability: float | None
    salinityRisk: str | None
    uncertainty: UncertaintyValue | None
    applicabilityScore: float | None
    dataYear: int | None
    modelVersion: str | None
    status: str
    period: str = "current"
    scenario: str = "current"


class ExplanationFeature(BaseModel):
    name: str
    contribution: float
    direction: str


class PredictionExplanationResponse(BaseModel):
    latitude: float
    longitude: float
    features: list[ExplanationFeature]
    modelVersion: str | None
    status: str


class MapLegendItem(BaseModel):
    label: str
    color: str | None = None
    minimum: float | None = None
    maximum: float | None = None


class MapLayer(BaseModel):
    id: str
    label: str
    available: bool
    tileUrl: str | None
    dataUrl: str | None
    unit: str | None
    legend: list[MapLegendItem]
    attribution: str | None
    status: str


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)
    locationLabel: str | None = Field(default=None, max_length=256)
    selectedLayer: str | None = Field(default=None, max_length=128)
    period: str = "current"
    scenario: str = "current"


class AssistantAction(BaseModel):
    type: str
    value: str


class AssistantResponse(BaseModel):
    answer: str
    actions: list[AssistantAction]
    status: str


def _configured_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _data_path() -> Path:
    return _configured_path(
        "DATA_PATH",
        Path(__file__).resolve().parent / "data" / "processed" / "master_observations.parquet",
    )


def _model_path() -> Path:
    return _configured_path("MODEL_PATH", Path(__file__).resolve().parent / "models" / "production")


def _cmip6_data_path() -> Path:
    return _configured_path("CMIP6_DATA_PATH", Path(__file__).resolve().parent / "data" / "cmip6")


def _period_scenario(period: str, scenario: str) -> tuple[str, str]:
    if period == "current" and scenario == "current":
        return "current", "current"
    if period in {"2030", "2050", "2080"} and scenario in {"ssp245", "ssp585"}:
        return scenario, period
    raise HTTPException(
        status_code=422,
        detail="Use current/current or period 2030|2050|2080 with scenario ssp245|ssp585",
    )


def _is_demo_product(dataset: xr.Dataset) -> bool:
    return str(dataset.attrs.get("demo_status", "")).lower() in {
        "synthetic_demo",
        "demo",
    } or "synthetic" in str(dataset.attrs.get("warning", "")).lower()


def _salinity_unit(dataset: xr.Dataset) -> str | None:
    value = dataset.attrs.get("salinity_unit")
    if value is None and "salinity_mean" in dataset:
        value = dataset["salinity_mean"].attrs.get("units")
    if value is None:
        return None
    unit = str(value).strip()
    normalized = unit.lower().replace("μ", "µ").replace(" ", "")
    if normalized not in {"µs/cm", "us/cm", "µscm-1", "uscm-1", "microsiemens/cm"}:
        return None
    return unit


def _point_number(point: xr.Dataset, name: str) -> float | None:
    if name not in point:
        return None
    try:
        value = float(point[name].values)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Product variable {name!r} is not numeric") from exc
    return value if np.isfinite(value) else None


def _product_status(dataset: xr.Dataset) -> str:
    if _is_demo_product(dataset):
        return "demo"
    if "salinity_mean" not in dataset:
        return "data_pending"
    model_version = str(dataset.attrs.get("model_version", "")).strip()
    data_year = dataset.attrs.get("data_year")
    if (
        not model_version
        or data_year is None
        or _salinity_unit(dataset) is None
    ):
        return "data_pending"
    try:
        if not 1 <= int(data_year) <= 9999:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Product has invalid data_year metadata") from exc
    return "model_output"


def _prediction_for_location(
    latitude: float,
    longitude: float,
    period: str,
    scenario: str,
) -> LocationPredictionResponse:
    product_scenario, product_period = _period_scenario(period, scenario)
    base = {
        "latitude": latitude,
        "longitude": longitude,
        "levelM": None,
        "salinityEcUsCm": None,
        "salinityProbability": None,
        "salinityRisk": None,
        "uncertainty": None,
        "applicabilityScore": None,
        "dataYear": None,
        "modelVersion": None,
        "period": period,
        "scenario": scenario,
    }
    path = _product_path(product_scenario, product_period)
    if not path.is_file():
        return LocationPredictionResponse(**base, status="data_pending")
    try:
        with xr.open_dataset(path) as dataset:
            status = _product_status(dataset)
            if status == "demo":
                return LocationPredictionResponse(**base, status="demo")
            if status != "model_output":
                return LocationPredictionResponse(**base, status="data_pending")
            point, actual_lat, actual_lon = _point(dataset, latitude, longitude)
            salinity = float(point["salinity_mean"].values)
            if not np.isfinite(salinity):
                raise HTTPException(status_code=404, detail="No salinity value is available at this location")
            level_name = next(
                (
                    name
                    for name in ("level_m", "groundwater_level_m")
                    if name in point
                    and str(point[name].attrs.get("units", "")).strip().lower() in {"m", "meter", "meters"}
                ),
                None,
            )
            probability = _point_number(point, "salinity_probability")
            uncertainty = None
            for variable in ("model_uncertainty_sd", "data_uncertainty_sd", "climate_model_spread_sd"):
                if variable in point:
                    value = float(point[variable].values)
                    method = str(dataset.attrs.get("uncertainty_method", ""))
                    units = dataset[variable].attrs.get("units") or _salinity_unit(dataset)
                    if np.isfinite(value) and method and units:
                        uncertainty = UncertaintyValue(value=value, unit=units, method=method)
                        break
            applicability = None
            if "applicability_probability" in point:
                score = float(point["applicability_probability"].values)
                applicability = score if np.isfinite(score) else None
            return LocationPredictionResponse(
                **{
                    **base,
                    "latitude": actual_lat,
                    "longitude": actual_lon,
                    "levelM": _point_number(point, level_name) if level_name else None,
                    "salinityEcUsCm": salinity,
                    "salinityProbability": probability,
                    "salinityRisk": str(point["salinity_risk"].values)
                    if "salinity_risk" in point
                    else None,
                    "uncertainty": uncertainty,
                    "applicabilityScore": applicability,
                    "dataYear": int(dataset.attrs["data_year"]),
                    "modelVersion": str(dataset.attrs["model_version"]),
                },
                status="model_output",
            )
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read prediction product: {exc}") from exc


def _layer_metadata(
    period: str,
    scenario: str,
    variable: str,
) -> tuple[str, str | None, list[MapLegendItem], str | None]:
    path = _product_path(scenario, period)
    if not path.is_file():
        return "data_pending", None, [], None
    try:
        with xr.open_dataset(path) as dataset:
            if variable not in dataset:
                return "data_pending", None, [], None
            if _is_demo_product(dataset):
                status = "demo"
            else:
                status = _product_status(dataset)
            raw_legend = dataset[variable].attrs.get("legend", "[]")
            try:
                legend_data = json.loads(raw_legend) if isinstance(raw_legend, str) else raw_legend
                legend = [MapLegendItem.model_validate(item) for item in legend_data]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Map product {path.name} has invalid legend metadata for {variable!r}",
                ) from exc
            return (
                status,
                str(dataset[variable].attrs.get("units", "")) or None,
                legend,
                str(dataset.attrs.get("attribution", "")) or None,
            )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not inspect map product {path.name}: {exc}") from exc


@app.get(
    "/maps/layers",
    response_model=list[MapLayer],
    summary="List map layers and their availability",
    description=(
        "Returns available production layers with source URLs and units. "
        "Only versioned products with declared conductivity units are marked model_output; "
        "synthetic products are identified as demo."
    ),
)
def get_map_layers() -> list[MapLayer]:
    definitions = [
        ("groundwater-level", "Groundwater Level", "level_m", "current", "current"),
        ("current-salinity", "Current Salinity", "salinity_mean", "current", "current"),
        ("salinity-risk", "Salinity Risk", "salinity_probability", "current", "current"),
        ("salinity-uncertainty", "Salinity Uncertainty", "model_uncertainty_sd", "current", "current"),
        ("applicability", "Model Applicability", "applicability_probability", "current", "current"),
        ("demo-salinity-index", "Synthetic Demo Salinity Index", "salinity_mean", "current", "current"),
    ]
    for future_scenario in ("ssp245", "ssp585"):
        for future_period in ("2030", "2050", "2080"):
            definitions.append(
                (
                    f"salinity-{future_scenario}-{future_period}",
                    f"Salinity projection {future_period} ({future_scenario.upper()})",
                    "salinity_mean",
                    future_scenario,
                    future_period,
                )
            )
    layers = []
    for layer_id, label, variable, scenario, period in definitions:
        status, unit, legend, attribution = _layer_metadata(period, scenario, variable)
        is_demo_layer = layer_id == "demo-salinity-index"
        if is_demo_layer:
            available = status == "demo"
            status = "demo" if available else "unavailable"
        else:
            available = status == "model_output"
            if status == "demo":
                status = "unavailable"
        layers.append(
            MapLayer(
                id=layer_id,
                label=label,
                available=available,
                tileUrl=None,
                dataUrl=f"/api/v1/maps/{scenario}/{period}?layer={variable}" if available else None,
                unit=unit,
                legend=legend,
                attribution=attribution,
                status=status,
            )
        )
    return layers


@app.get(
    "/predictions/location",
    response_model=LocationPredictionResponse,
    summary="Get groundwater and salinity values for a coordinate",
)
def get_prediction_location(
    latitude: float = Query(..., ge=-90, le=90, allow_inf_nan=False),
    longitude: float = Query(..., ge=-180, le=180, allow_inf_nan=False),
    period: str = Query("current", pattern="^(current|2030|2050|2080)$"),
    scenario: str = Query("current", pattern="^(current|ssp245|ssp585)$"),
) -> LocationPredictionResponse:
    return _prediction_for_location(latitude, longitude, period, scenario)


@app.get(
    "/projections/location",
    response_model=LocationPredictionResponse,
    summary="Get a future salinity projection for a coordinate",
)
def get_projection_location(
    latitude: float = Query(..., ge=-90, le=90, allow_inf_nan=False),
    longitude: float = Query(..., ge=-180, le=180, allow_inf_nan=False),
    period: str = Query(..., pattern="^(current|2030|2050|2080)$"),
    scenario: str = Query(..., pattern="^(current|ssp245|ssp585)$"),
) -> LocationPredictionResponse:
    if period == "current":
        raise HTTPException(status_code=422, detail="Projection period must be 2030, 2050, or 2080")
    return _prediction_for_location(latitude, longitude, period, scenario)


@app.get(
    "/predictions/explanation",
    response_model=PredictionExplanationResponse,
    summary="Get model explanation for a coordinate",
)
def get_prediction_explanation(
    latitude: float = Query(..., ge=-90, le=90, allow_inf_nan=False),
    longitude: float = Query(..., ge=-180, le=180, allow_inf_nan=False),
) -> PredictionExplanationResponse:
    prediction = _prediction_for_location(latitude, longitude, "current", "current")
    return PredictionExplanationResponse(
        latitude=latitude,
        longitude=longitude,
        features=[],
        modelVersion=prediction.modelVersion,
        status=prediction.status if prediction.status != "model_output" else "explanation_unavailable",
    )


@app.post(
    "/assistant/query",
    response_model=AssistantResponse,
    summary="Answer a question using validated backend data",
)
def assistant_query(request: AssistantRequest) -> AssistantResponse:
    if (request.latitude is None) != (request.longitude is None):
        raise HTTPException(status_code=422, detail="latitude and longitude must be provided together")
    if request.period not in {"current", "2030", "2050", "2080"}:
        raise HTTPException(status_code=422, detail="period must be current, 2030, 2050, or 2080")
    if request.scenario not in {"current", "ssp245", "ssp585"}:
        raise HTTPException(status_code=422, detail="scenario must be current, ssp245, or ssp585")
    _period_scenario(request.period, request.scenario)
    if request.selectedLayer is not None and request.selectedLayer not in {
        "groundwater-level",
        "current-salinity",
        "salinity-risk",
        "salinity-uncertainty",
        "applicability",
        "demo-salinity-index",
    }:
        raise HTTPException(status_code=422, detail="selectedLayer is not a recognized map layer")
    if request.latitude is not None:
        prediction = _prediction_for_location(
            request.latitude,
            request.longitude,
            request.period,
            request.scenario,
        )
        if prediction.status == "model_output" and prediction.salinityEcUsCm is not None:
            return AssistantResponse(
                answer=(
                    f"Validated salinity prediction: {prediction.salinityEcUsCm:g} "
                    f"µS/cm (model {prediction.modelVersion}, data year {prediction.dataYear})."
                ),
                actions=[],
                status="model_output",
            )
        if prediction.status == "demo":
            return AssistantResponse(
                answer="I don't have a validated prediction for this location yet.",
                actions=[],
                status="demo",
            )
    return AssistantResponse(
        answer="I don't have a validated prediction for this location yet.",
        actions=[],
        status="data_pending",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8000")),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "1.0.0",
        "dataConfigured": _data_path().is_file(),
        "modelConfigured": _model_path().exists(),
        "cmip6DataConfigured": _cmip6_data_path().exists(),
    }


@app.get("/api/v1/products")
def get_products() -> dict[str, Any]:
    products: dict[str, Any] = {
        "current": _product_path("current", "current").is_file(),
    }
    for scenario in ("ssp245", "ssp585"):
        products[scenario] = {
            period: _product_path(scenario, period).is_file()
            for period in ("2030", "2050", "2080")
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
            status=_product_status(dataset),
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
                                longitude=actual_lon, status=_product_status(dataset),
                                values=values, metadata=_metadata(dataset))


@app.get("/api/v1/drivers", response_model=DriversResponse)
def get_drivers(lat: float, lon: float, scenario: str = "current", period: str = "current"):
    with _open_product(scenario, period) as dataset:
        point, actual_lat, actual_lon = _point(dataset, lat, lon)
        drivers = {name: float(point[name].values) for name in point.data_vars if name.startswith("driver_")}
        if not drivers:
            raise HTTPException(status_code=404, detail="No driver_* layers are available in this product")
        return {"scenario": scenario, "period": period, "latitude": actual_lat,
                "longitude": actual_lon, "status": _product_status(dataset),
                "drivers": drivers, "metadata": _metadata(dataset)}


@app.get("/api/v1/uncertainty", response_model=UncertaintyResponse)
def get_uncertainty(lat: float, lon: float, scenario: str, period: str):
    with _open_product(scenario, period) as dataset:
        required = UNCERTAINTY_LAYERS | {"applicability_mask"}
        _require_layers(dataset, required)
        point, actual_lat, actual_lon = _point(dataset, lat, lon)
        layers = {name: float(point[name].values) for name in sorted(required)}
        return {"scenario": scenario, "period": period, "latitude": actual_lat,
                "longitude": actual_lon, "status": _product_status(dataset),
                "layers": layers, "metadata": _metadata(dataset)}


@app.get("/api/v1/timeseries", response_model=TimeSeriesResponse)
def get_timeseries(lat: float, lon: float, scenario: str = "ssp245"):
    if scenario not in {"ssp245", "ssp585"}:
        raise HTTPException(status_code=422, detail="Scenario must be ssp245 or ssp585")
    points = []
    product_statuses = []
    actual_lat = lat
    actual_lon = lon
    for period in ("2030", "2050", "2080"):
        with _open_product(scenario, period) as dataset:
            _require_layers(dataset, FUTURE_REQUIRED_LAYERS)
            product_statuses.append(_product_status(dataset))
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
    status = "demo" if "demo" in product_statuses else (
        "model_output" if all(value == "model_output" for value in product_statuses) else "data_pending"
    )
    return TimeSeriesResponse(
        scenario=scenario,
        status=status,
        latitude=actual_lat,
        longitude=actual_lon,
        points=points,
    )
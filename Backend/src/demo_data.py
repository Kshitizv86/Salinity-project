"""Generate isolated synthetic demo artifacts for the AquaShield API.

Nothing written by this module is observed data, a validated model, or a real
groundwater-salinity prediction. It is a reproducible integration demo only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

if __package__:
    from .projection.applicability import assess_applicability
    from .projection.bias_correction import monthly_delta_correction
    from .projection.feature_assembly import assemble_model_features
    from .projection.regrid import regrid_to_target_grid
    from .uncertainty.quantify import quantify_prediction_uncertainty
else:
    from projection.applicability import assess_applicability
    from projection.bias_correction import monthly_delta_correction
    from projection.feature_assembly import assemble_model_features
    from projection.regrid import regrid_to_target_grid
    from uncertainty.quantify import quantify_prediction_uncertainty


FEATURE_MAP = {
    "precipitation_mm_day": "pr",
    "mean_temperature_c": "tas",
    "maximum_temperature_c": "tasmax",
}
CLIMATE_FEATURE_NAMES = list(FEATURE_MAP)
STATIC_FEATURE_NAMES = ["elevation_m"]
FEATURE_NAMES = CLIMATE_FEATURE_NAMES + STATIC_FEATURE_NAMES
SCENARIO_WARMING = {
    "ssp245": {"2030": 0.6, "2050": 1.1, "2080": 1.8},
    "ssp585": {"2030": 0.9, "2050": 1.8, "2080": 3.0},
}
MODEL_NAMES = ("demo_gcm_01", "demo_gcm_02", "demo_gcm_03")
DEMO_LABEL = "SYNTHETIC DEMO ONLY - NOT REAL DATA OR REAL PREDICTIONS"
RANDOM_SEED = 20260929


def _demo_attrs(**extra: object) -> dict[str, object]:
    return {
        "demo_status": "synthetic_demo",
        "warning": DEMO_LABEL,
        "source": "deterministic synthetic generator; no external observations or CMIP6 data",
        **extra,
    }


def _base_features(latitude: np.ndarray, longitude: np.ndarray) -> dict[str, np.ndarray]:
    lat = np.asarray(latitude, dtype=float)[:, None]
    lon = np.asarray(longitude, dtype=float)[None, :]
    precipitation = 3.6 + 2.3 * np.exp(-((lat - 22.0) / 9.5) ** 2) + 0.5 * np.cos((lon - 82.0) / 7.0)
    temperature = 29.0 - 0.48 * (lat - 8.0) + 0.035 * (lon - 82.0)
    return {
        "precipitation_mm_day": np.broadcast_to(precipitation, (lat.shape[0], lon.shape[1])).copy(),
        "mean_temperature_c": np.broadcast_to(temperature, (lat.shape[0], lon.shape[1])).copy(),
        "maximum_temperature_c": np.broadcast_to(temperature + 8.0, (lat.shape[0], lon.shape[1])).copy(),
    }


def _point_features(latitude: np.ndarray, longitude: np.ndarray) -> dict[str, np.ndarray]:
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    precipitation = 3.6 + 2.3 * np.exp(-((lat - 22.0) / 9.5) ** 2) + 0.5 * np.cos((lon - 82.0) / 7.0)
    temperature = 29.0 - 0.48 * (lat - 8.0) + 0.035 * (lon - 82.0)
    return {
        FEATURE_NAMES[0]: precipitation,
        FEATURE_NAMES[1]: temperature,
        FEATURE_NAMES[2]: temperature + 8.0,
    }


def _elevation(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    return (
        150.0
        + 1200.0 * np.exp(-((np.asarray(latitude, dtype=float) - 30.0) / 8.0) ** 2)
        + 250.0 * np.cos((np.asarray(longitude, dtype=float) - 82.0) / 7.0)
    )


def _static_features(latitude: np.ndarray, longitude: np.ndarray) -> xr.Dataset:
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    return xr.Dataset(
        {"elevation_m": (("lat", "lon"), _elevation(lat_grid, lon_grid).astype(np.float32), {"units": "m"})},
        coords={"lat": latitude, "lon": longitude},
        attrs=_demo_attrs(dataset_role="synthetic static elevation feature"),
    )


def _make_monthly_dataset(
    base: dict[str, np.ndarray],
    latitude: np.ndarray,
    longitude: np.ndarray,
    times: pd.DatetimeIndex,
    precipitation_factor: float = 1.0,
    temperature_offset: float = 0.0,
) -> xr.Dataset:
    seasonal = np.sin(2.0 * np.pi * (times.month.to_numpy() - 1) / 12.0 - 0.8)
    precipitation_cycle = 1.0 + 0.35 * seasonal
    temperature_cycle = 2.5 * seasonal
    data_vars = {
        FEATURE_NAMES[0]: (("time", "lat", "lon"), (
            base[FEATURE_NAMES[0]][None, :, :] * precipitation_cycle[:, None, None] * precipitation_factor
        ).astype(np.float32)),
        FEATURE_NAMES[1]: (("time", "lat", "lon"), (
            base[FEATURE_NAMES[1]][None, :, :] + temperature_cycle[:, None, None] + temperature_offset
        ).astype(np.float32)),
        FEATURE_NAMES[2]: (("time", "lat", "lon"), (
            base[FEATURE_NAMES[2]][None, :, :] + temperature_cycle[:, None, None] + temperature_offset
        ).astype(np.float32)),
    }
    dataset = xr.Dataset(data_vars, coords={"time": times, "lat": latitude, "lon": longitude})
    dataset[FEATURE_NAMES[0]].attrs["units"] = "mm/day"
    dataset[FEATURE_NAMES[1]].attrs["units"] = "degC"
    dataset[FEATURE_NAMES[2]].attrs["units"] = "degC"
    return dataset


def _fit_demo_model(training: xr.Dataset) -> dict[str, object]:
    feature_matrix = np.column_stack([training[name].values for name in FEATURE_NAMES])
    means = feature_matrix.mean(axis=0)
    scales = feature_matrix.std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (feature_matrix - means) / scales
    design = np.column_stack([np.ones(len(feature_matrix)), standardized])
    coefficients, _, _, _ = np.linalg.lstsq(design, training["synthetic_salinity_label"].values, rcond=None)
    return {
        "model_type": "standardized linear regression fitted by numpy.linalg.lstsq",
        "feature_names_in_order": FEATURE_NAMES,
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "training_sample_count": int(len(feature_matrix)),
        "target_name": "synthetic_salinity_label",
        "target_units": "unitless synthetic demo index",
        "demo_status": "synthetic_demo",
        "warning": DEMO_LABEL,
    }


def _predict_ensemble(
    features: xr.Dataset,
    model: dict[str, object],
    rng: np.random.Generator,
) -> xr.Dataset:
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    intercept = float(model["intercept"])
    model_draw_count = 3
    observation_draw_count = 3
    prediction_arrays = []
    for climate_model in features.climate_model.values:
        values = np.stack([features[name].sel(climate_model=climate_model).values for name in FEATURE_NAMES], axis=-1)
        standardized = (values - means) / scales
        model_draws = []
        for _ in range(model_draw_count):
            draw_coefficients = coefficients + rng.normal(0.0, 0.035, size=coefficients.shape)
            base_prediction = intercept + np.einsum("...i,i->...", standardized, draw_coefficients)
            observation_draws = [
                np.maximum(0.05, base_prediction + rng.normal(0.0, 0.12, size=base_prediction.shape))
                for _ in range(observation_draw_count)
            ]
            model_draws.append(np.stack(observation_draws))
        prediction_arrays.append(np.stack(model_draws))
    prediction = np.stack(prediction_arrays).astype(np.float32)
    return xr.Dataset(
        {"salinity_prediction": (("climate_model", "model_draw", "observation_draw", "lat", "lon"), prediction)},
        coords={
            "climate_model": features.climate_model.values,
            "model_draw": np.arange(model_draw_count),
            "observation_draw": np.arange(observation_draw_count),
            "lat": features.lat.values,
            "lon": features.lon.values,
        },
        attrs=_demo_attrs(model_artifact="synthetic_demo_model.json"),
    )


def _collapse_applicability_layers(product: xr.Dataset) -> xr.Dataset:
    reducers = {
        "applicability_mask": "min",
        "covariate_shift_score": "max",
        "applicability_probability": "mean",
    }
    for name, reduction in reducers.items():
        extra_dims = [dim for dim in product[name].dims if dim not in {"lat", "lon"}]
        if extra_dims:
            product[name] = getattr(product[name], reduction)(extra_dims)
    probability = product["applicability_probability"]
    product["applicability_uncertainty"] = np.sqrt(probability * (1.0 - probability))
    for name in product.data_vars:
        remaining_dims = set(product[name].dims) - {"lat", "lon"}
        if remaining_dims:
            raise ValueError(f"Demo API layer {name!r} is not a spatial map: {product[name].dims}")
    product.attrs["demo_applicability_aggregation"] = (
        "minimum mask, maximum shift, mean probability across synthetic climate models"
    )
    return product


def _create_feature_product(
    scenario: str,
    period: str,
    warming: float,
    base: dict[str, np.ndarray],
    latitude: np.ndarray,
    longitude: np.ndarray,
    observed: xr.Dataset,
    historical_by_model: dict[str, xr.Dataset],
    target_grid: xr.Dataset,
    output_dir: Path,
) -> xr.Dataset:
    representative_year = "2020" if scenario == "current" else "2030"
    future_times = pd.date_range(f"{representative_year}-01-01", periods=12, freq="MS")
    climate_members = []
    for index, model_name in enumerate(MODEL_NAMES):
        model_temp_offset = (-0.35, 0.0, 0.4)[index]
        model_precip_factor = (0.94, 1.0, 1.06)[index]
        scenario_precip_factor = 1.0 - warming * 0.018
        future_monthly = _make_monthly_dataset(
            base,
            latitude,
            longitude,
            future_times,
            precipitation_factor=model_precip_factor * scenario_precip_factor,
            temperature_offset=warming + model_temp_offset,
        )
        corrected = monthly_delta_correction(
            future_monthly,
            historical_by_model[model_name],
            observed,
            FEATURE_MAP,
        )
        member = corrected.mean("time", keep_attrs=True).expand_dims(climate_model=[model_name])
        climate_members.append(member)
    features = xr.concat(climate_members, dim="climate_model", join="exact")
    features.attrs.update(_demo_attrs(
        scenario=scenario,
        period=period,
        representative_monthly_climatology="12 synthetic months; not a CMIP6 time series",
        feature_contract="feature_contract_demo.json",
        original_climate_resolution="2degree synthetic demo grid",
        bias_correction="synthetic monthly delta correction against fake observed reference",
    ))
    features = regrid_to_target_grid(features, target_grid)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_netcdf(output_dir / f"features_{scenario}_{period}_demo.nc")
    return features


def generate_demo(output_dir: Path) -> dict[str, Path]:
    rng = np.random.default_rng(RANDOM_SEED)
    output_dir = output_dir.resolve()
    products_dir = output_dir / "products"
    feature_dir = output_dir / "future_features"
    training_dir = output_dir / "training"
    model_dir = output_dir / "models"
    for directory in (products_dir, feature_dir, training_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    latitude = np.arange(5.5, 38.0, 1.0, dtype=np.float32)
    longitude = np.arange(65.5, 100.0, 1.0, dtype=np.float32)
    climate_latitude = np.arange(5.0, 40.0, 2.0, dtype=np.float32)
    climate_longitude = np.arange(65.0, 102.0, 2.0, dtype=np.float32)
    base = _base_features(climate_latitude, climate_longitude)
    grid_path = output_dir / "target_grid_india_1deg_demo.nc"
    grid = xr.Dataset(coords={"lat": latitude, "lon": longitude}, attrs=_demo_attrs(
        grid_spacing_degrees=1.0,
        target_resolution="1degree_demo",
        grid_description="1-degree rectangular bounding grid over India; not an India land boundary or mask",
        latitude_bounds="5N to 38N cell centers",
        longitude_bounds="65E to 100E cell centers",
    ))
    grid.to_netcdf(grid_path)

    historical_times = pd.date_range("2000-01-01", periods=24, freq="MS")
    historical_by_model = {
        name: _make_monthly_dataset(
            base, climate_latitude, climate_longitude, historical_times,
            precipitation_factor=(0.97, 1.0, 1.03)[index],
            temperature_offset=(-0.25, 0.0, 0.25)[index],
        )
        for index, name in enumerate(MODEL_NAMES)
    }
    observed = _make_monthly_dataset(
        base, climate_latitude, climate_longitude, historical_times,
        precipitation_factor=1.08,
        temperature_offset=0.65,
    )
    observed["maximum_temperature_c"] = observed["maximum_temperature_c"] + 0.25
    observed.attrs.update(_demo_attrs(
        dataset_role="fake observed monthly reference",
        time_period="2000-2001 synthetic monthly values",
    ))
    reference_path = output_dir / "fake_observed_reference_demo.nc"
    observed.to_netcdf(reference_path)

    sample_count = 800
    sampled_latitude = rng.uniform(latitude.min(), latitude.max(), sample_count)
    sampled_longitude = rng.uniform(longitude.min(), longitude.max(), sample_count)
    sample_base = _point_features(sampled_latitude, sampled_longitude)
    sample_base["elevation_m"] = _elevation(sampled_latitude, sampled_longitude)
    sample_month = rng.integers(1, 13, sample_count)
    seasonal = np.sin(2.0 * np.pi * (sample_month - 1) / 12.0 - 0.8)
    training_values = {
        FEATURE_NAMES[0]: sample_base[FEATURE_NAMES[0]] * (1.0 + 0.35 * seasonal) * rng.normal(1.02, 0.07, sample_count),
        FEATURE_NAMES[1]: sample_base[FEATURE_NAMES[1]] + 2.5 * seasonal + rng.normal(0.0, 0.5, sample_count),
        FEATURE_NAMES[2]: sample_base[FEATURE_NAMES[2]] + 2.5 * seasonal + rng.normal(0.0, 0.5, sample_count),
        FEATURE_NAMES[3]: sample_base[FEATURE_NAMES[3]],
    }
    synthetic_label = (
        2.8
        + 0.018 * training_values[FEATURE_NAMES[0]]
        - 0.045 * training_values[FEATURE_NAMES[1]]
        + 0.028 * training_values[FEATURE_NAMES[2]]
        + 0.00008 * training_values[FEATURE_NAMES[3]]
        + rng.normal(0.0, 0.12, sample_count)
    )
    training = xr.Dataset(
        {
            **{name: (("sample",), values.astype(np.float32)) for name, values in training_values.items()},
            "synthetic_salinity_label": (("sample",), synthetic_label.astype(np.float32)),
        },
        coords={"sample": np.arange(sample_count)},
        attrs=_demo_attrs(dataset_role="synthetic training table", training_samples=sample_count),
    )
    training_path = training_dir / "training_features_demo.nc"
    training.to_netcdf(training_path)

    model = _fit_demo_model(training)
    model_path = model_dir / "synthetic_demo_model.json"
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    contract = {
        "feature_names_in_order": FEATURE_NAMES,
        "cmip6_variable_map": FEATURE_MAP,
        "features": [
            {
                "name": "precipitation_mm_day",
                "cmip6_variable_id": "pr",
                "units": "mm/day",
                "preprocessing": "synthetic monthly precipitation; bias-adjusted by multiplicative monthly delta",
            },
            {
                "name": "mean_temperature_c",
                "cmip6_variable_id": "tas",
                "units": "degC",
                "preprocessing": "synthetic monthly mean temperature; bias-adjusted by additive monthly delta",
            },
            {
                "name": "maximum_temperature_c",
                "cmip6_variable_id": "tasmax",
                "units": "degC",
                "preprocessing": "synthetic monthly maximum temperature; bias-adjusted by additive monthly delta",
            },
            {
                "name": "elevation_m",
                "units": "m",
                "preprocessing": "synthetic static elevation surface; demo only",
            },
        ],
        "target_name": "synthetic_salinity_label",
        "target_units": "unitless synthetic demo index",
        "model_artifact": str(model_path.relative_to(output_dir)).replace("\\", "/"),
        "training_dataset": str(training_path.relative_to(output_dir)).replace("\\", "/"),
        "demo_status": "synthetic_demo",
        "warning": DEMO_LABEL,
    }
    contract_path = output_dir / "feature_contract_demo.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    static_features = _static_features(latitude, longitude)
    climate_products: dict[tuple[str, str], xr.Dataset] = {}
    climate_products[("current", "current")] = _create_feature_product(
        "current", "current", 0.0, base, climate_latitude, climate_longitude,
        observed, historical_by_model, grid, feature_dir,
    )
    for scenario, periods in SCENARIO_WARMING.items():
        for period, warming in periods.items():
            climate_products[(scenario, period)] = _create_feature_product(
                scenario, period, warming, base, climate_latitude, climate_longitude,
                observed, historical_by_model, grid, feature_dir,
            )

    feature_products: dict[tuple[str, str], xr.Dataset] = {}
    for key, climate_features in climate_products.items():
        assembled_features = assemble_model_features(
            climate_features, static_features, contract_path
        )
        assembled_features.to_netcdf(
            feature_dir / f"model_features_{key[0]}_{key[1]}_demo.nc"
        )
        feature_products[key] = assembled_features

    applicability_by_product = {}
    for key, features in feature_products.items():
        applicability = assess_applicability(
            training,
            features,
            FEATURE_NAMES,
            bootstrap_replicates=40,
            random_seed=RANDOM_SEED,
        )
        applicability.attrs.update(_demo_attrs(
            scenario=key[0],
            period=key[1],
            method_note="synthetic training-envelope screen; demo only",
        ))
        applicability_by_product[key] = applicability
        applicability.to_netcdf(feature_dir / f"applicability_{key[0]}_{key[1]}_demo.nc")

    written_products = {}
    prediction_dir = output_dir / "prediction_ensembles"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for (scenario, period), features in feature_products.items():
        prediction_ensemble = _predict_ensemble(features, model, rng)
        prediction_ensemble = xr.merge(
            [prediction_ensemble, applicability_by_product[(scenario, period)]],
            compat="override",
        )
        ensemble_path = prediction_dir / f"predictions_{scenario}_{period}_demo.nc"
        prediction_ensemble.attrs.update(_demo_attrs(scenario=scenario, period=period))
        prediction_ensemble.to_netcdf(ensemble_path)

        product = _collapse_applicability_layers(quantify_prediction_uncertainty(prediction_ensemble))
        for feature_name in FEATURE_NAMES:
            product[f"driver_{feature_name}"] = features[feature_name].mean("climate_model")
            product[f"driver_{feature_name}"].attrs.update({
                "demo_status": "synthetic_demo",
                "feature_name": feature_name,
            })
        product.attrs.update(_demo_attrs(
            product_role="API-ready standardized salinity demo product",
            scenario=scenario,
            period=period,
            grid_spacing_degrees=1.0,
            original_climate_resolution="2degree synthetic demo grid",
            target_resolution="1degree_demo",
            regridding_method="bilinear via xarray.interp",
            member_strategy="all three synthetic demo climate models",
            target_name="synthetic_salinity_label",
            target_units="unitless synthetic demo index",
            feature_contract="feature_contract_demo.json",
            model_artifact="synthetic_demo_model.json",
        ))
        filename = "current.nc" if scenario == "current" else f"future_{scenario}_{period}.nc"
        product_path = products_dir / filename
        product.to_netcdf(product_path)
        written_products[f"{scenario}_{period}"] = product_path

    return {
        "output_dir": output_dir,
        "grid": grid_path,
        "fake_observed_reference": reference_path,
        "training_features": training_path,
        "model": model_path,
        "feature_contract": contract_path,
        "products_dir": products_dir,
        **{f"product_{key}": value for key, value in written_products.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explicitly synthetic API demo data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "demo",
        help="Isolated output folder (default: Backend/data/demo)",
    )
    args = parser.parse_args()
    for label, path in generate_demo(args.output_dir).items():
        print(f"{label}: {path}")
    print(f"WARNING: {DEMO_LABEL}")


if __name__ == "__main__":
    main()
"""Assemble and validate model-ready climate and static feature datasets."""

import json
from collections.abc import Mapping
from pathlib import Path

import xarray as xr

from .validation import CANONICAL_UNITS, validate_and_convert_units


def _load_dataset(dataset: xr.Dataset | Path) -> xr.Dataset:
    if isinstance(dataset, xr.Dataset):
        return dataset
    with xr.open_dataset(dataset) as opened:
        return opened.load()


def _load_contract(contract: Mapping[str, object] | Path) -> Mapping[str, object]:
    if isinstance(contract, Mapping):
        return contract
    return json.loads(contract.read_text(encoding="utf-8"))


def assemble_model_features(
    future_climate: xr.Dataset | Path,
    static_features: xr.Dataset | Path,
    feature_contract: Mapping[str, object] | Path,
    output_path: Path | None = None,
) -> xr.Dataset:
    """Align inputs and emit exactly the ordered feature contract with validated units."""
    climate = _load_dataset(future_climate)
    static = _load_dataset(static_features)
    contract = _load_contract(feature_contract)
    feature_names = contract.get("feature_names_in_order")
    variable_map = contract.get("cmip6_variable_map", {})
    feature_specs = contract.get("features", [])
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Feature contract must define non-empty feature_names_in_order")
    if not isinstance(variable_map, Mapping) or not isinstance(feature_specs, list):
        raise ValueError("Feature contract has invalid variable mapping or feature definitions")
    if "lat" not in climate.coords or "lon" not in climate.coords:
        raise ValueError("Future climate features must contain lat/lon coordinates")

    units_by_name = {
        specification.get("name"): specification.get("units")
        for specification in feature_specs
        if isinstance(specification, Mapping)
    }
    missing_units = [name for name in feature_names if not units_by_name.get(name)]
    if missing_units:
        raise ValueError(f"Feature contract is missing units for: {missing_units}")

    assembled: dict[str, xr.DataArray] = {}
    for name in feature_names:
        in_climate = name in climate.data_vars
        in_static = name in static.data_vars
        if in_climate == in_static:
            state = "missing" if not in_climate else "ambiguous in both inputs"
            raise ValueError(f"Feature {name!r} is {state}")
        variable = (climate if in_climate else static)[name]
        if not in_climate:
            if "lat" not in static.coords or "lon" not in static.coords:
                raise ValueError(f"Static features must contain lat/lon coordinates: {name!r}")
            coordinates_match = static.lat.equals(climate.lat) and static.lon.equals(climate.lon)
            if not coordinates_match:
                if static.lat.ndim != 1 or static.lon.ndim != 1:
                    raise ValueError(
                        f"Curvilinear static feature {name!r} must already match the climate grid"
                    )
                variable = variable.interp(
                    lat=climate.lat,
                    lon=climate.lon,
                    method="linear",
                )
            if "climate_model" in climate.dims and "climate_model" not in variable.dims:
                variable = variable.expand_dims(climate_model=climate.climate_model)
        variable_id = variable_map.get(name) if in_climate else None
        expected_units = str(units_by_name[name])
        if variable_id in CANONICAL_UNITS:
            if expected_units != CANONICAL_UNITS[variable_id]:
                raise ValueError(
                    f"Feature contract unit for {name!r} must be {CANONICAL_UNITS[variable_id]!r}"
                )
            variable = validate_and_convert_units(variable, str(variable_id))
        else:
            variable = validate_and_convert_units(
                variable, str(variable_id or name), target_units=expected_units
            )
        variable.attrs = variable.attrs.copy()
        variable.attrs["units"] = expected_units
        assembled[name] = variable

    aligned = xr.align(*assembled.values(), join="exact", copy=False)
    result = xr.Dataset(
        {name: variable for name, variable in zip(assembled, aligned)},
        attrs=dict(climate.attrs),
    )
    result.attrs.update({
        "feature_names_in_order": json.dumps(feature_names),
        "feature_contract": str(feature_contract) if isinstance(feature_contract, Path) else "provided mapping",
        "feature_assembly": "climate and static inputs aligned with exact coordinate joins",
    })
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(output_path)
    return result
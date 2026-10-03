from __future__ import annotations

from typing import Final, Iterable

import numpy as np
import xarray as xr


CANONICAL_UNITS: Final[dict[str, str]] = {
    "pr": "mm/day",
    "tas": "degC",
    "tasmin": "degC",
    "tasmax": "degC",
}

CMIP6_UNIT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "pr": ("kgm-2s-1", "kg/m2/s", "mmday-1", "mm/day", "mmd-1"),
    "tas": ("k", "kelvin", "degc", "c", "celsius"),
    "tasmin": ("k", "kelvin", "degc", "c", "celsius"),
    "tasmax": ("k", "kelvin", "degc", "c", "celsius"),
}


def _normalize_units(units: str | None) -> str:
    if units is None:
        raise ValueError("Unit metadata is missing")
    normalized = units.strip().lower().replace("°", "deg")
    normalized = normalized.replace("^", "").replace(" ", "")
    normalized = normalized.replace("m^-2", "m-2")
    normalized = normalized.replace("m^2", "m2")
    normalized = normalized.replace("d^-1", "d-1")
    normalized = normalized.replace("day-1", "day-1")
    return normalized


def infer_cmip_variable_name(variable_name: str) -> str:
    """Map user-facing feature names to CMIP6 variable IDs when possible."""
    lookup = {
        "pr": "pr",
        "precipitation": "pr",
        "precipitation_mm_day": "pr",
        "tas": "tas",
        "temperature": "tas",
        "mean_temperature_c": "tas",
        "tasmax": "tasmax",
        "max_temperature": "tasmax",
        "maximum_temperature_c": "tasmax",
    }
    canonical = variable_name.strip().lower()
    return lookup.get(canonical, canonical)


def validate_and_convert_units(
    values: xr.DataArray,
    variable_id: str,
    *,
    target_units: str | None = None,
) -> xr.DataArray:
    """Validate units and emit precipitation in mm/day and temperature in degC."""
    units = _normalize_units(values.attrs.get("units"))
    variable_id = str(variable_id).strip().lower()
    expected_units = CANONICAL_UNITS.get(variable_id)

    if variable_id == "pr":
        if units in {"kgm-2s-1", "kg/m2/s"}:
            canonical = values * 86400.0
        elif units in {"mmday-1", "mm/day", "mmd-1"}:
            canonical = values.copy()
        else:
            raise ValueError(
                f"Unsupported units for {variable_id!r}: {values.attrs.get('units')!r}. "
                "Expected one of: kg m-2 s-1, mm day-1."
            )
        canonical.attrs = values.attrs.copy()
        canonical.attrs["units"] = expected_units or "mm/day"
        return canonical

    if variable_id in {"tas", "tasmin", "tasmax"}:
        if units in {"k", "kelvin"}:
            canonical = values - 273.15
        elif units in {"degc", "c", "celsius"}:
            canonical = values.copy()
        else:
            raise ValueError(
                f"Unsupported units for {variable_id!r}: {values.attrs.get('units')!r}. "
                "Expected one of: K, degC."
            )
        canonical.attrs = values.attrs.copy()
        canonical.attrs["units"] = expected_units or "degC"
        return canonical

    if target_units is not None:
        target = _normalize_units(target_units)
        if target == units:
            return values.copy()
        raise ValueError(
            f"Unsupported unit conversion for {variable_id!r}: "
            f"expected {target_units!r}, got {values.attrs.get('units')!r}."
        )

    raise ValueError(f"No unit validation rule is defined for {variable_id!r}")


def _as_numeric_array(data: xr.DataArray | xr.Dataset, name: str) -> np.ndarray:
    if isinstance(data, xr.Dataset):
        if name not in data:
            raise KeyError(f"Dataset is missing required variable {name!r}")
        data = data[name]
    values = np.asarray(data.values, dtype=float)
    if values.size == 0:
        raise ValueError(f"Variable {name!r} has no values for reconstruction metrics")
    return values[np.isfinite(values)]


def calculate_historical_reconstruction_metrics(
    observed: xr.Dataset,
    reconstructed: xr.Dataset,
    variables: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate MAE, RMSE, bias, and correlation for reconstructed climate variables."""
    observed = observed.copy()
    reconstructed = reconstructed.copy()
    candidate_variables = list(variables) if variables is not None else sorted(set(observed.data_vars) & set(reconstructed.data_vars))
    if not candidate_variables:
        raise ValueError("No overlapping variables were provided for historical reconstruction validation")

    metrics: dict[str, dict[str, float]] = {}
    for variable_name in candidate_variables:
        observed_values = _as_numeric_array(observed, variable_name)
        reconstructed_values = _as_numeric_array(reconstructed, variable_name)
        if observed_values.size != reconstructed_values.size:
            raise ValueError(
                f"Variable {variable_name!r} has mismatched observation/reconstruction lengths: "
                f"{observed_values.size} != {reconstructed_values.size}"
            )
        errors = reconstructed_values - observed_values
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        bias = float(np.mean(errors))
        denominator = np.std(observed_values, ddof=1) * np.std(reconstructed_values, ddof=1)
        correlation = 1.0 if np.isclose(denominator, 0.0) else float(np.corrcoef(observed_values, reconstructed_values)[0, 1])
        metrics[variable_name] = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "correlation": correlation,
        }
    return metrics


def validate_historical_reconstruction(
    observed: xr.Dataset,
    reconstructed: xr.Dataset,
    variables: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Convenience wrapper for climate reconstruction validation."""
    return calculate_historical_reconstruction_metrics(observed, reconstructed, variables=variables)

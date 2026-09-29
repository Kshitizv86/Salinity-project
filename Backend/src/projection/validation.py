from __future__ import annotations

from typing import Final

import xarray as xr


CANONICAL_UNITS: Final[dict[str, str]] = {
    "pr": "mm/day",
    "tas": "degC",
    "tasmin": "degC",
    "tasmax": "degC",
}


def _normalize_units(units: str | None) -> str:
    if units is None:
        raise ValueError("Unit metadata is missing")
    normalized = units.strip().lower().replace("°", "deg")
    normalized = normalized.replace("^", "").replace(" ", "")
    normalized = normalized.replace("m^-2", "m-2")
    normalized = normalized.replace("m^2", "m2")
    normalized = normalized.replace("d^-1", "d-1")
    return normalized


def validate_and_convert_units(
    values: xr.DataArray,
    variable_id: str,
    *,
    target_units: str | None = None,
) -> xr.DataArray:
    """Validate units and emit precipitation in mm/day and temperature in degC."""
    units = _normalize_units(values.attrs.get("units"))
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
        canonical.attrs["units"] = CANONICAL_UNITS[variable_id]
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
        canonical.attrs["units"] = CANONICAL_UNITS[variable_id]
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

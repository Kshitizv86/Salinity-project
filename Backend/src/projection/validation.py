from __future__ import annotations

from typing import Final

import xarray as xr


PRECIPITATION_UNITS: Final[set[str]] = {
    "kg m-2 s-1",
    "kg m^-2 s^-1",
    "kg/m2/s",
    "kg m-2 s^-1",
    "mm day-1",
    "mm/day",
    "mm d-1",
}
TEMPERATURE_UNITS: Final[set[str]] = {
    "k",
    "kelvin",
    "degc",
    "c",
    "celsius",
    "°c",
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
    """Validate a variable's units and convert compatible values to a common unit."""
    units = _normalize_units(values.attrs.get("units"))
    if variable_id == "pr":
        if units in {"kgm-2s-1", "kgm-2s^-1", "kgm2s"}:
            canonical = values.copy()
        elif units in {"mmday-1", "mm/day", "mmd-1"}:
            canonical = values / 86400.0
            canonical.attrs = values.attrs.copy()
            canonical.attrs["units"] = "kg m-2 s-1"
            return canonical
        else:
            raise ValueError(
                f"Unsupported units for {variable_id!r}: {values.attrs.get('units')!r}. "
                "Expected one of: kg m-2 s-1, mm day-1."
            )
        canonical = values.copy()
        canonical.attrs = values.attrs.copy()
        canonical.attrs["units"] = "kg m-2 s-1"
        return canonical

    if variable_id in {"tas", "tasmax"}:
        if units in {"k", "kelvin"}:
            canonical = values.copy()
        elif units in {"degc", "c", "celsius"}:
            canonical = values + 273.15
            canonical.attrs = values.attrs.copy()
            canonical.attrs["units"] = "K"
            return canonical
        else:
            raise ValueError(
                f"Unsupported units for {variable_id!r}: {values.attrs.get('units')!r}. "
                "Expected one of: K, degC."
            )
        canonical = values.copy()
        canonical.attrs = values.attrs.copy()
        canonical.attrs["units"] = "K"
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

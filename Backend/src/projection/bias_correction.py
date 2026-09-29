import xarray as xr

from .validation import validate_and_convert_units


def monthly_delta_correction(
    future: xr.Dataset,
    historical: xr.Dataset,
    observed: xr.Dataset,
    feature_map: dict[str, str],
) -> xr.Dataset:
    """Apply monthly additive temperature and multiplicative precipitation deltas.

    All three datasets must be on the same grid and use matching monthly time
    coverage for their historical/observed baseline. Precipitation is clamped
    to zero after ratio correction. This is a transparent mean-delta method,
    not quantile mapping; the chosen method and reference are recorded in attrs.
    """
    if not feature_map:
        raise ValueError("Feature map must name the model features to correct")
    for name in ("time",):
        if name not in future.coords or name not in historical.coords or name not in observed.coords:
            raise ValueError(f"All input datasets must have a {name!r} coordinate")

    corrected = xr.Dataset(attrs=future.attrs.copy())
    for feature_name, variable_id in feature_map.items():
        for label, dataset in (("future", future), ("historical", historical), ("observed", observed)):
            if feature_name not in dataset:
                raise ValueError(f"{label} dataset is missing feature {feature_name!r}")

        future_variable = future[feature_name]
        historical_variable = historical[feature_name]
        observed_variable = observed[feature_name]
        future_unit_checked = validate_and_convert_units(future_variable, variable_id)
        historical_unit_checked = validate_and_convert_units(historical_variable, variable_id)
        observed_unit_checked = validate_and_convert_units(observed_variable, variable_id)
        historical_monthly = historical_unit_checked.groupby("time.month").mean("time")
        observed_monthly = observed_unit_checked.groupby("time.month").mean("time")
        future_original_units = future_variable.attrs.get("units")
        if variable_id == "pr":
            ratio = observed_monthly / historical_monthly.where(historical_monthly > 1e-6)
            adjusted = (future_unit_checked.groupby("time.month") * ratio).clip(min=0)
            if future_original_units is not None and "mm" in future_original_units.lower():
                adjusted = adjusted * 86400.0
        else:
            delta = observed_monthly - historical_monthly
            adjusted = future_unit_checked.groupby("time.month") + delta
            if future_original_units is not None and "degc" in future_original_units.lower().replace(" ", ""):
                adjusted = adjusted - 273.15
        corrected[feature_name] = adjusted.transpose(*future_variable.dims)
        corrected[feature_name].attrs.update(future_variable.attrs)
        corrected[feature_name].attrs["units"] = future_original_units or corrected[feature_name].attrs.get("units")
        corrected[feature_name].attrs["bias_correction"] = (
            "monthly multiplicative delta" if variable_id == "pr" else "monthly additive delta"
        )

    corrected.attrs.update({
        "bias_correction_method": "monthly delta correction",
        "bias_correction_reference": "observed dataset supplied by caller",
        "feature_map": str(feature_map),
    })
    return corrected
import argparse
import json
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

from .bias_correction import monthly_delta_correction
from .validation import validate_and_convert_units


CMIP6_CATALOG_URL = "https://cmip6.storage.googleapis.com/pangeo-cmip6.csv"
SCENARIOS = {"ssp245": "ssp245", "ssp585": "ssp585"}
PERIODS = {"2030": (2021, 2040), "2050": (2041, 2060), "2080": (2071, 2090)}
def _standardize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    aliases = {"latitude": "lat", "longitude": "lon"}
    rename = {old: new for old, new in aliases.items() if old in ds.coords and new not in ds.coords}
    ds = ds.rename(rename)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("CMIP6 dataset must provide latitude/longitude coordinates")
    if ds.lon.ndim == 1:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    if ds.lat.ndim != 1 or ds.lon.ndim != 1:
        raise ValueError("Only rectilinear CMIP6 grids are supported for interpolation")
    return ds.sortby("lat")


def _load_target_grid(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as grid:
        if "lat" not in grid or "lon" not in grid:
            raise ValueError("Target grid file must contain 'lat' and 'lon'")
        return grid.load()


def regrid_to_target_grid(
    dataset: xr.Dataset,
    target_grid: xr.Dataset,
    *,
    require_india_mask: bool = False,
) -> xr.Dataset:
    """Interpolate a rectilinear climate dataset and optionally apply the India land mask."""
    if "lat" not in target_grid or "lon" not in target_grid:
        raise ValueError("Target grid must contain 'lat' and 'lon'")
    if require_india_mask and "india_mask" not in target_grid:
        raise ValueError("Production target grid must include an India boundary polygon mask")

    regridded = dataset.interp(
        lat=target_grid.lat,
        lon=target_grid.lon,
        method="linear",
    )
    if "india_mask" in target_grid:
        regridded = regridded.where(target_grid.india_mask.astype(bool))
    regridded.attrs.update(dataset.attrs)
    regridded.attrs.update({
        "original_climate_resolution": dataset.attrs.get(
            "original_climate_resolution", "CMIP6_native"
        ),
        "target_resolution": target_grid.attrs.get("target_resolution", "unspecified"),
        "regridding_method": "bilinear",
        "india_land_mask_applied": "india_mask" in target_grid,
    })
    return regridded


def _load_cmip_variable(fs, row, variable_id: str, start_year: int, end_year: int,
                        target_grid: xr.Dataset) -> xr.DataArray:
    with xr.open_zarr(fs.get_mapper(row.zstore), consolidated=True, use_cftime=True) as source:
        ds = _standardize_coordinates(source)
        variable = ds[variable_id]
        years = ds.time.dt.year
        variable = variable.where((years >= start_year) & (years <= end_year), drop=True)
        if variable.sizes.get("time", 0) == 0:
            raise ValueError(f"No {variable_id} data for {start_year}-{end_year}")

        variable = validate_and_convert_units(variable, variable_id)
        source_dataset = xr.Dataset({variable_id: variable})
        source_dataset.attrs["original_climate_resolution"] = "CMIP6_native"
        return regrid_to_target_grid(
            source_dataset, target_grid, require_india_mask=True
        )[variable_id].load()


def build_future_features(
    feature_map: dict[str, str],
    target_grid_path: Path,
    output_dir: Path,
    model_ids: list[str] | None = None,
    catalog_url: str = CMIP6_CATALOG_URL,
    observed_reference_path: Path | None = None,
    historical_start_year: int = 1995,
    historical_end_year: int = 2014,
    member_strategy: str = "first_common",
) -> list[Path]:
    """Build canonical-unit CMIP6 features on a boundary-masked 1-km grid."""
    if not feature_map:
        raise ValueError("Provide the validated model feature-name to CMIP6-variable mapping")
    if member_strategy != "first_common":
        raise ValueError("Supported CMIP6 realization strategy is 'first_common'")
    if not target_grid_path.is_file():
        raise FileNotFoundError(f"Target grid not found: {target_grid_path}")

    target_grid = _load_target_grid(target_grid_path)
    if "india_mask" not in target_grid:
        raise ValueError(
            "Production future maps require a target grid created from an India boundary shapefile"
        )
    if int(target_grid.attrs.get("target_resolution_m", 0)) != 1000:
        raise ValueError("Production future maps require a 1-km target grid")
    catalog = pd.read_csv(catalog_url)
    required_columns = {"table_id", "variable_id", "experiment_id", "source_id", "zstore"}
    missing = required_columns - set(catalog.columns)
    if missing:
        raise ValueError(f"CMIP6 catalog is missing columns: {sorted(missing)}")
    observed_reference = None
    if observed_reference_path is not None:
        if not observed_reference_path.is_file():
            raise FileNotFoundError(f"Observed reference not found: {observed_reference_path}")
        with xr.open_dataset(observed_reference_path) as reference:
            observed_reference = reference.load()
        if "time" not in observed_reference.coords:
            raise ValueError("Observed reference must contain monthly time-series feature data")
    fs = gcsfs.GCSFileSystem(token="anon")
    outputs = []

    for scenario, experiment in SCENARIOS.items():
        for period, (start_year, end_year) in PERIODS.items():
            candidates = catalog[
                (catalog.table_id == "Amon")
                & (catalog.experiment_id == experiment)
                & (catalog.variable_id.isin(set(feature_map.values())))
            ]
            model_sets = [
                set(candidates.loc[candidates.variable_id == variable_id, "source_id"])
                for variable_id in feature_map.values()
            ]
            common_models = set.intersection(*model_sets) if model_sets else set()
            if model_ids:
                common_models &= set(model_ids)
            if not common_models:
                raise ValueError(f"No CMIP6 models provide all requested features for {scenario}")

            model_datasets = []
            selected_members: dict[str, str | None] = {}
            for model_id in sorted(common_models):
                rows_by_feature = {
                    variable_id: candidates[
                        (candidates.source_id == model_id)
                        & (candidates.variable_id == variable_id)
                    ]
                    for variable_id in feature_map.values()
                }
                member_column = "member_id" if "member_id" in candidates.columns else None
                common_members = set.intersection(*[
                    set(rows[member_column].dropna()) for rows in rows_by_feature.values()
                ]) if member_column else set()
                if member_column and not common_members:
                    continue
                selected_member = sorted(common_members)[0] if common_members else None
                selected_members[model_id] = selected_member
                feature_arrays = {}
                for feature_name, variable_id in feature_map.items():
                    rows = rows_by_feature[variable_id]
                    if selected_member is not None:
                        rows = rows[rows[member_column] == selected_member]
                    future_row = rows.sort_values("zstore").iloc[0]
                    future_monthly = _load_cmip_variable(
                        fs, future_row, variable_id, start_year, end_year, target_grid
                    )
                    if observed_reference is not None:
                        historical_rows = catalog[
                            (catalog.table_id == "Amon")
                            & (catalog.experiment_id == "historical")
                            & (catalog.source_id == model_id)
                            & (catalog.variable_id == variable_id)
                        ]
                        if member_column and selected_member is not None:
                            historical_rows = historical_rows[
                                historical_rows[member_column] == selected_member
                            ]
                        if historical_rows.empty:
                            raise ValueError(
                                f"No historical CMIP6 baseline for {model_id}/{variable_id}/"
                                f"{selected_member}"
                            )
                        if feature_name not in observed_reference:
                            raise ValueError(f"Observed reference is missing {feature_name!r}")
                        historical_monthly = _load_cmip_variable(
                            fs, historical_rows.sort_values("zstore").iloc[0], variable_id,
                            historical_start_year, historical_end_year, target_grid,
                        )
                        corrected = monthly_delta_correction(
                            xr.Dataset({feature_name: future_monthly}),
                            xr.Dataset({feature_name: historical_monthly}),
                            observed_reference,
                            {feature_name: variable_id},
                        )[feature_name]
                        feature_arrays[feature_name] = corrected.mean("time", keep_attrs=True)
                    else:
                        feature_arrays[feature_name] = future_monthly.mean("time", keep_attrs=True)
                if not feature_arrays:
                    continue
                model_datasets.append(xr.Dataset(feature_arrays).expand_dims(climate_model=[model_id]))
            if not model_datasets:
                raise ValueError(f"No compatible CMIP6 members remain for {scenario}")

            result = xr.concat(model_datasets, dim="climate_model", join="exact")
            result.attrs.update({
                "scenario": scenario,
                "experiment_id": experiment,
                "period": period,
                "period_start_year": start_year,
                "period_end_year": end_year,
                "feature_map": json.dumps(feature_map, sort_keys=True),
                "target_grid": str(target_grid_path.resolve()),
                "original_climate_resolution": "CMIP6_native",
                "target_resolution": target_grid.attrs["target_resolution"],
                "regridding_method": "bilinear via xarray.interp",
                "geographic_domain": "India boundary polygon mask",
                "member_strategy": member_strategy,
                "selected_members": json.dumps(selected_members, sort_keys=True),
                "canonical_units": json.dumps({
                    name: values.attrs.get("units") for name, values in result.data_vars.items()
                }, sort_keys=True),
                "bias_correction": (
                    "monthly additive temperature/multiplicative precipitation delta correction"
                    if observed_reference is not None else "not applied; no observed reference supplied"
                ),
            })
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"future_features_{scenario}_{period}.nc"
            result.to_netcdf(output_path)
            outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CMIP6 future climate features")
    parser.add_argument("--feature-map", type=Path, required=True,
                        help="JSON object mapping exact model feature names to CMIP6 variable IDs")
    parser.add_argument("--target-grid", type=Path, required=True,
                        help="NetCDF grid containing lat/lon coordinates")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "future")
    parser.add_argument("--models", nargs="*", help="Optional CMIP6 source_id allowlist")
    parser.add_argument(
        "--member-strategy", choices=("first_common",), default="first_common",
        help="Realization selection policy; first_common selects the first shared member per GCM",
    )
    args = parser.parse_args()
    feature_map = json.loads(args.feature_map.read_text(encoding="utf-8"))
    paths = build_future_features(
        feature_map, args.target_grid, args.output_dir, args.models,
        member_strategy=args.member_strategy,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
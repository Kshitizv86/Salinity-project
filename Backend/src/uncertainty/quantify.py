from pathlib import Path
import numpy as np
import xarray as xr

def calculate_uncertainty():
    # 1. Define paths relative to workspace root
    base_dir = Path.cwd()
    input_file = base_dir / "Backend" / "data" / "processed" / "regridded_salinity.nc"
    output_file = base_dir / "Backend" / "data" / "processed" / "salinity_uncertainty.nc"

    # 2. Load regridded data
    ds = xr.open_dataset(input_file)

    # Determine core dataset key ('salinity' or 'so')
    var_name = "salinity" if "salinity" in ds else "so"

    # 3. Compute uncertainty metrics
    if "model" in ds[var_name].dims:
        uncertainty_std = ds[var_name].std(dim="time", keep_attrs=True)
        mean_salinity = ds[var_name].mean(dim="time", keep_attrs=True)
    else:
        uncertainty_std = ds[var_name].std(keep_attrs=True)
        mean_salinity = ds[var_name].mean(keep_attrs=True)

    # 4. Create output dataset
    ds_out = xr.Dataset(
        data_vars={
            "salinity_mean": mean_salinity,
            "salinity_uncertainty_std": uncertainty_std,
        },
        attrs={"description": "Salinity projection uncertainty analysis"},
    )

    # 5. Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_file)
    print(f"Uncertainty quantification complete. Saved to {output_file}")

if __name__ == "__main__":
    calculate_uncertainty()
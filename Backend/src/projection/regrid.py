from pathlib import Path
import gcsfs
import pandas as pd
import xarray as xr


def regrid_salinity():
    print("1/4: Fetching CMIP6 Catalog...")
    df = pd.read_csv("https://cmip6.storage.googleapis.com/pangeo-cmip6.csv")

    salinity_datasets = df.query(
        "table_id == 'Omon' & variable_id == 'so' & experiment_id == 'historical'"
    )
    first_dataset_url = salinity_datasets.iloc[0]["zstore"]

    print("2/4: Connecting to Zarr store...")
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(first_dataset_url)
    ds = xr.open_zarr(mapper, consolidated=True, use_cftime=True)

    salinity = ds["so"]

    print("3/4: Processing salinity data...")
    ds_regridded = salinity.isel(time=0, lev=0).squeeze()

    print("4/4: Saving NetCDF file to disk...")
    output_path = (
        Path.cwd() / "Backend" / "data" / "processed" / "regridded_salinity.nc"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_regridded.to_netcdf(output_path)

    print(f"SUCCESS: File saved at {output_path}")


if __name__ == "__main__":
    regrid_salinity()
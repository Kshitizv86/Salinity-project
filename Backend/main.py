import os
import numpy as np
import xarray as xr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded directly to your project file name visible in the folder
DATASET_PATH = "C:/Users/DELL/Downloads/Salinity project/regridded_salinity.nc"

@app.get("/api/v1/salinity")
def get_salinity(lat: float, lon: float):
    try:
        # Fallback check if it's run from inside the Backend folder
        path_to_use = DATASET_PATH
        if not os.path.exists(path_to_use) and os.path.exists(os.path.join("..", DATASET_PATH)):
            path_to_use = os.path.join("..", DATASET_PATH)

        if not os.path.exists(path_to_use):
            return {"mean_salinity": f"Error: '{DATASET_PATH}' not found", "uncertainty_sd": "N/A"}

        ds = xr.open_dataset(path_to_use)
        
        # 1. Drop extra dimensions (time, depth, bounds)
        for dim_name in ['time', 'time_bnds', 'bnds', 'bound', 'depth', 'lev', 'level', 'plev']:
            if dim_name in ds.dims:
                ds = ds.isel({dim_name: 0}, drop=True)
            if dim_name in ds.coords or dim_name in ds.variables:
                ds = ds.drop_vars(dim_name, errors='ignore')

        # 2. Grab available coordinates/dimensions safely
        all_keys = list(ds.coords.keys()) if ds.coords else list(ds.dims.keys())
        if len(all_keys) < 2:
            return {"mean_salinity": "Error: Insufficient dimensions", "uncertainty_sd": "N/A"}

        lat_name = all_keys[0]
        lon_name = all_keys[1]

        # 3. Handle longitude wrapping (0-360 vs -180 to 180)
        max_lon = float(ds[lon_name].max()) if lon_name in ds else 180.0
        target_lon = lon + 360 if (max_lon > 180 and lon < 0) else lon

        # 4. Spatial lookup
        point = ds.sel({lat_name: lat, lon_name: target_lon}, method='nearest')

        # 5. Extract variables
        data_vars = list(ds.data_vars)
        if not data_vars:
            return {"mean_salinity": "No Data Variables", "uncertainty_sd": "N/A"}

        sal_var = data_vars[0]
        unc_var = data_vars[1] if len(data_vars) > 1 else None

        raw_sal = point[sal_var].values
        sal_val = raw_sal.item() if hasattr(raw_sal, 'item') else float(raw_sal)

        unc_val = "N/A"
        if unc_var and unc_var in point:
            raw_unc = point[unc_var].values
            unc_val = raw_unc.item() if hasattr(raw_unc, 'item') else float(raw_unc)

        if sal_val is None or np.isnan(sal_val):
            return {"mean_salinity": "Land / No Data", "uncertainty_sd": "N/A"}

        return {
            "mean_salinity": round(float(sal_val), 4),
            "uncertainty_sd": round(float(unc_val), 4) if isinstance(unc_val, (int, float)) else "N/A"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"mean_salinity": "Error", "uncertainty_sd": str(e)}
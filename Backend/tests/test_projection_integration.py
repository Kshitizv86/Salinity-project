import xarray as xr
import pytest

from Backend import main as api
from Backend.src.demo_data import generate_demo
from Backend.src.projection.feature_assembly import assemble_model_features
from Backend.src.projection.regrid import build_future_features


def test_synthetic_projection_pipeline_writes_all_api_products(tmp_path):
    demo_dir = tmp_path / "demo"
    outputs = generate_demo(demo_dir)

    assert len([key for key in outputs if key.startswith("product_")]) == 7
    with xr.open_dataset(demo_dir / "future_features" / "model_features_ssp245_2030_demo.nc") as features:
        assert set(features.data_vars) >= {
            "precipitation_mm_day",
            "mean_temperature_c",
            "maximum_temperature_c",
            "elevation_m",
        }
        assert features.precipitation_mm_day.attrs["units"] == "mm/day"
        assert features.mean_temperature_c.attrs["units"] == "degC"
        assert features.attrs["scenario"] == "ssp245"
        assert features.attrs["period"] == "2030"
        assert features.attrs["regridding_method"] == "bilinear"

    for scenario, periods in (("ssp245", ("2030", "2050", "2080")),
                              ("ssp585", ("2030", "2050", "2080"))):
        for period in periods:
            with xr.open_dataset(demo_dir / "products" / f"future_{scenario}_{period}.nc") as product:
                assert "salinity_mean" in product
                assert "climate_model_spread_sd" in product
                assert "model_uncertainty_sd" in product
                assert product.attrs["demo_status"] == "synthetic_demo"
                assert product.attrs["original_climate_resolution"] == "2degree synthetic demo grid"
                assert product.attrs["target_resolution"] == "1degree_demo"
                assert product.attrs["regridding_method"] == "bilinear via xarray.interp"


def test_products_endpoint_reports_nested_availability(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PRODUCT_DIR", tmp_path)
    (tmp_path / "current.nc").touch()
    (tmp_path / "future_ssp245_2030.nc").touch()

    assert api.get_products() == {
        "products": {
            "current": True,
            "ssp245": {"2030": True, "2050": False, "2080": False},
            "ssp585": {"2030": False, "2050": False, "2080": False},
        }
    }


def test_feature_assembly_regrids_static_features_to_climate_grid():
    climate = xr.Dataset(
        {"precipitation_mm_day": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mm/day"})},
        coords={"lat": [0.5, 1.5], "lon": [10.5, 11.5]},
    )
    static = xr.Dataset(
        {"elevation_m": (("lat", "lon"), [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]], {"units": "m"})},
        coords={"lat": [0.0, 1.0, 2.0], "lon": [10.0, 11.0, 12.0]},
    )
    contract = {
        "feature_names_in_order": ["precipitation_mm_day", "elevation_m"],
        "cmip6_variable_map": {"precipitation_mm_day": "pr"},
        "features": [
            {"name": "precipitation_mm_day", "units": "mm/day"},
            {"name": "elevation_m", "units": "m"},
        ],
    }

    assembled = assemble_model_features(climate, static, contract)

    assert assembled.elevation_m.dims == ("lat", "lon")
    assert assembled.elevation_m.values[0, 0] == 15.5


def test_production_future_builder_rejects_unmasked_grid_before_catalog_access(tmp_path):
    grid_path = tmp_path / "rectangle.nc"
    xr.Dataset(
        coords={"lat": [10.0], "lon": [70.0]},
        attrs={"target_resolution": "1km", "target_resolution_m": 1000},
    ).to_netcdf(grid_path)

    with pytest.raises(ValueError, match="India boundary shapefile"):
        build_future_features(
            {"precipitation_mm_day": "pr"},
            grid_path,
            tmp_path / "outputs",
            catalog_url="this-catalog-must-not-be-read.csv",
        )
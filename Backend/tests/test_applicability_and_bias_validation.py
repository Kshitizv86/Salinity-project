import numpy as np
import xarray as xr

from src.projection.applicability import assess_applicability
from src.projection.bias_correction import monthly_delta_correction


def test_assess_applicability_outputs_feature_diagnostics_and_probability_bounds():
    training = xr.Dataset(
        {
            "precipitation_mm_day": (("sample",), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])),
            "mean_temperature_c": (("sample",), np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])),
        },
        coords={"sample": np.arange(10)},
    )
    future = xr.Dataset(
        {
            "precipitation_mm_day": (("lat", "lon"), np.array([[3.0, 4.0], [5.0, 6.0]])),
            "mean_temperature_c": (("lat", "lon"), np.array([[12.0, 18.0], [22.0, 28.0]])),
        },
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )

    result = assess_applicability(
        training,
        future,
        ["precipitation_mm_day", "mean_temperature_c"],
        bootstrap_replicates=20,
        random_seed=7,
    )

    assert "shift_precipitation_mm_day" in result
    assert "shift_mean_temperature_c" in result
    assert np.isfinite(result["applicability_probability"]).all()
    assert ((result["applicability_probability"] >= 0.0) & (result["applicability_probability"] <= 1.0)).all()


def test_monthly_delta_correction_rejects_mismatched_units():
    time = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]")
    lat = np.array([0.0, 1.0])
    future = xr.Dataset(
        {
            "precipitation_mm_day": (("time", "lat"), np.array([[1.0, 2.0], [3.0, 4.0]]), {"units": "mm day-1"}),
            "mean_temperature_c": (("time", "lat"), np.array([[10.0, 11.0], [12.0, 13.0]]), {"units": "degC"}),
        },
        coords={"time": time, "lat": lat},
    )
    historical = xr.Dataset(
        {
            "precipitation_mm_day": (("time", "lat"), np.array([[2.0, 2.5], [3.0, 3.5]]), {"units": "mm day-1"}),
            "mean_temperature_c": (("time", "lat"), np.array([[9.0, 10.0], [11.0, 12.0]]), {"units": "degC"}),
        },
        coords={"time": time, "lat": lat},
    )
    observed = xr.Dataset(
        {
            "precipitation_mm_day": (("time", "lat"), np.array([[2.0, 3.0], [4.0, 5.0]]), {"units": "m s-1"}),
            "mean_temperature_c": (("time", "lat"), np.array([[8.0, 9.0], [10.0, 11.0]]), {"units": "degF"}),
        },
        coords={"time": time, "lat": lat},
    )

    feature_map = {"precipitation_mm_day": "pr", "mean_temperature_c": "tas"}

    try:
        monthly_delta_correction(future, historical, observed, feature_map)
    except ValueError as exc:
        assert "units" in str(exc).lower()
    else:
        raise AssertionError("monthly_delta_correction should reject mismatched units")

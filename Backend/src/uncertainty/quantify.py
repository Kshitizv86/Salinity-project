import argparse
from pathlib import Path

import xarray as xr


ENSEMBLE_DIMS = ("climate_model", "model_draw", "observation_draw")


def quantify_prediction_uncertainty(
    predictions: xr.Dataset,
    prediction_variable: str = "salinity_prediction",
) -> xr.Dataset:
    """Compute distinct spatial uncertainty layers from prediction ensembles.

    The prediction variable must retain independent climate-model, model-draw,
    and observation-draw dimensions. Standard deviations are calculated over
    one source at a time after averaging across the other sources.
    """
    if prediction_variable not in predictions:
        raise ValueError(f"Missing prediction variable {prediction_variable!r}")
    prediction = predictions[prediction_variable]
    missing = [dim for dim in ENSEMBLE_DIMS if dim not in prediction.dims]
    if missing:
        raise ValueError(
            "Cannot decompose uncertainty; prediction is missing ensemble dimensions: "
            + ", ".join(missing)
        )
    for dim in ENSEMBLE_DIMS:
        if prediction.sizes[dim] < 2:
            raise ValueError(f"At least two samples are required along {dim!r}")

    climate_mean = prediction.mean(("model_draw", "observation_draw"))
    model_mean = prediction.mean(("climate_model", "observation_draw"))
    data_mean = prediction.mean(("climate_model", "model_draw"))
    result = xr.Dataset({
        "salinity_mean": prediction.mean(ENSEMBLE_DIMS),
        "climate_model_spread_sd": climate_mean.std("climate_model", ddof=1),
        "model_uncertainty_sd": model_mean.std("model_draw", ddof=1),
        "data_uncertainty_sd": data_mean.std("observation_draw", ddof=1),
    })

    required_layers = (
        "applicability_mask",
        "covariate_shift_score",
        "applicability_probability",
        "applicability_uncertainty",
    )
    applicability = {}
    for required in required_layers:
        if required not in predictions:
            raise ValueError(f"Prediction dataset must include {required!r}")
        applicability[required] = predictions[required]

    if "climate_model" in applicability["applicability_mask"].dims:
        applicability["applicability_mask"] = applicability["applicability_mask"].min("climate_model")
        applicability["covariate_shift_score"] = applicability["covariate_shift_score"].max("climate_model")
        applicability["applicability_probability"] = applicability[
            "applicability_probability"
        ].mean("climate_model")
        probability = applicability["applicability_probability"]
        applicability["applicability_uncertainty"] = (probability * (1 - probability)) ** 0.5
    result.update(applicability)
    result.attrs.update({
        "description": "Spatial future groundwater salinity and separate uncertainty components",
        "uncertainty_components": "climate-model spread, model ensemble, observation/data ensemble",
        "uncertainty_method": "sample standard deviation, one independent ensemble axis at a time",
        "applicability_aggregation": "minimum mask, maximum shift score, mean probability across climate models",
        "applicability_required": "mask and covariate shift score carried from feature diagnostics",
        "prediction_variable": prediction_variable,
    })
    return result


def calculate_uncertainty(input_file: Path, output_file: Path) -> Path:
    with xr.open_dataset(input_file) as predictions:
        result = quantify_prediction_uncertainty(predictions)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose future salinity uncertainty")
    parser.add_argument("input", type=Path, help="Prediction ensemble NetCDF")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "future_uncertainty.nc")
    args = parser.parse_args()
    print(calculate_uncertainty(args.input, args.output))


if __name__ == "__main__":
    main()
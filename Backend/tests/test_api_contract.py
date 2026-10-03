import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException
from pydantic import ValidationError

from Backend import main as api
from Backend.src.demo_data import generate_demo


def test_scientific_routes_report_pending_and_validate_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PRODUCT_DIR", tmp_path)

    prediction = api.get_prediction_location(
        latitude=26.85,
        longitude=80.95,
        period="current",
        scenario="current",
    )
    assert prediction.status == "data_pending"
    assert prediction.salinityEcUsCm is None

    projection = api.get_projection_location(
        latitude=26.85,
        longitude=80.95,
        period="2050",
        scenario="ssp585",
    )
    assert projection.status == "data_pending"

    explanation = api.get_prediction_explanation(latitude=26.85, longitude=0.0)
    assert explanation.status == "data_pending"
    assert explanation.features == []

    with pytest.raises(HTTPException) as invalid_pair:
        api.get_projection_location(
            latitude=0,
            longitude=0,
            period="2050",
            scenario="current",
        )
    assert invalid_pair.value.status_code == 422
    with pytest.raises(ValidationError):
        api.AssistantRequest(message="Bad coordinates", latitude=-91, longitude=0)


def test_demo_products_are_never_reported_as_scientific_predictions(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo"
    generate_demo(demo_dir)
    monkeypatch.setattr(api, "PRODUCT_DIR", demo_dir / "products")

    prediction = api.get_prediction_location(
        latitude=26.85,
        longitude=80.95,
        period="current",
        scenario="current",
    )
    assert prediction.status == "demo"
    assert prediction.salinityEcUsCm is None

    layers = {layer.id: layer for layer in api.get_map_layers()}
    assert layers["current-salinity"].available is False
    assert layers["demo-salinity-index"].available is True
    assert layers["demo-salinity-index"].status == "demo"


def test_versioned_production_product_returns_only_present_values(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PRODUCT_DIR", tmp_path)
    xr.Dataset(
        {
            "salinity_mean": (
                ("lat", "lon"),
                np.array([[1200.0]]),
                {"units": "µS/cm"},
            ),
            "level_m": (("lat", "lon"), np.array([[12.5]]), {"units": "m"}),
            "salinity_probability": (("lat", "lon"), np.array([[0.7]]), {"units": "1"}),
            "model_uncertainty_sd": (("lat", "lon"), np.array([[80.0]]), {"units": "µS/cm"}),
        },
        coords={"lat": [26.85], "lon": [80.95]},
        attrs={
            "model_version": "xgboost-v1.0",
            "data_year": 2024,
            "uncertainty_method": "ensemble_spread",
        },
    ).to_netcdf(tmp_path / "current.nc")
    response = api.get_prediction_location(
        latitude=26.85,
        longitude=80.95,
        period="current",
        scenario="current",
    )

    assert response.status == "model_output"
    assert response.salinityEcUsCm == 1200.0
    assert response.levelM == 12.5
    assert response.salinityProbability == 0.7
    assert response.uncertainty.model_dump() == {
        "value": 80.0,
        "unit": "µS/cm",
        "method": "ensemble_spread",
    }
    assert response.modelVersion == "xgboost-v1.0"
    assert response.dataYear == 2024


def test_assistant_returns_safe_pending_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PRODUCT_DIR", tmp_path)

    response = api.assistant_query(
        api.AssistantRequest(
            message="What is the salinity here?",
            latitude=26.85,
            longitude=80.95,
        )
    )

    assert response.model_dump() == {
        "answer": "I don't have a validated prediction for this location yet.",
        "actions": [],
        "status": "data_pending",
    }
    with pytest.raises(HTTPException) as invalid_coordinates:
        api.assistant_query(
            api.AssistantRequest(
                message="Tell me about this place",
                latitude=26.85,
            )
        )
    assert invalid_coordinates.value.status_code == 422

"""
REST API and Endpoint Unit Tests for Simplified EV Range Predictor.
Verifies all 9 Flask routes, HTTP status codes, JSON response shapes,
validation boundaries, and error contracts.
"""
import pytest
import json


def test_api_health(flask_client):
    """Test 1: GET /api/health"""
    res = flask_client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["dataset_available"] is True
    assert data["models_available"] is True


def test_api_summary(flask_client):
    """Test 2: GET /api/summary"""
    res = flask_client.get("/api/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total_vehicles"] == 478
    assert data["total_features"] == 18
    assert data["total_models"] == 3
    assert data["best_model"] == "Linear Regression"
    assert data["best_model_mae"] == 14.37


def test_api_dataset_preview(flask_client):
    """Test 3: GET /api/dataset/preview"""
    res = flask_client.get("/api/dataset/preview?page=1&page_size=10")
    assert res.status_code == 200
    data = res.get_json()
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["data"]) == 10
    assert "Brand" in data["columns"]
    assert "Range (km)" in data["columns"]

    # Search filter
    res_search = flask_client.get("/api/dataset/preview?search=Tesla&page_size=10")
    assert res_search.status_code == 200
    search_data = res_search.get_json()
    assert search_data["total_records"] > 0
    assert all("tesla" in str(r["brand"]).lower() for r in search_data["data"])


def test_api_cleaning(flask_client):
    """Test 4: GET /api/cleaning"""
    res = flask_client.get("/api/cleaning")
    assert res.status_code == 200
    data = res.get_json()
    assert "bullets" in data
    assert len(data["bullets"]) == 6
    assert any("478 cars" in b for b in data["bullets"])


def test_api_insights(flask_client):
    """Test 5: GET /api/insights"""
    res = flask_client.get("/api/insights")
    assert res.status_code == 200
    data = res.get_json()
    assert "distribution" in data
    assert "battery_scatter" in data
    assert "acceleration_scatter" in data
    assert "correlations" in data
    assert len(data["distribution"]["bins"]) > 0
    assert len(data["battery_scatter"]["points"]) > 0
    assert len(data["acceleration_scatter"]["points"]) > 0
    assert len(data["correlations"]["items"]) == 8


def test_api_models_metrics(flask_client):
    """Test 6: GET /api/models/metrics"""
    res = flask_client.get("/api/models/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["models"]) == 3
    model_names = [m["name"] for m in data["models"]]
    assert "Linear Regression" in model_names
    assert "Random Forest" in model_names
    assert "Gradient Boosting" in model_names
    assert data["best_model"] == "Linear Regression"
    assert "actual_vs_predicted" in data
    assert len(data["actual_vs_predicted"]["points"]) > 0


def test_api_predict_schema(flask_client):
    """Test 7: GET /api/predict/schema"""
    res = flask_client.get("/api/predict/schema")
    assert res.status_code == 200
    data = res.get_json()
    assert "main_fields" in data
    assert "optional_fields" in data
    assert len(data["main_fields"]) == 6
    field_names = [f["name"] for f in data["main_fields"]]
    assert "battery_capacity_kWh" in field_names
    assert "top_speed_kmh" in field_names


def test_api_predict_example(flask_client):
    """Test 8: GET /api/predict/example"""
    res = flask_client.get("/api/predict/example")
    assert res.status_code == 200
    data = res.get_json()
    assert "inputs" in data
    assert "actual_range_km" in data
    assert data["actual_range_km"] > 0
    assert "battery_capacity_kWh" in data["inputs"]


def test_api_predict_inference(flask_client):
    """Test 9: POST /api/predict"""
    # Valid prediction with example inputs
    ex_res = flask_client.get("/api/predict/example")
    ex_data = ex_res.get_json()

    res = flask_client.post("/api/predict", json=ex_data["inputs"])
    assert res.status_code == 200
    data = res.get_json()
    assert "predictions" in data
    assert "Linear Regression" in data["predictions"]
    assert "Random Forest" in data["predictions"]
    assert "Gradient Boosting" in data["predictions"]
    assert data["predictions"]["Linear Regression"] > 0
    assert "bars" in data
    assert "best_model" in data

    # Validation: missing battery capacity
    res_no_bat = flask_client.post("/api/predict", json={"top_speed_kmh": 180})
    assert res_no_bat.status_code == 422
    assert "Battery capacity is required" in res_no_bat.get_json()["message"]

    # Validation: negative battery capacity
    res_neg_bat = flask_client.post("/api/predict", json={"battery_capacity_kWh": -50})
    assert res_neg_bat.status_code == 422
    assert "greater than 0" in res_neg_bat.get_json()["message"]

    # Validation: negative numeric field
    res_neg_val = flask_client.post("/api/predict", json={"battery_capacity_kWh": 75, "torque_nm": -100})
    assert res_neg_val.status_code == 422
    assert "cannot be negative" in res_neg_val.get_json()["message"]

    # Validation: non-JSON content type
    res_non_json = flask_client.post("/api/predict", data="not json", content_type="text/plain")
    assert res_non_json.status_code == 415


def test_no_nan_in_json_responses(flask_client):
    """Ensures raw NaN never appears in serialized JSON strings for any endpoint."""
    endpoints = [
        "/api/health",
        "/api/summary",
        "/api/dataset/preview",
        "/api/cleaning",
        "/api/insights",
        "/api/models/metrics",
        "/api/predict/schema",
        "/api/predict/example"
    ]
    for ep in endpoints:
        res = flask_client.get(ep)
        assert res.status_code == 200
        text = res.get_data(as_text=True)
        assert ": NaN" not in text
        assert ": NaN," not in text

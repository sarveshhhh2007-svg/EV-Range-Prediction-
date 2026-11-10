"""
REST API and Endpoint Unit Tests.
Verifies all Flask routes, HTTP status codes, JSON response integrity,
validation boundaries, and error contract shapes.
"""
import pytest
import json


def test_api_health(flask_client):
    """Test /api/health endpoint."""
    res = flask_client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["dataset_available"] is True
    assert data["models_available"] is True


def test_api_summary(flask_client):
    """Test /api/summary endpoint."""
    res = flask_client.get("/api/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total_records"] == 478
    assert data["total_columns"] == 22
    assert data["target_variable"] == "range_km"
    assert data["models_trained"] is True


def test_api_dataset_preview(flask_client):
    """Test /api/dataset/preview endpoint pagination and search."""
    res = flask_client.get("/api/dataset/preview?page=1&page_size=10&view=processed")
    assert res.status_code == 200
    data = res.get_json()
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["data"]) == 10
    assert "range_km" in data["columns"]

    # Search filter
    res_search = flask_client.get("/api/dataset/preview?search=Tesla&page_size=10")
    assert res_search.status_code == 200
    search_data = res_search.get_json()
    assert search_data["total_records"] > 0
    assert all("tesla" in str(r["brand"]).lower() for r in search_data["data"])


def test_api_preprocessing(flask_client):
    """Test /api/preprocessing endpoint."""
    res = flask_client.get("/api/preprocessing")
    assert res.status_code == 200
    data = res.get_json()
    assert "cleaning_log" in data
    assert "leakage_audit" in data
    assert "outliers_detected" in data


def test_api_eda_endpoints(flask_client):
    """Test EDA endpoints: stats, correlation, distribution, scatter."""
    # Stats
    res_stats = flask_client.get("/api/eda")
    assert res_stats.status_code == 200
    assert "descriptive_statistics" in res_stats.get_json()

    # Correlation
    res_corr = flask_client.get("/api/eda/correlation")
    assert res_corr.status_code == 200
    assert "target_correlations" in res_corr.get_json()

    # Distribution
    res_dist = flask_client.get("/api/eda/distribution")
    assert res_dist.status_code == 200
    assert len(res_dist.get_json()["histogram"]) > 0

    # Whitelisted Scatter
    res_scat = flask_client.get("/api/eda/scatter?feature=battery_capacity_kWh")
    assert res_scat.status_code == 200
    assert len(res_scat.get_json()["points"]) > 0

    # Invalid Scatter feature
    res_scat_inv = flask_client.get("/api/eda/scatter?feature=invalid_feature_name")
    assert res_scat_inv.status_code == 400


def test_api_models_metrics(flask_client):
    """Test /api/models/metrics endpoint."""
    res = flask_client.get("/api/models/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "Linear Regression" in data["models"]
    assert "Random Forest" in data["models"]
    assert "Gradient Boosting" in data["models"]
    assert data["recommended_model"] is not None


def test_api_predict_flow(flask_client):
    """Test prediction schema, test example loader, and prediction inference."""
    # Schema
    res_schema = flask_client.get("/api/predict/schema")
    assert res_schema.status_code == 200
    assert len(res_schema.get_json()["groups"]) >= 4

    # Example
    res_ex = flask_client.get("/api/predict/example")
    assert res_ex.status_code == 200
    ex_data = res_ex.get_json()
    assert "actual_range_km" in ex_data
    assert "inputs" in ex_data

    # Valid POST prediction
    res_pred = flask_client.post("/api/predict", json=ex_data["inputs"])
    assert res_pred.status_code == 200
    pred_data = res_pred.get_json()
    assert "Linear Regression" in pred_data["predictions"]
    assert "Random Forest" in pred_data["predictions"]
    assert "Gradient Boosting" in pred_data["predictions"]
    assert pred_data["spread_between_models_km"] >= 0

    # Invalid POST: negative battery
    res_neg = flask_client.post("/api/predict", json={"battery_capacity_kWh": -15.0})
    assert res_neg.status_code == 422
    assert "cannot be negative" in res_neg.get_json()["error"]["message"]

    # Invalid POST: missing battery capacity
    res_miss = flask_client.post("/api/predict", json={"top_speed_kmh": 180})
    assert res_miss.status_code == 422
    assert "Missing required field" in res_miss.get_json()["error"]["message"]

    # Non-JSON content type
    res_bad_ct = flask_client.post("/api/predict", data="hello world")
    assert res_bad_ct.status_code in [415, 422]


def test_no_nan_in_json_responses(flask_client):
    """Ensures raw NaN never appears in serialized JSON strings."""
    for ep in ["/api/summary", "/api/dataset/preview", "/api/preprocessing", "/api/eda", "/api/models/metrics"]:
        res = flask_client.get(ep)
        text = res.get_data(as_text=True)
        assert ": NaN" not in text
        assert ": NaN," not in text
        assert ": Infinity" not in text


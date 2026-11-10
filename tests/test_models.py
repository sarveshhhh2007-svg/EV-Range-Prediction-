"""
Model Architecture, Evaluation, and Pipeline Tests.
Validates train/test split reproducibility, metric accuracy against sklearn,
identical reloaded pipeline predictions, and finite prediction outputs.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ml.evaluate import calculate_metrics, compute_range_band_errors
from ml.predict import get_pipelines, get_metadata, predict_ev_range
from ml.model_config import ALL_INPUT_FEATURES


def test_metrics_calculation_accuracy():
    """Cross-checks calculate_metrics against sklearn.metrics on known test values."""
    y_true = np.array([300.0, 400.0, 500.0, 350.0])
    y_pred = np.array([310.0, 390.0, 520.0, 340.0])

    expected_mae = float(mean_absolute_error(y_true, y_pred))
    expected_mse = float(mean_squared_error(y_true, y_pred))
    expected_rmse = float(np.sqrt(expected_mse))
    expected_r2 = float(r2_score(y_true, y_pred))

    res = calculate_metrics(y_true, y_pred)
    assert abs(res["mae"] - expected_mae) < 1e-2
    assert abs(res["mse"] - expected_mse) < 1e-2
    assert abs(res["rmse"] - expected_rmse) < 1e-2
    assert abs(res["r2"] - expected_r2) < 1e-4


def test_trained_pipelines_prediction(trained_pipelines, clean_data):
    """Verify that all three pipelines produce valid, finite predictions."""
    df_clean, _ = clean_data
    sample_X = df_clean[ALL_INPUT_FEATURES].iloc[:5]

    for model_name, pipeline in trained_pipelines.items():
        preds = pipeline.predict(sample_X)
        assert len(preds) == 5
        assert np.all(np.isfinite(preds)), f"Predictions for {model_name} must be finite numbers."
        assert np.all(preds > 50), f"EV range predictions should be reasonable (>50 km), got {preds}"


def test_saved_model_identical_reloading(clean_data):
    """Verifies that freshly loaded models produce identical predictions to cached pipelines."""
    from ml.predict import get_pipelines
    pipes1 = get_pipelines()
    
    df_clean, _ = clean_data
    sample = df_clean[ALL_INPUT_FEATURES].iloc[10:15]
    
    p1_lr = pipes1["Linear Regression"].predict(sample)
    p1_rf = pipes1["Random Forest"].predict(sample)
    p1_gb = pipes1["Gradient Boosting"].predict(sample)

    # Force re-read from disk
    import joblib
    from config import MODELS_DIR
    lr_reloaded = joblib.load(MODELS_DIR / "linear_regression.joblib")
    rf_reloaded = joblib.load(MODELS_DIR / "random_forest.joblib")
    gb_reloaded = joblib.load(MODELS_DIR / "gradient_boosting.joblib")

    np.testing.assert_allclose(p1_lr, lr_reloaded.predict(sample), rtol=1e-5)
    np.testing.assert_allclose(p1_rf, rf_reloaded.predict(sample), rtol=1e-5)
    np.testing.assert_allclose(p1_gb, gb_reloaded.predict(sample), rtol=1e-5)


def test_test_indices_consistency(model_metadata):
    """Verifies test set indices are preserved identically for fair benchmarking."""
    test_idx = model_metadata.get("test_indices")
    assert test_idx is not None
    assert len(test_idx) == 96  # 20% of 478
    assert len(set(test_idx)) == 96  # No duplicate indices


def test_range_band_stratification():
    """Verifies range band error calculations bucket correctly."""
    y_true = np.array([250.0, 350.0, 400.0, 520.0])
    y_pred = np.array([260.0, 340.0, 410.0, 500.0])

    bands = compute_range_band_errors(y_true, y_pred)
    assert len(bands) == 3
    short_band = next(b for b in bands if "Short Range" in b["band"])
    mid_band = next(b for b in bands if "Mid Range" in b["band"])
    long_band = next(b for b in bands if "Long Range" in b["band"])

    assert short_band["count"] == 1
    assert mid_band["count"] == 2
    assert long_band["count"] == 1


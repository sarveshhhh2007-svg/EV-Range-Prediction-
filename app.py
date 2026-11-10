"""
Flask Application and REST API.
Electric Vehicle Range Prediction & ML Analytics Platform.
Provides RESTful endpoints for data exploration, preprocessing audits,
interactive EDA, model evaluation, background retraining, and live range inference.
"""
import sys
import os
import json
import time
import math
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, render_template, Response

import config
from ml.model_config import (
    DATASET_FILENAME,
    EXPECTED_COLUMNS,
    EXPECTED_STATS,
    ALL_INPUT_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_UNITS
)
from ml.data_loader import load_raw_dataset, audit_raw_dataset, compute_file_hash, locate_dataset_file
from ml.preprocessing import clean_dataset, audit_target_leakage, detect_outliers_iqr
from ml.eda import (
    compute_descriptive_statistics,
    compute_correlation_matrix,
    compute_distribution_data,
    get_scatter_data
)
from ml.predict import (
    get_prediction_schema,
    get_test_set_example,
    predict_ev_range,
    get_metadata,
    invalidate_model_cache
)
from ml.train import run_training_pipeline

# Configure application logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EVRangeApp")

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATES_DIR),
    static_folder=str(config.STATIC_DIR)
)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Background Training State & Locks
training_lock = threading.Lock()
is_training_active = False
training_status_state = {
    "status": "idle",       # "idle", "running", "completed", "failed"
    "progress_message": "No training in progress",
    "started_at": None,
    "completed_at": None,
    "error": None
}
LAST_TRAIN_COMPLETION_TIME = 0
TRAIN_COOLDOWN_SECONDS = 10


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively converts float NaN, Infinity, -Infinity to None (null in JSON),
    and numpy scalars to Python native types, ensuring 100% compliant RFC 8259 JSON.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj) or np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif pd.isna(obj):
        return None
    return obj


def error_response(code: str, message: str, status_code: int) -> Response:
    """Standardized API error response format."""
    return jsonify({"error": {"code": code, "message": message}}), status_code


# -------------------------------------------------------------
# Frontend Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    """Serves the single-page application dashboard."""
    return render_template("index.html")


# -------------------------------------------------------------
# Core & Health Endpoints
# -------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def api_health():
    """Returns backend status, dataset availability, models availability, and version checks."""
    dataset_available = False
    dataset_hash = None
    try:
        data_path = locate_dataset_file()
        dataset_available = data_path.exists()
        if dataset_available:
            dataset_hash = compute_file_hash(data_path)
    except Exception:
        dataset_available = False

    models_available = False
    metadata_warning = None
    try:
        meta = get_metadata()
        models_available = True
        # Check if hash matches
        trained_hash = meta.get("dataset_hash_sha256")
        if dataset_hash and trained_hash and dataset_hash != trained_hash:
            metadata_warning = "Dataset on disk differs from the dataset used during model training."
    except Exception:
        models_available = False

    payload = {
        "status": "healthy",
        "dataset_available": dataset_available,
        "models_available": models_available,
        "dataset_filename": DATASET_FILENAME,
        "dataset_hash": dataset_hash,
        "metadata_warning": metadata_warning,
        "python_version": sys.version.split()[0],
        "is_training_active": is_training_active
    }
    return jsonify(sanitize_for_json(payload)), 200


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """Provides high-level dataset statistics for hero and summary counters."""
    try:
        df_raw, audit_info = load_raw_dataset()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return error_response("DATASET_NOT_FOUND", "Could not load dataset file.", 500)

    rows, cols = df_raw.shape
    exact_duplicates = int(df_raw.duplicated().sum())
    missing_count = int(df_raw.isnull().sum().sum())

    meta = None
    try:
        meta = get_metadata()
    except Exception:
        pass

    summary = {
        "dataset_filename": DATASET_FILENAME,
        "total_records": rows,
        "total_columns": cols,
        "numerical_features_count": len(ALL_NUMERICAL_FEATURES),
        "categorical_features_count": len(CATEGORICAL_FEATURES),
        "total_missing_values": missing_count,
        "exact_duplicates": exact_duplicates,
        "target_variable": TARGET_COLUMN,
        "models_trained": meta is not None,
        "recommended_model": meta.get("recommended_model") if meta else None,
        "trained_at": meta.get("trained_at") if meta else None,
        "source": "https://www.kaggle.com/datasets/afnansaifafnan/electric-car-performance-and-battery-dataset"
    }
    return jsonify(sanitize_for_json(summary)), 200


# -------------------------------------------------------------
# Dataset Explorer Endpoints
# -------------------------------------------------------------
@app.route("/api/dataset/preview", methods=["GET"])
def api_dataset_preview():
    """
    Returns server-side paginated, searchable, sortable view of either
    the raw ('original') or preprocessed ('processed') dataset.
    """
    view_type = request.args.get("view", "processed").lower()
    page = request.args.get("page", default=1, type=int)
    page_size = request.args.get("page_size", default=10, type=int)
    search = request.args.get("search", default="", type=str).strip().lower()
    sort_col = request.args.get("sort", default="", type=str).strip()

    # Clamp page_size
    if page_size not in [10, 25, 50]:
        page_size = 10
    if page < 1:
        page = 1

    try:
        df_raw, _ = load_raw_dataset()
        if view_type == "processed":
            df_display, _ = clean_dataset(df_raw)
        else:
            df_display = df_raw.copy()
    except Exception as e:
        logger.error(f"Error loading dataset preview: {e}")
        return error_response("DATA_ERROR", str(e), 500)

    # Search filter across string fields (brand, model, segment, drivetrain, car_body_type)
    if search:
        searchable_cols = [c for c in ["brand", "model", "drivetrain", "segment", "car_body_type", "battery_type"] if c in df_display.columns]
        mask = pd.Series(False, index=df_display.index)
        for col in searchable_cols:
            mask |= df_display[col].astype(str).str.lower().str.contains(search, na=False)
        df_display = df_display[mask]

    # Sorting
    if sort_col:
        ascending = True
        if sort_col.startswith("-"):
            ascending = False
            sort_col = sort_col[1:]
        if sort_col in df_display.columns:
            df_display = df_display.sort_values(by=sort_col, ascending=ascending)

    total_records = len(df_display)
    total_pages = max(1, math.ceil(total_records / page_size))
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_data = df_display.iloc[start_idx:end_idx]

    # Convert records to dictionary with None for NaN
    records = page_data.to_dict(orient="records")

    payload = {
        "view": view_type,
        "page": page,
        "page_size": page_size,
        "total_records": total_records,
        "total_pages": total_pages,
        "columns": list(df_display.columns),
        "data": records
    }
    return jsonify(sanitize_for_json(payload)), 200


@app.route("/api/dataset/schema", methods=["GET"])
def api_dataset_schema():
    """Provides metadata and column descriptions for all 22 dataset columns."""
    try:
        df_raw, _ = load_raw_dataset()
    except Exception as e:
        return error_response("DATA_ERROR", str(e), 500)

    columns_info = []
    for col in EXPECTED_COLUMNS:
        series = df_raw[col]
        missing_count = int(series.isnull().sum())
        missing_pct = round(missing_count / len(df_raw) * 100, 2)
        unique_count = int(series.nunique())
        is_num = pd.api.types.is_numeric_dtype(series)

        columns_info.append({
            "name": col,
            "label": col.replace("_", " ").title(),
            "dtype": str(series.dtype),
            "is_numeric": is_num,
            "is_target": col == TARGET_COLUMN,
            "is_excluded": col in ["source_url", "model", "battery_type", TARGET_COLUMN],
            "unit": FEATURE_UNITS.get(col, ""),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_count": unique_count
        })

    return jsonify(sanitize_for_json({"columns": columns_info})), 200


@app.route("/api/dataset/download", methods=["GET"])
def api_dataset_download():
    """Serves the cleaned EV dataset as a downloadable CSV."""
    cleaned_file = config.DATA_DIR / "cleaned_electric_vehicles.csv"
    if not cleaned_file.exists():
        # Clean dynamically if not yet written
        df_raw, _ = load_raw_dataset()
        df_clean, _ = clean_dataset(df_raw)
        df_clean.to_csv(cleaned_file, index=False)
        
    return send_file(
        cleaned_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="cleaned_electric_vehicles_spec_2025.csv"
    )


# -------------------------------------------------------------
# Data Preprocessing & Audit Endpoints
# -------------------------------------------------------------
@app.route("/api/preprocessing", methods=["GET"])
def api_preprocessing():
    """Returns data cleaning log, IQR outlier table, and target leakage audit findings."""
    report_path = config.REPORTS_DIR / "preprocessing_summary.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(sanitize_for_json(data)), 200

    # Fallback to computing dynamically
    try:
        df_raw, _ = load_raw_dataset()
        df_clean, cleaning_log = clean_dataset(df_raw)
        leakage = audit_target_leakage(df_clean)
        outliers = detect_outliers_iqr(df_clean)
        payload = {
            "cleaning_log": cleaning_log,
            "outliers_detected": outliers,
            "leakage_audit": leakage
        }
        return jsonify(sanitize_for_json(payload)), 200
    except Exception as e:
        logger.error(f"Preprocessing audit error: {e}")
        return error_response("AUDIT_ERROR", str(e), 500)


# -------------------------------------------------------------
# Exploratory Data Analysis (EDA) Endpoints
# -------------------------------------------------------------
@app.route("/api/eda", methods=["GET"])
def api_eda_statistics():
    """Provides numerical descriptive statistics, quartiles, and covariances with range."""
    try:
        df_raw, _ = load_raw_dataset()
        df_clean, _ = clean_dataset(df_raw)
        stats = compute_descriptive_statistics(df_clean)
        return jsonify(sanitize_for_json({"descriptive_statistics": stats})), 200
    except Exception as e:
        return error_response("EDA_ERROR", str(e), 500)


@app.route("/api/eda/correlation", methods=["GET"])
def api_eda_correlation():
    """Returns numerical correlation matrix and ordered correlations with range_km."""
    try:
        df_raw, _ = load_raw_dataset()
        df_clean, _ = clean_dataset(df_raw)
        corrs = compute_correlation_matrix(df_clean)
        return jsonify(sanitize_for_json(corrs)), 200
    except Exception as e:
        return error_response("CORRELATION_ERROR", str(e), 500)


@app.route("/api/eda/distribution", methods=["GET"])
def api_eda_distribution():
    """Returns histogram bin frequencies and boxplot percentiles for range_km."""
    try:
        df_raw, _ = load_raw_dataset()
        df_clean, _ = clean_dataset(df_raw)
        dist = compute_distribution_data(df_clean)
        return jsonify(sanitize_for_json(dist)), 200
    except Exception as e:
        return error_response("DISTRIBUTION_ERROR", str(e), 500)


@app.route("/api/eda/scatter", methods=["GET"])
def api_eda_scatter():
    """Returns scatter points for a chosen whitelisted numerical feature vs range_km."""
    feature = request.args.get("feature", "battery_capacity_kWh").strip()
    if feature not in ALL_NUMERICAL_FEATURES:
        return error_response(
            "INVALID_FEATURE",
            f"Feature '{feature}' is not in allowed numerical feature list: {ALL_NUMERICAL_FEATURES}",
            400
        )
    try:
        df_raw, _ = load_raw_dataset()
        df_clean, _ = clean_dataset(df_raw)
        scatter_data = get_scatter_data(df_clean, feature)
        return jsonify(sanitize_for_json(scatter_data)), 200
    except Exception as e:
        return error_response("SCATTER_ERROR", str(e), 500)


# -------------------------------------------------------------
# Model Laboratory & Retraining Endpoints
# -------------------------------------------------------------
@app.route("/api/models/status", methods=["GET"])
def api_models_status():
    """Returns model training status and metadata."""
    try:
        meta = get_metadata()
        return jsonify(sanitize_for_json({
            "is_trained": True,
            "trained_at": meta.get("trained_at"),
            "dataset_hash": meta.get("dataset_hash_sha256"),
            "library_versions": meta.get("library_versions"),
            "recommended_model": meta.get("recommended_model"),
            "models_available": list(meta.get("models", {}).keys())
        })), 200
    except FileNotFoundError:
        return jsonify({"is_trained": False, "message": "Models not yet trained"}), 200
    except Exception as e:
        return error_response("STATUS_ERROR", str(e), 500)


@app.route("/api/models/metrics", methods=["GET"])
def api_models_metrics():
    """Returns evaluation metrics, cross-validation, ablations, and baseline."""
    try:
        meta = get_metadata()
        payload = {
            "models": meta.get("models", {}),
            "metric_winners": meta.get("metric_winners", {}),
            "recommended_model": meta.get("recommended_model"),
            "recommendation_justification": meta.get("recommendation_justification"),
            "naive_baseline": meta.get("naive_baseline", {}),
            "ablations": meta.get("ablations", {}),
            "leakage_audit": meta.get("leakage_audit", {})
        }
        return jsonify(sanitize_for_json(payload)), 200
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Models have not been trained yet. Please train models first.", 503)
    except Exception as e:
        return error_response("METRICS_ERROR", str(e), 500)


@app.route("/api/models/importance", methods=["GET"])
def api_models_importance():
    """Returns test-set permutation importance and model-native importances."""
    try:
        meta = get_metadata()
        importances = meta.get("feature_importances", {})
        return jsonify(sanitize_for_json(importances)), 200
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Models not trained yet.", 503)
    except Exception as e:
        return error_response("IMPORTANCE_ERROR", str(e), 500)


def _background_training_worker():
    """Worker function executing model retraining in background thread."""
    global is_training_active, training_status_state, LAST_TRAIN_COMPLETION_TIME
    try:
        training_status_state["status"] = "running"
        training_status_state["started_at"] = time.time()
        training_status_state["error"] = None

        def progress_cb(msg):
            training_status_state["progress_message"] = msg

        run_training_pipeline(progress_callback=progress_cb)
        invalidate_model_cache()

        training_status_state["status"] = "completed"
        training_status_state["completed_at"] = time.time()
        training_status_state["progress_message"] = "Retraining completed successfully!"
        LAST_TRAIN_COMPLETION_TIME = time.time()
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        training_status_state["status"] = "failed"
        training_status_state["error"] = "Training pipeline failed. Please check server logs."
        training_status_state["progress_message"] = f"Training failed: {e}"
    finally:
        is_training_active = False


@app.route("/api/models/train", methods=["POST"])
def api_models_train():
    """
    Triggers model training in background thread.
    Uses locking to prevent concurrent runs; returns 409 if already active.
    Enforces cooldown to prevent rapid accidental retries.
    """
    global is_training_active, LAST_TRAIN_COMPLETION_TIME
    
    # Check cooldown
    if time.time() - LAST_TRAIN_COMPLETION_TIME < TRAIN_COOLDOWN_SECONDS:
        remaining = int(TRAIN_COOLDOWN_SECONDS - (time.time() - LAST_TRAIN_COMPLETION_TIME))
        return error_response("COOLDOWN_ACTIVE", f"Please wait {remaining}s before retraining again.", 429)

    with training_lock:
        if is_training_active:
            return error_response("TRAINING_IN_PROGRESS", "A model training task is already active.", 409)
        is_training_active = True

    thread = threading.Thread(target=_background_training_worker, daemon=True)
    thread.start()

    return jsonify({
        "status": "accepted",
        "message": "Model training started in background thread.",
        "poll_endpoint": "/api/models/train/status"
    }), 202


@app.route("/api/models/train/status", methods=["GET"])
def api_models_train_status():
    """Pollable endpoint for training progress updates."""
    return jsonify({
        "is_training_active": is_training_active,
        "state": training_status_state
    }), 200


# -------------------------------------------------------------
# Prediction Studio Endpoints
# -------------------------------------------------------------
@app.route("/api/predict/schema", methods=["GET"])
def api_predict_schema():
    """Returns dynamic input form schema grouped logically with validation rules."""
    try:
        schema = get_prediction_schema()
        return jsonify(sanitize_for_json(schema)), 200
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Model metadata not found. Run training first.", 503)
    except Exception as e:
        return error_response("SCHEMA_ERROR", str(e), 500)


@app.route("/api/predict/example", methods=["GET"])
def api_predict_example():
    """Provides a genuine test-set EV record with true actual range for comparison."""
    idx = request.args.get("index", default=None, type=int)
    try:
        example = get_test_set_example(idx)
        return jsonify(sanitize_for_json(example)), 200
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Models not trained yet.", 503)
    except Exception as e:
        return error_response("EXAMPLE_ERROR", str(e), 500)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts vehicle specification parameters, validates inputs on server,
    and returns driving range predictions from all three trained pipelines.
    """
    if not request.is_json:
        return error_response("UNSUPPORTED_MEDIA_TYPE", "Request Content-Type must be 'application/json'.", 415)

    try:
        data = request.get_json(force=True)
    except Exception:
        return error_response("INVALID_JSON", "Malformed JSON payload.", 400)

    try:
        result = predict_ev_range(data)
        return jsonify(sanitize_for_json(result)), 200
    except ValueError as ve:
        return error_response("VALIDATION_ERROR", str(ve), 422)
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Trained models not found on server.", 503)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return error_response("PREDICTION_ERROR", "Internal server error during prediction.", 500)


# -------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------
@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return error_response("NOT_FOUND", f"Endpoint '{request.path}' not found.", 404)
    return render_template("index.html"), 404


@app.errorhandler(405)
def handle_405(e):
    return error_response("METHOD_NOT_ALLOWED", "Method not allowed on this endpoint.", 405)


@app.errorhandler(500)
def handle_500(e):
    logger.error(f"Internal 500: {e}")
    return error_response("INTERNAL_ERROR", "An unexpected server error occurred.", 500)


# Cache control headers
@app.after_request
def set_cache_headers(response):
    if request.path.startswith("/api/predict") or request.path.startswith("/api/models/train"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    elif request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


if __name__ == "__main__":
    logger.info(f"Starting EV Range Prediction server at http://{config.HOST}:{config.PORT}")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )

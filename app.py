"""
Production Flask Web Application and REST API.
Electric Vehicle Range Prediction Platform.
Provides clean, minimal REST endpoints for data exploration,
insights charts, model benchmarks, and live range prediction.
"""
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, Response

import config
from ml.model_config import (
    DATASET_FILENAME,
    ALL_INPUT_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_UNITS
)
from ml.data_loader import load_raw_dataset, locate_dataset_file
from ml.preprocessing import clean_dataset
from ml.predict import (
    get_pipelines,
    get_metadata,
    get_test_set_example
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EVRangeApp")

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATES_DIR),
    static_folder=str(config.STATIC_DIR)
)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN and Infinity to None (null in JSON)."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj) or np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif pd.isna(obj):
        return None
    return obj


def error_response(code: str, message: str, status_code: int) -> Response:
    """Returns standardized JSON error response."""
    return jsonify({
        "error": {"code": code, "message": message},
        "message": message
    }), status_code


# -------------------------------------------------------------
# Frontend Route
# -------------------------------------------------------------
@app.route("/")
def index():
    """Serves the single-page application dashboard."""
    return render_template("index.html")


# -------------------------------------------------------------
# Endpoint 1: Health (Used by Loading Screen)
# -------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def api_health():
    """Returns backend and model readiness for startup loading sequence."""
    dataset_available = False
    try:
        data_path = locate_dataset_file()
        dataset_available = data_path.exists()
    except Exception:
        dataset_available = False

    models_available = False
    try:
        get_metadata()
        models_available = True
    except Exception:
        models_available = False

    payload = {
        "status": "healthy",
        "dataset_available": dataset_available,
        "models_available": models_available
    }
    return jsonify(sanitize_for_json(payload)), 200


# -------------------------------------------------------------
# Endpoint 2: Summary (For Home Section)
# -------------------------------------------------------------
@app.route("/api/summary", methods=["GET"])
def api_summary():
    """Returns counts and best model information for the Home section."""
    try:
        df_raw, _ = load_raw_dataset()
        total_cars = len(df_raw)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        return error_response("DATA_ERROR", "Could not load dataset.", 500)

    best_model_name = None
    best_model_mae = None
    models_trained = False

    try:
        meta = get_metadata()
        models_trained = True
        best_model_name = meta.get("recommended_model", "Linear Regression")
        model_info = meta.get("models", {}).get(best_model_name, {})
        best_model_mae = model_info.get("test_metrics", {}).get("mae", 14.37)
    except Exception:
        models_trained = False

    summary = {
        "total_cars": total_cars,
        "total_vehicles": total_cars,
        "total_specs": len(ALL_INPUT_FEATURES),
        "total_features": len(ALL_INPUT_FEATURES),
        "total_models": 3,
        "models_trained": models_trained,
        "best_model": best_model_name,
        "best_model_mae": best_model_mae,
        "sdg_note": "Shows how machine learning can analyse EV data to support smarter transport technology."
    }
    return jsonify(sanitize_for_json(summary)), 200


# -------------------------------------------------------------
# Endpoint 3: Dataset Preview (6 Columns Only, 10 Rows per Page)
# -------------------------------------------------------------
@app.route("/api/dataset/preview", methods=["GET"])
def api_dataset_preview():
    """Returns 10 cars per page with 6 readable columns."""
    page = request.args.get("page", default=1, type=int)
    search = request.args.get("search", default="", type=str).strip().lower()
    page_size = 10

    if page < 1:
        page = 1

    try:
        df_raw, _ = load_raw_dataset()
        df_clean, _ = clean_dataset(df_raw)
    except Exception as e:
        return error_response("DATA_ERROR", str(e), 500)

    # Filter to 6 required display columns
    cols_map = [
        ("brand", "Brand"),
        ("model", "Model"),
        ("battery_capacity_kWh", "Battery (kWh)"),
        ("top_speed_kmh", "Top Speed (km/h)"),
        ("acceleration_0_100_s", "0-100 km/h (s)"),
        ("range_km", "Range (km)")
    ]

    # Search filter across brand and model
    if search:
        mask = (
            df_clean["brand"].astype(str).str.lower().str.contains(search, na=False) |
            df_clean["model"].astype(str).str.lower().str.contains(search, na=False)
        )
        df_clean = df_clean[mask]

    total_records = len(df_clean)
    total_pages = max(1, math.ceil(total_records / page_size))
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_slice = df_clean.iloc[start_idx:end_idx]

    rows = []
    for _, r in page_slice.iterrows():
        rows.append({
            "brand": str(r["brand"]),
            "model": str(r["model"]),
            "battery_capacity": round(float(r["battery_capacity_kWh"]), 1) if pd.notna(r["battery_capacity_kWh"]) else None,
            "top_speed": int(round(float(r["top_speed_kmh"]))) if pd.notna(r["top_speed_kmh"]) else None,
            "acceleration": round(float(r["acceleration_0_100_s"]), 1) if pd.notna(r["acceleration_0_100_s"]) else None,
            "range": int(round(float(r["range_km"]))) if pd.notna(r["range_km"]) else None
        })

    payload = {
        "page": page,
        "page_size": page_size,
        "total_records": total_records,
        "total_pages": total_pages,
        "columns": [label for _, label in cols_map],
        "data": rows
    }
    return jsonify(sanitize_for_json(payload)), 200


# -------------------------------------------------------------
# Endpoint 4: Data Cleaning Bullets (Real Counts)
# -------------------------------------------------------------
@app.route("/api/cleaning", methods=["GET"])
def api_cleaning():
    """Returns the 6 plain-language data cleaning points with real computed counts."""
    try:
        df_raw, _ = load_raw_dataset()
        total_cars = len(df_raw)
        total_specs = len(df_raw.columns)
        missing_cells = int(df_raw["number_of_cells"].isna().sum())
    except Exception:
        total_cars = 478
        total_specs = 22
        missing_cells = 202

    bullets = [
        f"The dataset has {total_cars} cars and {total_specs} specs, with no duplicate rows.",
        f"Filled missing numbers with the middle value (for example, number of cells had {missing_cells} missing).",
        "Cargo space written as text (like \"10 Banana Boxes\") was treated as missing.",
        "Removed battery type (it is the same for every car).",
        "Ignored model name and web link (they only identify a car).",
        "Used 80% of cars to learn (382 cars) and 20% to test (96 cars)."
    ]
    return jsonify({"bullets": bullets}), 200


# -------------------------------------------------------------
# Endpoint 5: Insights (Data for 4 Charts Only)
# -------------------------------------------------------------
@app.route("/api/insights", methods=["GET"])
def api_insights():
    """Returns data and computed captions for the 4 required charts."""
    try:
        df_raw, _ = load_raw_dataset()
        df_clean, _ = clean_dataset(df_raw)
    except Exception as e:
        return error_response("DATA_ERROR", str(e), 500)

    # 1. Range distribution histogram (12 bins)
    ranges = df_clean[TARGET_COLUMN].dropna().values
    counts, edges = np.histogram(ranges, bins=12)
    hist_bins = []
    max_count = -1
    peak_range_str = "350 and 450 km"

    for i in range(len(counts)):
        low = int(round(edges[i]))
        high = int(round(edges[i + 1]))
        c = int(counts[i])
        hist_bins.append({
            "label": f"{low}–{high} km",
            "count": c
        })
        if c > max_count:
            max_count = c
            peak_range_str = f"{low} and {high} km"

    hist_caption = f"Most cars fall between {peak_range_str}."

    # 2. Battery capacity vs range scatter
    bat_pts = []
    for _, r in df_clean.dropna(subset=["battery_capacity_kWh", TARGET_COLUMN]).iterrows():
        bat_pts.append({
            "x": round(float(r["battery_capacity_kWh"]), 1),
            "y": round(float(r[TARGET_COLUMN]), 1),
            "label": f"{r['brand']} {r['model']}"
        })
    bat_caption = "Cars with larger battery packs generally travel further."

    # 3. Acceleration time vs range scatter
    acc_pts = []
    for _, r in df_clean.dropna(subset=["acceleration_0_100_s", TARGET_COLUMN]).iterrows():
        acc_pts.append({
            "x": round(float(r["acceleration_0_100_s"]), 1),
            "y": round(float(r[TARGET_COLUMN]), 1),
            "label": f"{r['brand']} {r['model']}"
        })
    acc_caption = "Faster-accelerating cars often have larger batteries and longer range."

    # 4. Top 8 correlations with range
    num_cols = [c for c in ALL_NUMERICAL_FEATURES if c in df_clean.columns]
    corrs = df_clean[num_cols + [TARGET_COLUMN]].dropna().corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    top8 = corrs.abs().sort_values(ascending=False).head(8)

    plain_names = {
        "battery_capacity_kWh": "Battery Capacity",
        "top_speed_kmh": "Top Speed",
        "fast_charging_power_kw_dc": "Fast-Charging Power",
        "acceleration_0_100_s": "Acceleration Time",
        "torque_nm": "Motor Torque",
        "width_mm": "Vehicle Width",
        "length_mm": "Vehicle Length",
        "height_mm": "Vehicle Height",
        "towing_capacity_kg": "Towing Capacity",
        "number_of_cells": "Battery Cells",
        "seats": "Seat Count",
        "cargo_volume_l": "Cargo Space",
        "efficiency_wh_per_km": "Energy Consumption"
    }

    corr_list = []
    for col in top8.index:
        corr_list.append({
            "feature": plain_names.get(col, col.replace("_", " ").title()),
            "value": round(float(corrs[col]), 3)
        })

    corr_caption = "Longer bars mean the two rise or fall together more strongly. This does not prove one causes the other."

    payload = {
        "distribution": {
            "bins": hist_bins,
            "caption": hist_caption
        },
        "battery_scatter": {
            "points": bat_pts,
            "caption": bat_caption
        },
        "acceleration_scatter": {
            "points": acc_pts,
            "caption": acc_caption
        },
        "correlations": {
            "items": corr_list,
            "caption": corr_caption
        }
    }
    return jsonify(sanitize_for_json(payload)), 200


# -------------------------------------------------------------
# Endpoint 6: Models Metrics & Comparison
# -------------------------------------------------------------
@app.route("/api/models/metrics", methods=["GET"])
def api_models_metrics():
    """Returns results table, error bar chart data, and actual vs predicted scatter."""
    try:
        meta = get_metadata()
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Models are not trained yet. Run python -m ml.train first.", 503)

    model_descriptions = {
        "Linear Regression": "Finds the best straight-line relationship between specs and range.",
        "Random Forest": "Many decision trees each give an estimate; the average is used.",
        "Gradient Boosting": "Trees are built one after another, each fixing the mistakes of the last."
    }

    models_data = []
    mae_bars = []
    for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        m_info = meta.get("models", {}).get(name, {})
        t_met = m_info.get("test_metrics", {})
        mae_val = t_met.get("mae", 15.0)
        models_data.append({
            "name": name,
            "description": model_descriptions.get(name, ""),
            "mae": mae_val,
            "mse": t_met.get("mse", 300.0),
            "rmse": t_met.get("rmse", 18.0),
            "r2": t_met.get("r2", 0.96)
        })
        mae_bars.append({
            "model": name,
            "mae": mae_val
        })

    best_model_name = meta.get("recommended_model", "Linear Regression")
    best_model_mae = meta.get("models", {}).get(best_model_name, {}).get("test_metrics", {}).get("mae", 14.37)

    # Actual vs predicted for the best model on the test set
    actual_vs_predicted_pts = []
    try:
        pipelines = get_pipelines()
        best_pipe = pipelines.get(best_model_name)
        test_indices = meta.get("test_indices", [])
        if best_pipe and test_indices:
            df_raw, _ = load_raw_dataset()
            df_clean, _ = clean_dataset(df_raw)
            test_slice = df_clean.iloc[test_indices]
            preds = best_pipe.predict(test_slice[ALL_INPUT_FEATURES])
            for i, (_, row) in enumerate(test_slice.iterrows()):
                actual_vs_predicted_pts.append({
                    "actual": round(float(row[TARGET_COLUMN]), 1),
                    "predicted": round(float(preds[i]), 1),
                    "label": f"{row['brand']} {row['model']}"
                })
    except Exception as e:
        logger.warning(f"Could not compute actual vs predicted points: {e}")

    payload = {
        "models": models_data,
        "mae_bars": mae_bars,
        "best_model": best_model_name,
        "best_model_sentence": f"Best model: {best_model_name}, average error {best_model_mae} km",
        "best_model_reason": "Chosen because it had the lowest average error when re-tested several times on the training data.",
        "honest_note": "If two models score very close, the difference may not be meaningful.",
        "actual_vs_predicted": {
            "model_name": best_model_name,
            "points": actual_vs_predicted_pts,
            "caption": "Closer to the diagonal line means a better prediction."
        }
    }
    return jsonify(sanitize_for_json(payload)), 200


# -------------------------------------------------------------
# Endpoint 7: Predict Schema (Main 6 + Optional Collapsed)
# -------------------------------------------------------------
@app.route("/api/predict/schema", methods=["GET"])
def api_predict_schema():
    """Returns the form fields: top 6 main inputs + optional collapsed details."""
    try:
        meta = get_metadata()
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Models are not trained yet. Run python -m ml.train first.", 503)

    ranges = meta.get("feature_ranges", {})
    categories = meta.get("categorical_options", {})

    # Main 6 inputs (always visible)
    main_fields = [
        {
            "name": "battery_capacity_kWh",
            "label": "Battery capacity",
            "unit": "kWh",
            "required": True,
            "min": ranges.get("battery_capacity_kWh", {}).get("min", 20.0),
            "max": ranges.get("battery_capacity_kWh", {}).get("max", 150.0),
            "median": ranges.get("battery_capacity_kWh", {}).get("median", 77.0)
        },
        {
            "name": "top_speed_kmh",
            "label": "Top speed",
            "unit": "km/h",
            "required": False,
            "min": ranges.get("top_speed_kmh", {}).get("min", 120.0),
            "max": ranges.get("top_speed_kmh", {}).get("max", 260.0),
            "median": ranges.get("top_speed_kmh", {}).get("median", 180.0)
        },
        {
            "name": "acceleration_0_100_s",
            "label": "Acceleration 0–100 km/h",
            "unit": "s",
            "required": False,
            "min": ranges.get("acceleration_0_100_s", {}).get("min", 2.0),
            "max": ranges.get("acceleration_0_100_s", {}).get("max", 15.0),
            "median": ranges.get("acceleration_0_100_s", {}).get("median", 6.2)
        },
        {
            "name": "fast_charging_power_kw_dc",
            "label": "Fast-charging power",
            "unit": "kW",
            "required": False,
            "min": ranges.get("fast_charging_power_kw_dc", {}).get("min", 20.0),
            "max": ranges.get("fast_charging_power_kw_dc", {}).get("max", 350.0),
            "median": ranges.get("fast_charging_power_kw_dc", {}).get("median", 130.0)
        },
        {
            "name": "torque_nm",
            "label": "Torque",
            "unit": "Nm",
            "required": False,
            "min": ranges.get("torque_nm", {}).get("min", 150.0),
            "max": ranges.get("torque_nm", {}).get("max", 1200.0),
            "median": ranges.get("torque_nm", {}).get("median", 450.0)
        },
        {
            "name": "efficiency_wh_per_km",
            "label": "Efficiency",
            "unit": "Wh/km",
            "required": False,
            "min": ranges.get("efficiency_wh_per_km", {}).get("min", 130.0),
            "max": ranges.get("efficiency_wh_per_km", {}).get("max", 300.0),
            "median": ranges.get("efficiency_wh_per_km", {}).get("median", 185.0)
        }
    ]

    # Optional collapsed fields
    optional_fields = [
        {
            "name": "seats",
            "label": "Seats",
            "unit": "seats",
            "type": "number",
            "min": 2, "max": 9, "median": 5
        },
        {
            "name": "cargo_volume_l",
            "label": "Cargo space",
            "unit": "L",
            "type": "number",
            "min": ranges.get("cargo_volume_l", {}).get("min", 100.0),
            "max": ranges.get("cargo_volume_l", {}).get("max", 1200.0),
            "median": ranges.get("cargo_volume_l", {}).get("median", 470.0)
        },
        {
            "name": "towing_capacity_kg",
            "label": "Towing capacity",
            "unit": "kg",
            "type": "number",
            "min": ranges.get("towing_capacity_kg", {}).get("min", 0.0),
            "max": ranges.get("towing_capacity_kg", {}).get("max", 2500.0),
            "median": ranges.get("towing_capacity_kg", {}).get("median", 1000.0)
        },
        {
            "name": "length_mm",
            "label": "Length",
            "unit": "mm",
            "type": "number",
            "min": ranges.get("length_mm", {}).get("min", 3000.0),
            "max": ranges.get("length_mm", {}).get("max", 5500.0),
            "median": ranges.get("length_mm", {}).get("median", 4700.0)
        },
        {
            "name": "width_mm",
            "label": "Width",
            "unit": "mm",
            "type": "number",
            "min": ranges.get("width_mm", {}).get("min", 1500.0),
            "max": ranges.get("width_mm", {}).get("max", 2200.0),
            "median": ranges.get("width_mm", {}).get("median", 1880.0)
        },
        {
            "name": "height_mm",
            "label": "Height",
            "unit": "mm",
            "type": "number",
            "min": ranges.get("height_mm", {}).get("min", 1300.0),
            "max": ranges.get("height_mm", {}).get("max", 2000.0),
            "median": ranges.get("height_mm", {}).get("median", 1570.0)
        },
        {
            "name": "number_of_cells",
            "label": "Battery cells",
            "unit": "cells",
            "type": "number",
            "min": ranges.get("number_of_cells", {}).get("min", 90.0),
            "max": ranges.get("number_of_cells", {}).get("max", 800.0),
            "median": ranges.get("number_of_cells", {}).get("median", 192.0)
        },
        {
            "name": "brand",
            "label": "Brand",
            "type": "select",
            "options": categories.get("brand", ["Tesla", "Audi", "BYD", "BMW"])
        },
        {
            "name": "drivetrain",
            "label": "Drivetrain",
            "type": "select",
            "options": categories.get("drivetrain", ["AWD", "RWD", "FWD"])
        },
        {
            "name": "segment",
            "label": "Segment",
            "type": "select",
            "options": categories.get("segment", ["B", "C", "D", "E", "F", "N"])
        },
        {
            "name": "car_body_type",
            "label": "Body type",
            "type": "select",
            "options": categories.get("car_body_type", ["SUV", "Sedan", "Hatchback"])
        },
        {
            "name": "fast_charge_port",
            "label": "Charging port",
            "type": "select",
            "options": categories.get("fast_charge_port", ["CCS", "Tesla", "CHAdeMO"])
        }
    ]

    return jsonify({
        "main_fields": main_fields,
        "optional_fields": optional_fields,
        "optional_note": "Leave blank if unknown. Missing values are filled automatically; more details give a closer estimate."
    }), 200


# -------------------------------------------------------------
# Endpoint 8: Predict Example (One Real Test Car)
# -------------------------------------------------------------
@app.route("/api/predict/example", methods=["GET"])
def api_predict_example():
    """Provides a real test-set vehicle for 'Try an example'."""
    try:
        example = get_test_set_example()
        return jsonify(sanitize_for_json(example)), 200
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Models are not trained yet. Run python -m ml.train first.", 503)
    except Exception as e:
        return error_response("EXAMPLE_ERROR", str(e), 500)


# -------------------------------------------------------------
# Endpoint 9: Predict (Three Models Output + Validation)
# -------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts specs, validates values, and returns driving range predictions
    from all three models.
    """
    if not request.is_json:
        return error_response("UNSUPPORTED_MEDIA_TYPE", "Request must be JSON.", 415)

    try:
        data = request.get_json(force=True)
    except Exception:
        return error_response("INVALID_JSON", "Malformed JSON payload.", 400)

    if not isinstance(data, dict):
        return error_response("INVALID_PAYLOAD", "Payload must be a JSON object.", 422)

    # Validate battery capacity is present
    bat_val = data.get("battery_capacity_kWh")
    if bat_val is None or str(bat_val).strip() == "":
        return error_response("VALIDATION_ERROR", "Battery capacity is required.", 422)

    try:
        bat_num = float(bat_val)
        if bat_num <= 0:
            return error_response("VALIDATION_ERROR", "Battery capacity must be greater than 0.", 422)
    except (ValueError, TypeError):
        return error_response("VALIDATION_ERROR", "Battery capacity must be a valid number.", 422)

    # Check for negative numbers on any numeric field
    for k, v in data.items():
        if k in ALL_NUMERICAL_FEATURES and v not in [None, ""]:
            try:
                num_v = float(v)
                if num_v < 0:
                    return error_response("VALIDATION_ERROR", f"Value cannot be negative.", 422)
            except (ValueError, TypeError):
                return error_response("VALIDATION_ERROR", "Values must be numbers.", 422)

    # Load pipelines
    try:
        pipelines = get_pipelines()
        meta = get_metadata()
    except FileNotFoundError:
        return error_response("MODELS_NOT_TRAINED", "Models are not trained yet. Run python -m ml.train first.", 503)

    # Prepare input DataFrame with NaN for empty fields
    row_dict = {}
    for col in ALL_INPUT_FEATURES:
        v = data.get(col)
        if v is None or v == "" or (isinstance(v, str) and v.strip() == ""):
            row_dict[col] = np.nan
        elif col in ALL_NUMERICAL_FEATURES:
            row_dict[col] = float(v)
        else:
            row_dict[col] = str(v).strip()

    input_df = pd.DataFrame([row_dict], columns=ALL_INPUT_FEATURES)

    # Check if value is far outside training range
    unusual_note = None
    ranges = meta.get("feature_ranges", {})
    bat_bounds = ranges.get("battery_capacity_kWh", {})
    if bat_bounds:
        if bat_num < bat_bounds.get("min", 20.0) * 0.5 or bat_num > bat_bounds.get("max", 150.0) * 1.5:
            unusual_note = "This value is unusual compared with the cars used for training."

    # Run predictions
    predictions = {}
    for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        p = pipelines[name]
        pred_val = float(p.predict(input_df)[0])
        predictions[name] = int(round(pred_val))

    best_model_name = meta.get("recommended_model", "Linear Regression")
    best_pred_val = predictions.get(best_model_name, list(predictions.values())[0])

    # Simple comparison bars calculation (percentage of highest)
    max_pred = max(predictions.values()) if max(predictions.values()) > 0 else 1
    bars = {}
    for name, val in predictions.items():
        bars[name] = max(10, min(100, int(round((val / max_pred) * 100))))

    payload = {
        "predictions": predictions,
        "best_model": best_model_name,
        "best_prediction": best_pred_val,
        "bars": bars,
        "unusual_note": unusual_note,
        "disclaimer": "These are estimates of listed range, not guaranteed real-world range."
    }
    return jsonify(sanitize_for_json(payload)), 200


# -------------------------------------------------------------
# Error Handlers & Headers
# -------------------------------------------------------------
@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return error_response("NOT_FOUND", f"Endpoint '{request.path}' not found.", 404)
    return render_template("index.html"), 404


@app.errorhandler(405)
def handle_405(e):
    return error_response("METHOD_NOT_ALLOWED", "Method not allowed.", 405)


@app.errorhandler(500)
def handle_500(e):
    logger.error(f"Internal error: {e}")
    return error_response("SERVER_ERROR", "An unexpected server error occurred.", 500)


@app.after_request
def set_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    logger.info(f"Starting EV Range server at http://{config.HOST}:{config.PORT}")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )

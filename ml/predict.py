"""
Prediction and Schema Service Module.
Loads cached model pipelines, validates inference inputs, executes multi-model prediction,
computes ensemble spreads, and provides test-set reference examples.
"""
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR, DATA_DIR
from ml.model_config import (
    ALL_INPUT_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_UNITS
)
from ml.data_loader import load_raw_dataset

logger = logging.getLogger(__name__)

# Global in-memory cache for loaded pipelines and metadata
_CACHED_PIPELINES = None
_CACHED_METADATA = None
_CACHED_TEST_EXAMPLES = None


def get_metadata() -> Dict[str, Any]:
    """Retrieves cached model metadata from models/model_metadata.json."""
    global _CACHED_METADATA
    if _CACHED_METADATA is None:
        meta_path = MODELS_DIR / "model_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("Model metadata not found. Please train models first.")
        with open(meta_path, "r", encoding="utf-8") as f:
            _CACHED_METADATA = json.load(f)
    return _CACHED_METADATA


def get_pipelines() -> Dict[str, Any]:
    """Loads and caches the three scikit-learn model pipelines."""
    global _CACHED_PIPELINES
    if _CACHED_PIPELINES is None:
        lr_path = MODELS_DIR / "linear_regression.joblib"
        rf_path = MODELS_DIR / "random_forest.joblib"
        gb_path = MODELS_DIR / "gradient_boosting.joblib"

        if not (lr_path.exists() and rf_path.exists() and gb_path.exists()):
            raise FileNotFoundError("Trained model pipelines not found in models/ directory. Run training first.")

        _CACHED_PIPELINES = {
            "Linear Regression": joblib.load(lr_path),
            "Random Forest": joblib.load(rf_path),
            "Gradient Boosting": joblib.load(gb_path)
        }
        logger.info("Successfully loaded and cached all 3 ML pipelines.")
    return _CACHED_PIPELINES


def invalidate_model_cache():
    """Invalidates in-memory model caches upon retraining."""
    global _CACHED_PIPELINES, _CACHED_METADATA, _CACHED_TEST_EXAMPLES
    _CACHED_PIPELINES = None
    _CACHED_METADATA = None
    _CACHED_TEST_EXAMPLES = None
    logger.info("Cleared ML in-memory pipeline cache.")


def get_prediction_schema() -> Dict[str, Any]:
    """
    Constructs comprehensive schema for dynamic UI form generation,
    grouped logically with units, min/max training boundaries, and category options.
    """
    metadata = get_metadata()
    ranges = metadata.get("feature_ranges", {})
    categories = metadata.get("categorical_options", {})

    schema = {
        "groups": [
            {
                "group_id": "battery_energy",
                "title": "Battery & Energy",
                "description": "Fundamental electrical energy storage and consumption metrics",
                "fields": [
                    {
                        "name": "battery_capacity_kWh",
                        "label": "Battery Capacity",
                        "type": "number",
                        "unit": "kWh",
                        "required": True,
                        "min": ranges.get("battery_capacity_kWh", {}).get("min", 20.0),
                        "max": ranges.get("battery_capacity_kWh", {}).get("max", 150.0),
                        "median": ranges.get("battery_capacity_kWh", {}).get("median", 77.0),
                        "help": "Total gross capacity of the high-voltage traction pack"
                    },
                    {
                        "name": "efficiency_wh_per_km",
                        "label": "Efficiency Rating",
                        "type": "number",
                        "unit": "Wh/km",
                        "required": False,
                        "min": ranges.get("efficiency_wh_per_km", {}).get("min", 130.0),
                        "max": ranges.get("efficiency_wh_per_km", {}).get("max", 300.0),
                        "median": ranges.get("efficiency_wh_per_km", {}).get("median", 185.0),
                        "help": "Rated energy consumption rate (Optional; imputer uses median)"
                    },
                    {
                        "name": "number_of_cells",
                        "label": "Number of Battery Cells",
                        "type": "number",
                        "unit": "cells",
                        "required": False,
                        "min": ranges.get("number_of_cells", {}).get("min", 90.0),
                        "max": ranges.get("number_of_cells", {}).get("max", 800.0),
                        "median": ranges.get("number_of_cells", {}).get("median", 192.0),
                        "help": "Total individual cell count across modules (Optional; imputer uses median)"
                    }
                ]
            },
            {
                "group_id": "performance",
                "title": "Performance & Dynamics",
                "description": "Speed, torque, and acceleration dynamics",
                "fields": [
                    {
                        "name": "top_speed_kmh",
                        "label": "Top Speed",
                        "type": "number",
                        "unit": "km/h",
                        "required": False,
                        "min": ranges.get("top_speed_kmh", {}).get("min", 120.0),
                        "max": ranges.get("top_speed_kmh", {}).get("max", 260.0),
                        "median": ranges.get("top_speed_kmh", {}).get("median", 180.0),
                        "help": "Maximum rated governed velocity"
                    },
                    {
                        "name": "torque_nm",
                        "label": "Motor Torque",
                        "type": "number",
                        "unit": "Nm",
                        "required": False,
                        "min": ranges.get("torque_nm", {}).get("min", 150.0),
                        "max": ranges.get("torque_nm", {}).get("max", 1200.0),
                        "median": ranges.get("torque_nm", {}).get("median", 450.0),
                        "help": "Peak drivetrain rotational torque output"
                    },
                    {
                        "name": "acceleration_0_100_s",
                        "label": "0-100 km/h Acceleration",
                        "type": "number",
                        "unit": "s",
                        "required": False,
                        "min": ranges.get("acceleration_0_100_s", {}).get("min", 2.0),
                        "max": ranges.get("acceleration_0_100_s", {}).get("max", 15.0),
                        "median": ranges.get("acceleration_0_100_s", {}).get("median", 6.2),
                        "help": "Elapsed time from standstill to 100 km/h"
                    }
                ]
            },
            {
                "group_id": "charging",
                "title": "Charging Capabilities",
                "description": "Direct Current fast-charging characteristics and connector type",
                "fields": [
                    {
                        "name": "fast_charging_power_kw_dc",
                        "label": "DC Fast Charging Power",
                        "type": "number",
                        "unit": "kW DC",
                        "required": False,
                        "min": ranges.get("fast_charging_power_kw_dc", {}).get("min", 20.0),
                        "max": ranges.get("fast_charging_power_kw_dc", {}).get("max", 350.0),
                        "median": ranges.get("fast_charging_power_kw_dc", {}).get("median", 130.0),
                        "help": "Peak DC rapid charging reception acceptance rate"
                    },
                    {
                        "name": "fast_charge_port",
                        "label": "Fast Charge Port Standard",
                        "type": "select",
                        "required": False,
                        "options": categories.get("fast_charge_port", ["CCS", "Tesla", "CHAdeMO"]),
                        "default": categories.get("fast_charge_port", ["CCS"])[0],
                        "help": "High-voltage DC connector specification"
                    }
                ]
            },
            {
                "group_id": "dimensions_capacity",
                "title": "Dimensions & Capacity",
                "description": "Exterior dimensions, passenger seating, cargo, and towing capacity",
                "fields": [
                    {
                        "name": "seats",
                        "label": "Seating Capacity",
                        "type": "number",
                        "unit": "seats",
                        "required": False,
                        "min": 2,
                        "max": 9,
                        "median": 5,
                        "help": "Permitted designated seat count"
                    },
                    {
                        "name": "towing_capacity_kg",
                        "label": "Towing Capacity",
                        "type": "number",
                        "unit": "kg",
                        "required": False,
                        "min": ranges.get("towing_capacity_kg", {}).get("min", 0.0),
                        "max": ranges.get("towing_capacity_kg", {}).get("max", 2500.0),
                        "median": ranges.get("towing_capacity_kg", {}).get("median", 1000.0),
                        "help": "Braked towing allowance (0 if not rated)"
                    },
                    {
                        "name": "cargo_volume_l",
                        "label": "Cargo Trunk Volume",
                        "type": "number",
                        "unit": "L",
                        "required": False,
                        "min": ranges.get("cargo_volume_l", {}).get("min", 100.0),
                        "max": ranges.get("cargo_volume_l", {}).get("max", 1200.0),
                        "median": ranges.get("cargo_volume_l", {}).get("median", 470.0),
                        "help": "Trunk / boot cargo storage volume in Litres"
                    },
                    {
                        "name": "length_mm",
                        "label": "Vehicle Length",
                        "type": "number",
                        "unit": "mm",
                        "required": False,
                        "min": ranges.get("length_mm", {}).get("min", 3000.0),
                        "max": ranges.get("length_mm", {}).get("max", 5500.0),
                        "median": ranges.get("length_mm", {}).get("median", 4700.0),
                        "help": "Overall bumper-to-bumper length"
                    },
                    {
                        "name": "width_mm",
                        "label": "Vehicle Width",
                        "type": "number",
                        "unit": "mm",
                        "required": False,
                        "min": ranges.get("width_mm", {}).get("min", 1500.0),
                        "max": ranges.get("width_mm", {}).get("max", 2200.0),
                        "median": ranges.get("width_mm", {}).get("median", 1880.0),
                        "help": "Vehicle width excluding side mirrors"
                    },
                    {
                        "name": "height_mm",
                        "label": "Vehicle Height",
                        "type": "number",
                        "unit": "mm",
                        "required": False,
                        "min": ranges.get("height_mm", {}).get("min", 1300.0),
                        "max": ranges.get("height_mm", {}).get("max", 2000.0),
                        "median": ranges.get("height_mm", {}).get("median", 1570.0),
                        "help": "Overall ground-to-roof height"
                    }
                ]
            },
            {
                "group_id": "vehicle_identity",
                "title": "Vehicle Categorization",
                "description": "Manufacturer, drive topology, and body style",
                "fields": [
                    {
                        "name": "brand",
                        "label": "Manufacturer Brand",
                        "type": "select",
                        "required": False,
                        "options": categories.get("brand", []),
                        "default": categories.get("brand", ["Tesla"])[0] if categories.get("brand") else "Tesla",
                        "help": "Automotive manufacturing brand"
                    },
                    {
                        "name": "drivetrain",
                        "label": "Drivetrain Configuration",
                        "type": "select",
                        "required": False,
                        "options": categories.get("drivetrain", ["AWD", "RWD", "FWD"]),
                        "default": categories.get("drivetrain", ["AWD"])[0],
                        "help": "Wheel drive topology (All-Wheel, Rear-Wheel, Front-Wheel)"
                    },
                    {
                        "name": "segment",
                        "label": "Market Segment",
                        "type": "select",
                        "required": False,
                        "options": categories.get("segment", ["B", "C", "D", "E", "F", "N"]),
                        "default": categories.get("segment", ["D"])[0],
                        "help": "Automotive market classification"
                    },
                    {
                        "name": "car_body_type",
                        "label": "Body Architecture",
                        "type": "select",
                        "required": False,
                        "options": categories.get("car_body_type", ["SUV", "Sedan", "Hatchback", "Estate", "Liftback", "Coupe"]),
                        "default": categories.get("car_body_type", ["SUV"])[0],
                        "help": "Structural passenger body type"
                    }
                ]
            }
        ],
        "excluded_features_note": "Target variable 'range_km', unique ID 'source_url', vehicle model name, and constant 'battery_type' are intentionally excluded from predictive inputs."
    }
    return schema


def get_test_set_example(index: Optional[int] = None) -> Dict[str, Any]:
    """
    Returns an actual vehicle record from the held-out test set for the UI demo helper.
    Includes the true actual range for direct visual comparison against model estimates.
    """
    global _CACHED_TEST_EXAMPLES
    metadata = get_metadata()
    test_indices = metadata.get("test_indices", [])
    if not test_indices:
        raise ValueError("No test set indices saved in metadata.")

    if _CACHED_TEST_EXAMPLES is None:
        df_raw, _ = load_raw_dataset()
        _CACHED_TEST_EXAMPLES = df_raw.iloc[test_indices].copy()

    if index is None or index < 0 or index >= len(_CACHED_TEST_EXAMPLES):
        # Pick a balanced representative test EV (e.g. 5th test record)
        selected_row = _CACHED_TEST_EXAMPLES.iloc[4]
    else:
        selected_row = _CACHED_TEST_EXAMPLES.iloc[index]

    inputs = {}
    for col in ALL_INPUT_FEATURES:
        val = selected_row.get(col, None)
        if pd.isna(val):
            inputs[col] = None
        elif isinstance(val, (np.integer, int)):
            inputs[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            inputs[col] = round(float(val), 2)
        else:
            inputs[col] = str(val)

    actual_range = float(selected_row[TARGET_COLUMN])

    return {
        "brand": str(selected_row.get("brand", "Unknown")),
        "model": str(selected_row.get("model", "Unknown")),
        "actual_range_km": round(actual_range, 1),
        "inputs": inputs,
        "note": "Loaded from official held-out test set (unseen during model training and tuning)"
    }


def validate_prediction_input(data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Validates user prediction inputs against schema boundaries.
    Enforces non-negativity, checks finiteness, warns on extrapolation,
    and returns sanitized 1-row DataFrame.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid payload: expected JSON object.")

    # Disallowed fields check
    for disallowed in [TARGET_COLUMN, "source_url", "battery_type", "model"]:
        if disallowed in data:
            data.pop(disallowed, None)

    metadata = get_metadata()
    ranges = metadata.get("feature_ranges", {})
    categories = metadata.get("categorical_options", {})
    
    # Check required fields
    if "battery_capacity_kWh" not in data or data["battery_capacity_kWh"] is None:
        raise ValueError("Missing required field: 'battery_capacity_kWh' (Battery Capacity).")

    sanitized_row = {}
    soft_warnings = []

    # 1. Numerical fields validation
    for col in ALL_NUMERICAL_FEATURES:
        val = data.get(col)
        if val is None or val == "" or (isinstance(val, str) and val.strip() == ""):
            # Imputer will handle missing values with median
            sanitized_row[col] = np.nan
            continue

        try:
            num_val = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Field '{col}' must be a valid numeric number, got '{val}'.")

        if not np.isfinite(num_val):
            raise ValueError(f"Field '{col}' must be a finite number.")

        # Physical impossibility checks (hard blocks)
        if num_val < 0:
            raise ValueError(f"Field '{col}' cannot be negative (got {num_val}).")
        
        if col == "battery_capacity_kWh" and num_val <= 0:
            raise ValueError("Battery capacity must be strictly greater than 0 kWh.")
            
        if col == "seats" and (num_val < 1 or num_val > 20):
            raise ValueError(f"Seating capacity must be a realistic number of seats (1-20), got {num_val}.")

        # Soft warning boundary checks (extrapolation warning)
        bounds = ranges.get(col, {})
        min_seen = bounds.get("min")
        max_seen = bounds.get("max")
        if min_seen is not None and max_seen is not None:
            if num_val < min_seen or num_val > max_seen:
                soft_warnings.append(
                    f"{col.replace('_', ' ').title()} ({num_val} {FEATURE_UNITS.get(col, '')}) is outside the training distribution [{min_seen}, {max_seen}]. Prediction reliability may be diminished."
                )

        sanitized_row[col] = num_val

    # 2. Categorical fields validation
    for col in CATEGORICAL_FEATURES:
        val = data.get(col)
        if val is None or str(val).strip() == "":
            sanitized_row[col] = None
        else:
            cat_str = str(val).strip()
            # If category is unseen in training, OneHotEncoder(handle_unknown='ignore') will safely map it to zeros
            sanitized_row[col] = cat_str

    input_df = pd.DataFrame([sanitized_row], columns=ALL_INPUT_FEATURES)
    return input_df, sanitized_row, soft_warnings


def predict_ev_range(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes range prediction using all three saved pipelines:
    Linear Regression, Random Forest, and Gradient Boosting.
    """
    input_df, sanitized_inputs, soft_warnings = validate_prediction_input(data)
    pipelines = get_pipelines()
    metadata = get_metadata()

    predictions = {}
    for name, pipe in pipelines.items():
        raw_pred = float(pipe.predict(input_df)[0])
        predictions[name] = round(raw_pred, 1)

    # Calculate spread between models
    pred_values = list(predictions.values())
    spread = round(float(max(pred_values) - min(pred_values)), 1)

    # Calculate Random Forest internal tree variance (ensemble spread across trees)
    rf_pipeline = pipelines["Random Forest"]
    rf_regressor = rf_pipeline.named_steps["regressor"]
    rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
    
    transformed_X = rf_preprocessor.transform(input_df)
    tree_preds = [float(tree.predict(transformed_X)[0]) for tree in rf_regressor.estimators_]
    rf_tree_min = round(float(np.min(tree_preds)), 1)
    rf_tree_max = round(float(np.max(tree_preds)), 1)
    rf_tree_std = round(float(np.std(tree_preds)), 1)

    recommended_model = metadata.get("recommended_model", "Linear Regression")
    recommended_prediction = predictions.get(recommended_model, pred_values[0])
    
    # Historical held-out MAE
    model_meta = metadata.get("models", {}).get(recommended_model, {})
    historical_mae = model_meta.get("test_metrics", {}).get("mae", 15.0)

    return {
        "predictions": predictions,
        "recommended_model": recommended_model,
        "recommended_prediction_km": recommended_prediction,
        "spread_between_models_km": spread,
        "random_forest_tree_spread": {
            "spread_label": "spread across trees",
            "min_tree_km": rf_tree_min,
            "max_tree_km": rf_tree_max,
            "std_trees_km": rf_tree_std,
            "note": "Ensemble divergence across individual trees; reflects decision tree consensus, not a confidence interval."
        },
        "selected_model_test_mae": historical_mae,
        "mae_description": f"Typical error on held-out vehicles: ±{historical_mae} km (historical test-set MAE, not a confidence interval)",
        "soft_warnings": soft_warnings,
        "sanitized_inputs": sanitized_inputs,
        "scope_disclaimer": "Predictions are estimates of official listed laboratory test range, not guaranteed real-world range across every weather, driving, and battery degradation condition."
    }

"""
Machine Learning Configuration.
Defines constants, schema, feature sets, hyperparameters, and palette for EV Range Prediction.
"""
from pathlib import Path

# Dataset filename - exactly as required by specification with intentional double extension
DATASET_FILENAME = "electric_vehicles_spec_2025.csv.csv"

# Target variable - strictly continuous target for regression
TARGET_COLUMN = "range_km"

# Expected initial dataset schema (all 22 columns in exact order)
EXPECTED_COLUMNS = [
    "brand",
    "model",
    "top_speed_kmh",
    "battery_capacity_kWh",
    "battery_type",
    "number_of_cells",
    "torque_nm",
    "efficiency_wh_per_km",
    "range_km",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "fast_charge_port",
    "towing_capacity_kg",
    "cargo_volume_l",
    "seats",
    "drivetrain",
    "segment",
    "length_mm",
    "width_mm",
    "height_mm",
    "car_body_type",
    "source_url"
]

# Expected initial dataset statistics
EXPECTED_STATS = {
    "rows": 478,
    "columns": 22,
    "exact_duplicates": 0,
    "missing_number_of_cells": 202,
    "target_mean": 393.18,
    "target_median": 397.50,
    "target_min": 135.0,
    "target_max": 685.0,
    "target_std": 103.29
}

# Features excluded from model training
EXCLUDED_FEATURES = [
    TARGET_COLUMN,   # Target variable - never an input feature
    "source_url",    # Identifier field
    "model",         # High-cardinality vehicle designation (477 unique) to avoid memorization
    "battery_type"   # Constant categorical feature ("Lithium-ion" in all records)
]

# Primary numerical features (13 candidate numerical columns)
# Note: number_of_cells is handled via a dedicated pipeline branch with missingness indicator
NUMERICAL_FEATURES_GENERAL = [
    "top_speed_kmh",
    "battery_capacity_kWh",
    "torque_nm",
    "efficiency_wh_per_km",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "towing_capacity_kg",
    "cargo_volume_l",
    "seats",
    "length_mm",
    "width_mm",
    "height_mm"
]

# Cell count column requiring median imputation + missingness indicator
CELLS_FEATURE = "number_of_cells"

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES_GENERAL + [CELLS_FEATURE]

# Categorical features for one-hot encoding
CATEGORICAL_FEATURES = [
    "brand",
    "fast_charge_port",
    "drivetrain",
    "segment",
    "car_body_type"
]

# All input features expected by model pipelines
ALL_INPUT_FEATURES = ALL_NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Split and CV settings
TRAIN_TEST_SPLIT_RATIO = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Hyperparameter search spaces for tuning (configured to finish comfortably in 1-3 minutes)
PARAM_GRID_RF = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5],
    "regressor__min_samples_leaf": [1, 2]
}

PARAM_GRID_GBR = {
    "regressor__n_estimators": [100, 200],
    "regressor__learning_rate": [0.05, 0.1, 0.15],
    "regressor__max_depth": [3, 4, 5],
    "regressor__subsample": [0.8, 1.0]
}

# Design System Colors (exact specification)
PALETTE = {
    "burgundy_900": "#59171B",
    "burgundy_950": "#3A0E11",
    "burgundy_700": "#8B4947",
    "burgundy_500": "#A86B63",
    "peach_500": "#E8B38D",
    "peach_300": "#FED7B8",
    "peach_200": "#FEE6D3",
    "peach_100": "#FFF1E6",
    "peach_50": "#FFF9F4"
}

# Model series colors for charts (palette-only)
MODEL_COLORS = {
    "Linear Regression": "#59171B",
    "Random Forest": "#A86B63",
    "Gradient Boosting": "#E8B38D"
}

# Unit dictionary for human-friendly displays
FEATURE_UNITS = {
    "battery_capacity_kWh": "kWh",
    "top_speed_kmh": "km/h",
    "number_of_cells": "cells",
    "torque_nm": "Nm",
    "efficiency_wh_per_km": "Wh/km",
    "acceleration_0_100_s": "s",
    "fast_charging_power_kw_dc": "kW DC",
    "towing_capacity_kg": "kg",
    "cargo_volume_l": "L",
    "seats": "seats",
    "length_mm": "mm",
    "width_mm": "mm",
    "height_mm": "mm",
    "range_km": "km"
}


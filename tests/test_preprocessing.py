"""
Dataset and Preprocessing Unit Tests.
Verifies dataset integrity, cleaning operations, target-leakage prevention,
and absence of excluded features from predictive input candidate sets.
"""
import pytest
import pandas as pd
import numpy as np
from ml.model_config import (
    EXPECTED_COLUMNS,
    EXPECTED_STATS,
    ALL_INPUT_FEATURES,
    TARGET_COLUMN,
    EXCLUDED_FEATURES
)
from ml.data_loader import load_raw_dataset, audit_raw_dataset
from ml.preprocessing import clean_dataset, audit_target_leakage, detect_outliers_iqr


def test_csv_loading_and_schema(raw_data):
    """Test raw CSV loading, row/column counts, and exact schema."""
    df, info = raw_data
    assert len(df) == EXPECTED_STATS["rows"], f"Expected {EXPECTED_STATS['rows']} rows, got {len(df)}"
    assert len(df.columns) == EXPECTED_STATS["columns"], f"Expected {EXPECTED_STATS['columns']} cols, got {len(df.columns)}"
    assert list(df.columns) == EXPECTED_COLUMNS, "Columns do not match expected schema."
    assert info["file_hash_sha256"] is not None
    assert len(info["file_hash_sha256"]) == 64


def test_audit_recalculations(raw_data):
    """Test audit recalculations and duplicate detection."""
    df, info = raw_data
    audit = audit_raw_dataset(df, info)
    assert audit["exact_duplicates"] == 0
    assert audit["missing_counts"].get("number_of_cells") == 202
    assert audit["target_statistics"]["count"] == 478
    assert abs(audit["target_statistics"]["mean"] - 393.18) < 0.1
    assert abs(audit["target_statistics"]["median"] - 397.50) < 0.1


def test_cargo_volume_cleaning(raw_data):
    """Verify banana-box text strings in cargo_volume_l are coerced to NaN without inventing values."""
    df_raw, _ = raw_data
    banana_box_rows = df_raw["cargo_volume_l"].astype(str).str.contains("Banana Boxes")
    assert banana_box_rows.sum() == 3, "Expected 3 banana-box entries in raw data."

    df_clean, cleaning_log = clean_dataset(df_raw)
    assert pd.api.types.is_numeric_dtype(df_clean["cargo_volume_l"]), "cargo_volume_l must be numeric after cleaning."
    assert df_clean.loc[banana_box_rows, "cargo_volume_l"].isna().all(), "All banana-box entries must be coerced to NaN."

    # Verify cleaning log entry
    cargo_log = next((item for item in cleaning_log if item["step"] == "Cargo Volume Coercion"), None)
    assert cargo_log is not None
    assert cargo_log["count_affected"] >= 3


def test_constant_feature_removal(raw_data):
    """Verify battery_type is constant across all 478 rows and excluded from features."""
    df_raw, _ = raw_data
    assert df_raw["battery_type"].nunique() == 1
    assert df_raw["battery_type"].unique()[0] == "Lithium-ion"
    assert "battery_type" not in ALL_INPUT_FEATURES
    assert "battery_type" in EXCLUDED_FEATURES


def test_target_leakage_and_excluded_features():
    """Verify range_km, source_url, model, battery_type are never in ALL_INPUT_FEATURES."""
    assert TARGET_COLUMN not in ALL_INPUT_FEATURES
    assert "source_url" not in ALL_INPUT_FEATURES
    assert "model" not in ALL_INPUT_FEATURES
    assert "battery_type" not in ALL_INPUT_FEATURES


def test_target_leakage_audit_calculation(clean_data):
    """Verify target leakage calculation formula and correlation."""
    df_clean, _ = clean_data
    audit = audit_target_leakage(df_clean)
    assert audit["pearson_correlation"] > 0.90
    assert audit["share_within_10_percent"] > 15.0
    assert "implied_range = battery_capacity_kWh * 1000 / efficiency_wh_per_km" in audit["formula"]


def test_outlier_detection(clean_data):
    """Verify IQR outlier detection records entries with reasons and keeps data."""
    df_clean, _ = clean_data
    outliers = detect_outliers_iqr(df_clean)
    assert len(outliers) > 0
    for out in outliers[:5]:
        assert out["action"] == "Retained"
        assert len(out["reason"]) > 10
        assert out["value"] is not None


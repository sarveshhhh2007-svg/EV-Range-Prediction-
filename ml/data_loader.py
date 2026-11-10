"""
Data Loader and Audit Module.
Handles robust loading, character encoding fallback, schema validation,
SHA-256 integrity verification, and initial dataset auditing.
"""
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from config import DATA_DIR, BASE_DIR
from ml.model_config import (
    DATASET_FILENAME,
    EXPECTED_COLUMNS,
    EXPECTED_STATS,
    TARGET_COLUMN
)

logger = logging.getLogger(__name__)


def locate_dataset_file(custom_path: Path = None) -> Path:
    """Finds the dataset file across standard locations."""
    if custom_path and custom_path.exists():
        return custom_path
    
    # Check data/ directory first
    data_path = DATA_DIR / DATASET_FILENAME
    if data_path.exists():
        return data_path
    
    # Check project base directory
    base_path = BASE_DIR / DATASET_FILENAME
    if base_path.exists():
        return base_path
        
    raise FileNotFoundError(
        f"Dataset file '{DATASET_FILENAME}' not found in '{DATA_DIR}' or '{BASE_DIR}'. "
        "Please place the CSV in the data/ directory."
    )


def compute_file_hash(file_path: Path) -> str:
    """Calculates SHA-256 hash of the raw dataset file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_raw_dataset(file_path: Path = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV with encoding fallbacks (utf-8 -> utf-8-sig -> latin-1).
    Validates schema against the expected 22 columns.
    Strips whitespace from column names and string fields.
    """
    target_path = locate_dataset_file(file_path)
    file_hash = compute_file_hash(target_path)
    
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    loaded_df = None
    successful_encoding = None
    
    for enc in encodings:
        try:
            loaded_df = pd.read_csv(target_path, encoding=enc)
            successful_encoding = enc
            logger.info(f"Loaded dataset successfully with encoding '{enc}'")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file at {target_path}: {e}")
            
    if loaded_df is None:
        raise ValueError(
            f"Failed to decode '{target_path.name}' using supported encodings: {encodings}"
        )
        
    # Strip whitespace from column names
    loaded_df.columns = [c.strip() for c in loaded_df.columns]
    
    # Validate column count and names
    actual_cols = list(loaded_df.columns)
    if actual_cols != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(actual_cols)
        unexpected = set(actual_cols) - set(EXPECTED_COLUMNS)
        raise ValueError(
            f"Dataset schema mismatch! Expected {len(EXPECTED_COLUMNS)} columns, got {len(actual_cols)}. "
            f"Missing columns: {list(missing)}. Unexpected columns: {list(unexpected)}."
        )
        
    # Strip whitespace from string values
    str_cols = loaded_df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        loaded_df[col] = loaded_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
    audit_info = {
        "file_path": str(target_path),
        "file_hash_sha256": file_hash,
        "encoding_used": successful_encoding,
        "raw_rows": int(len(loaded_df)),
        "raw_cols": int(len(loaded_df.columns))
    }
    
    return loaded_df, audit_info


def audit_raw_dataset(df: pd.DataFrame, audit_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recalculates dataset statistics, audits missing values, duplicates,
    and compares strictly with EXPECTED_STATS.
    """
    rows, cols = df.shape
    exact_duplicates = int(df.duplicated().sum())
    missing_counts = {col: int(count) for col, count in df.isnull().sum().items() if count > 0}
    missing_percentages = {col: round(float(count / rows * 100), 2) for col, count in missing_counts.items()}
    
    target_series = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    target_stats = {
        "count": int(target_series.count()),
        "mean": round(float(target_series.mean()), 2),
        "median": round(float(target_series.median()), 2),
        "min": round(float(target_series.min()), 2),
        "max": round(float(target_series.max()), 2),
        "std": round(float(target_series.std()), 2)
    }
    
    # Check against expectations
    differs_notes = []
    if rows != EXPECTED_STATS["rows"]:
        differs_notes.append(f"Row count differs: actual {rows} vs expected {EXPECTED_STATS['rows']}")
    if cols != EXPECTED_STATS["columns"]:
        differs_notes.append(f"Column count differs: actual {cols} vs expected {EXPECTED_STATS['columns']}")
    if exact_duplicates != EXPECTED_STATS["exact_duplicates"]:
        differs_notes.append(f"Duplicates differ: actual {exact_duplicates} vs expected {EXPECTED_STATS['exact_duplicates']}")
    if missing_counts.get("number_of_cells", 0) != EXPECTED_STATS["missing_number_of_cells"]:
        differs_notes.append(
            f"Missing cells count differs: actual {missing_counts.get('number_of_cells', 0)} "
            f"vs expected {EXPECTED_STATS['missing_number_of_cells']}"
        )
    if abs(target_stats["mean"] - EXPECTED_STATS["target_mean"]) > 0.1:
        differs_notes.append(f"Target mean differs: actual {target_stats['mean']} vs expected {EXPECTED_STATS['target_mean']}")
    if abs(target_stats["median"] - EXPECTED_STATS["target_median"]) > 0.1:
        differs_notes.append(f"Target median differs: actual {target_stats['median']} vs expected {EXPECTED_STATS['target_median']}")

    audit_report = {
        "audit_metadata": audit_info,
        "rows": rows,
        "columns": cols,
        "exact_duplicates": exact_duplicates,
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "target_variable": TARGET_COLUMN,
        "target_statistics": target_stats,
        "expected_stats": EXPECTED_STATS,
        "differs_notes": differs_notes,
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    return audit_report


"""
Pytest configuration and shared fixtures for EV Range Prediction platform.
"""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from app import app
from ml.data_loader import load_raw_dataset
from ml.preprocessing import clean_dataset
from ml.predict import get_pipelines, get_metadata


@pytest.fixture(scope="session")
def flask_client():
    """Provides test client for Flask application."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="session")
def raw_data():
    """Loads raw dataset once for test session."""
    df, info = load_raw_dataset()
    return df, info


@pytest.fixture(scope="session")
def clean_data(raw_data):
    """Provides cleaned dataframe and cleaning log."""
    df_raw, _ = raw_data
    df_clean, log = clean_dataset(df_raw)
    return df_clean, log


@pytest.fixture(scope="session")
def trained_pipelines():
    """Loads trained scikit-learn pipelines."""
    return get_pipelines()


@pytest.fixture(scope="session")
def model_metadata():
    """Loads model metadata."""
    return get_metadata()

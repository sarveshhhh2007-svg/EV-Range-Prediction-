"""
Configuration module for Electric Vehicle Range Prediction & ML Analytics Platform.
Defines project-wide settings, paths, and environment defaults.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = BASE_DIR / "ml"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Web Server Configuration
# Bind to 127.0.0.1 by default; port 5000 (configurable via env var, e.g. for macOS port 5000 conflicts)
HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLASK_PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")

# Maximum content length for API POST requests (5MB)
MAX_CONTENT_LENGTH = 5 * 1024 * 1024

# Ensure core directories exist
for folder in [DATA_DIR, ML_DIR, MODELS_DIR, REPORTS_DIR, STATIC_DIR, TEMPLATES_DIR, DOCS_DIR, NOTEBOOKS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


"""Paths and constants shared across the package."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("PROPERTY_APPRAISAL_DATA_DIR", PROJECT_ROOT / "data"))
TRAINING_DATA_DIR = DATA_DIR / "training"
"""Historical listings used to fit the valuation model."""
INFERENCE_DATA_DIR = DATA_DIR / "inference"
"""Listings appraised by the web app."""

RANDOM_STATE = 42

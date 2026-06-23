"""
Shared model/training constants for Project Watchdog.
"""

import os
from pathlib import Path

# Project layout
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "docs"))

# Ground Station config
GS_LATITUDE = float(os.getenv("GS_LATITUDE", "29.0661"))
GS_LONGITUDE = float(os.getenv("GS_LONGITUDE", "31.0994"))
GS_ELEVATION = float(os.getenv("GS_ELEVATION", "32.0"))
GS_NAME = os.getenv("GS_NAME", "Beni Suef, Egypt")


HIDDEN_DIM = 12
LATENT_DIM = 3
DEFAULT_KLD_WEIGHT = 0.05

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
THRESHOLD_PERCENTILE = 99.9
DEFAULT_INFERENCE_MODE = "deterministic"

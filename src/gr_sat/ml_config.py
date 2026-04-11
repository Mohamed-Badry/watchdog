"""
Shared model/training constants for Project Watchdog.
"""

BASE_FEATURES = [
    "batt_voltage",
    "batt_current",
    "temp_batt_a",
    "temp_batt_b",
    "temp_panel_z",
]

ALL_FEATURES = BASE_FEATURES

HIDDEN_DIM = 12
LATENT_DIM = 3
DEFAULT_KLD_WEIGHT = 0.05

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
THRESHOLD_PERCENTILE = 95.0
DEFAULT_INFERENCE_MODE = "deterministic"

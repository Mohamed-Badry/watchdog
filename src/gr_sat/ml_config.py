"""
Shared model/training constants for Project Watchdog.
"""

from gr_sat.satellite_profiles import DEFAULT_PROFILE

BASE_FEATURES = list(DEFAULT_PROFILE.feature_contract.diagnosis_feature_names)
ALL_FEATURES = list(DEFAULT_PROFILE.feature_contract.feature_names)
DEFAULT_FEATURE_CONTRACT_VERSION = DEFAULT_PROFILE.feature_contract.version

HIDDEN_DIM = 12
LATENT_DIM = 3
DEFAULT_KLD_WEIGHT = 0.05

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
THRESHOLD_PERCENTILE = 95.0
DEFAULT_INFERENCE_MODE = "deterministic"

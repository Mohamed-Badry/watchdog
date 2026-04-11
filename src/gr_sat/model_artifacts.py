"""
Artifact metadata and split helpers for anomaly-model training/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from gr_sat.ml_config import (
    ALL_FEATURES,
    DEFAULT_INFERENCE_MODE,
    DEFAULT_KLD_WEIGHT,
    HIDDEN_DIM,
    LATENT_DIM,
    THRESHOLD_PERCENTILE,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from gr_sat.models import TelemetryVAE

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ArtifactPaths:
    scaler: Path
    weights: Path
    metadata: Path


@dataclass(frozen=True)
class ModelArtifactMetadata:
    version: int
    norad_id: str
    feature_names: list[str]
    hidden_dim: int
    latent_dim: int
    kld_weight: float
    threshold: float
    threshold_percentile: float
    inference_mode: str
    train_rows: int
    validation_rows: int
    test_rows: int
    train_start: str | None
    train_end: str | None
    validation_start: str | None
    validation_end: str | None
    test_start: str | None
    test_end: str | None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "norad_id": self.norad_id,
            "feature_names": self.feature_names,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "kld_weight": self.kld_weight,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "inference_mode": self.inference_mode,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelArtifactMetadata":
        return cls(**payload)

    @classmethod
    def from_split(
        cls,
        norad_id: str,
        split: ChronologicalSplit,
        threshold: float,
        feature_names: list[str] | None = None,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        kld_weight: float = DEFAULT_KLD_WEIGHT,
        threshold_percentile: float = THRESHOLD_PERCENTILE,
        inference_mode: str = DEFAULT_INFERENCE_MODE,
    ) -> "ModelArtifactMetadata":
        features = list(feature_names or ALL_FEATURES)
        train_start, train_end = _timestamp_bounds(split.train)
        validation_start, validation_end = _timestamp_bounds(split.validation)
        test_start, test_end = _timestamp_bounds(split.test)
        return cls(
            version=ARTIFACT_VERSION,
            norad_id=str(norad_id),
            feature_names=features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            kld_weight=float(kld_weight),
            threshold=float(threshold),
            threshold_percentile=float(threshold_percentile),
            inference_mode=inference_mode,
            train_rows=len(split.train),
            validation_rows=len(split.validation),
            test_rows=len(split.test),
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
        )


def _timestamp_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty or "timestamp" not in df.columns:
        return None, None
    start = pd.to_datetime(df["timestamp"].min())
    end = pd.to_datetime(df["timestamp"].max())
    return start.isoformat(), end.isoformat()


def split_chronological(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_SPLIT,
    validation_fraction: float = VALIDATION_SPLIT,
) -> ChronologicalSplit:
    if len(df) < 3:
        raise ValueError("Need at least 3 clean frames for train/validation/test splits.")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must leave room for test data.")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True).copy()
    n_rows = len(df_sorted)

    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))

    train_end = min(max(train_end, 1), n_rows - 2)
    validation_end = min(max(validation_end, train_end + 1), n_rows - 1)

    if train_end >= validation_end or validation_end >= n_rows:
        raise ValueError("Unable to create non-empty chronological splits.")

    return ChronologicalSplit(
        train=df_sorted.iloc[:train_end].copy(),
        validation=df_sorted.iloc[train_end:validation_end].copy(),
        test=df_sorted.iloc[validation_end:].copy(),
    )


def threshold_from_scores(
    scores: np.ndarray | list[float],
    percentile: float = THRESHOLD_PERCENTILE,
) -> float:
    score_array = np.asarray(scores, dtype=float)
    if score_array.size == 0:
        raise ValueError("Cannot derive a threshold from an empty score array.")
    return float(np.percentile(score_array, percentile))


def model_artifact_paths(models_dir: Path, norad_id: str) -> ArtifactPaths:
    model_root = Path(models_dir)
    sat_id = str(norad_id)
    return ArtifactPaths(
        scaler=model_root / f"{sat_id}_scaler.pkl",
        weights=model_root / f"{sat_id}_vae.pt",
        metadata=model_root / f"{sat_id}_metadata.json",
    )


def save_model_metadata(path: Path, metadata: ModelArtifactMetadata) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_model_metadata(path: Path) -> ModelArtifactMetadata:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelArtifactMetadata.from_dict(payload)


def _load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def load_model_artifacts(norad_id: str, models_dir: Path) -> tuple:
    paths = model_artifact_paths(models_dir, norad_id)
    scaler = joblib.load(paths.scaler)
    metadata = load_model_metadata(paths.metadata)
    vae = TelemetryVAE(
        input_dim=len(metadata.feature_names),
        hidden_dim=metadata.hidden_dim,
        latent_dim=metadata.latent_dim,
    )
    vae.load_state_dict(_load_state_dict(paths.weights))
    vae.eval()
    return scaler, vae, metadata

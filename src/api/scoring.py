"""ML model status, inference, and anomaly scoring.

Isolates all PyTorch / scikit-learn operations behind a clean service
interface so that the rest of the API layer stays ML-free.
"""

from __future__ import annotations

from loguru import logger
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gr_sat.ml.model_artifacts import (
    ModelArtifactMetadata,
    load_model_artifacts,
    load_model_metadata,
    model_artifact_paths,
)
from gr_sat.ml.vae import compute_anomaly_scores

try:
    from .serialization import bool_value, json_value
except ImportError:
    from serialization import bool_value, json_value




# ── Data transfer object ─────────────────────────────────────────────


@dataclass(frozen=True)
class ModelStatus:
    """Snapshot of a satellite's anomaly-detection model readiness."""

    status: str
    detail: str
    metadata: ModelArtifactMetadata | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.metadata is None:
            return {
                "status": self.status,
                "detail": self.detail,
                "error": self.error,
                "threshold": None,
                "inference_mode": None,
                "artifact_version": None,
                "feature_names": [],
                "feature_contract_version": None,
                "train_rows": 0,
                "validation_rows": 0,
                "test_rows": 0,
            }

        return {
            "status": self.status,
            "detail": self.detail,
            "error": self.error,
            "threshold": json_value(self.metadata.threshold),
            "inference_mode": self.metadata.inference_mode,
            "artifact_version": self.metadata.version,
            "feature_names": list(self.metadata.feature_names),
            "diagnosis_feature_names": list(
                self.metadata.diagnosis_feature_names or self.metadata.feature_names
            ),
            "feature_contract_version": self.metadata.feature_contract_version,
            "train_rows": self.metadata.train_rows,
            "validation_rows": self.metadata.validation_rows,
            "test_rows": self.metadata.test_rows,
            "train_start": self.metadata.train_start,
            "train_end": self.metadata.train_end,
            "validation_start": self.metadata.validation_start,
            "validation_end": self.metadata.validation_end,
            "test_start": self.metadata.test_start,
            "test_end": self.metadata.test_end,
        }


# ── Service ──────────────────────────────────────────────────────────


class ScoringService:
    """Manages ML model lifecycle: status checks, batch scoring,
    single-frame reconstruction for root-cause attribution.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._status_cache: dict[int, ModelStatus] = {}
        self._loaded_models: dict[int, tuple] = {}

    # ── Public API ───────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear model artifacts from memory so they can be reloaded from disk."""
        self._status_cache.clear()
        self._loaded_models.clear()
        logger.info("ScoringService cache cleared.")

    def model_status(self, norad_id: int) -> ModelStatus:
        """Check whether a model is available and ready for inference."""
        sat_id = int(norad_id)
        if sat_id not in self._status_cache:
            self._status_cache[sat_id] = self._resolve_status(sat_id)
        return self._status_cache[sat_id]

    def score_frames(
        self,
        norad_id: int,
        df: pd.DataFrame,
        model_status: ModelStatus,
    ) -> pd.DataFrame:
        """Batch-score a DataFrame of telemetry frames using the VAE model."""
        assert model_status.metadata is not None
        working = df.copy()
        try:
            scaler, model, metadata = self._load_artifacts(norad_id)
            feature_names = list(metadata.feature_names)
            complete_mask = working[feature_names].notna().all(axis=1)
            if not complete_mask.any():
                return working

            feature_matrix = (
                working.loc[complete_mask, feature_names].astype(float).to_numpy()
            )
            scaled = scaler.transform(feature_matrix)
            x_tensor = torch.FloatTensor(scaled)
            model.eval()
            with torch.no_grad():
                recon_x, mu, logvar = model(x_tensor)
                scores = compute_anomaly_scores(
                    recon_x,
                    x_tensor,
                    mu,
                    logvar,
                    kld_weight=metadata.kld_weight,
                ).numpy()

            working.loc[complete_mask, "anomaly_score"] = scores
            working["is_anomaly"] = (
                working["anomaly_score"].gt(metadata.threshold).fillna(False)
            )
        except Exception as exc:
            self._status_cache[int(norad_id)] = ModelStatus(
                status="error",
                detail="Model artifacts exist, but dashboard scoring failed.",
                metadata=model_status.metadata,
                error=str(exc),
            )

        return working

    def reconstruct_frame(
        self, norad_id: int, features: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Run single-frame VAE reconstruction for anomaly root-cause
        attribution.

        Returns a dict with ``reconstructed_features``,
        ``scaled_features``, ``scaled_reconstructed_features``, and
        ``feature_contributions``, or ``None`` if inference fails.
        """
        try:
            scaler, vae, metadata = self._load_artifacts(norad_id)
        except Exception as exc:
            logger.warning("Failed to load model for anomaly detail: %s", exc)
            return None

        try:
            feat_names = metadata.feature_names
            input_vec = [features.get(f) for f in feat_names]
            if not all(v is not None for v in input_vec):
                return None

            X = np.array([input_vec], dtype=np.float32)
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled)

            with torch.no_grad():
                recon, mu, logvar = vae(X_tensor)

            recon_np = recon.numpy()
            recon_unscaled = scaler.inverse_transform(recon_np)[0]

            diagnosis_feature_names = set(
                metadata.diagnosis_feature_names or feat_names
            )
            return {
                "reconstructed_features": {
                    f: float(recon_unscaled[i]) for i, f in enumerate(feat_names)
                },
                "scaled_features": {
                    f: float(X_scaled[0][i]) for i, f in enumerate(feat_names)
                },
                "scaled_reconstructed_features": {
                    f: float(recon_np[0][i]) for i, f in enumerate(feat_names)
                },
                "feature_contributions": {
                    f: float(abs(X_scaled[0][i] - recon_np[0][i]))
                    for i, f in enumerate(feat_names)
                    if f in diagnosis_feature_names
                },
            }
        except Exception as exc:
            logger.warning("Inference failed for %d: %s", norad_id, exc)
            return None

    def ensure_scores(self, norad_id: int, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly scores to *df* if missing and a model is available."""
        working = df.copy()

        if "is_anomaly" in working.columns:
            working["is_anomaly"] = working["is_anomaly"].map(bool_value).fillna(False)
        else:
            working["is_anomaly"] = False

        if "anomaly_score" not in working.columns:
            working["anomaly_score"] = np.nan

        status = self.model_status(norad_id)
        if working["anomaly_score"].isna().all() and status.status == "ready":
            working = self.score_frames(norad_id, working, status)
        elif status.metadata is not None:
            threshold = status.metadata.threshold
            missing_anomaly_flags = (
                "is_anomaly" not in df.columns or working["is_anomaly"].isna().any()
            )
            if missing_anomaly_flags:
                working["is_anomaly"] = (
                    working["anomaly_score"].gt(threshold).fillna(False)
                )

        return working

    # ── Private helpers ──────────────────────────────────────────────

    def _load_artifacts(self, norad_id: int) -> tuple:
        """Load model artifacts (scaler, vae, metadata), caching for reuse."""
        sat_id = int(norad_id)
        if sat_id not in self._loaded_models:
            self._loaded_models[sat_id] = load_model_artifacts(
                str(sat_id), self.models_dir
            )
        return self._loaded_models[sat_id]

    def _resolve_status(self, sat_id: int) -> ModelStatus:
        """Probe the filesystem to determine model readiness."""
        paths = model_artifact_paths(self.models_dir, str(sat_id))
        if not paths.metadata.exists():
            return ModelStatus(
                status="missing",
                detail=f"No model metadata found at {paths.metadata}.",
            )

        try:
            metadata = load_model_metadata(paths.metadata)
        except Exception as exc:
            return ModelStatus(
                status="error",
                detail="Model metadata could not be loaded.",
                error=str(exc),
            )

        missing = [
            str(path) for path in (paths.scaler, paths.weights) if not path.exists()
        ]
        if missing:
            return ModelStatus(
                status="metadata_only",
                detail="Model metadata is available, but inference artifacts are incomplete.",
                metadata=metadata,
                error=f"Missing artifacts: {', '.join(missing)}",
            )

        return ModelStatus(
            status="ready",
            detail="Scaler, VAE weights, and metadata are available.",
            metadata=metadata,
        )

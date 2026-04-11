"""
Minimal online watchdog runtime for deterministic packet-by-packet inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from gr_sat.model_artifacts import ModelArtifactMetadata, load_model_artifacts
from gr_sat.models import compute_anomaly_scores
from gr_sat.telemetry import TelemetryFrame, process_frame

STATE_IDLE = "idle"
STATE_RECEIVING = "receiving"
STATE_GAP = "gap"
STATE_DEGRADED = "degraded"
STATE_ALERTING = "alerting"


@dataclass(frozen=True)
class WatchdogAlert:
    norad_id: str
    timestamp: datetime
    score: float
    threshold: float
    source: str
    features: dict[str, float]


@dataclass(frozen=True)
class WatchdogResult:
    status: str
    state: str
    score: float | None = None
    threshold: float | None = None
    is_anomaly: bool = False
    error: str | None = None
    frame: TelemetryFrame | None = None


class OnlineWatchdog:
    def __init__(
        self,
        norad_id: str,
        scaler,
        model,
        metadata: ModelArtifactMetadata,
        gap_timeout_seconds: float = 180.0,
        alert_sink: Callable[[WatchdogAlert], None] | None = None,
    ):
        self.norad_id = int(norad_id)
        self.scaler = scaler
        self.model = model
        self.metadata = metadata
        self.gap_timeout_seconds = float(gap_timeout_seconds)
        self.alert_sink = alert_sink
        self.state = STATE_IDLE
        self.last_packet_at: datetime | None = None

    @classmethod
    def from_artifacts(
        cls,
        norad_id: str,
        models_dir: Path = Path("models"),
        gap_timeout_seconds: float = 180.0,
        alert_sink: Callable[[WatchdogAlert], None] | None = None,
    ) -> "OnlineWatchdog":
        scaler, model, metadata = load_model_artifacts(norad_id, models_dir)
        return cls(
            norad_id=norad_id,
            scaler=scaler,
            model=model,
            metadata=metadata,
            gap_timeout_seconds=gap_timeout_seconds,
            alert_sink=alert_sink,
        )

    def _feature_vector(self, frame: TelemetryFrame) -> np.ndarray:
        values = []
        for feature_name in self.metadata.feature_names:
            value = getattr(frame, feature_name, None)
            if value is None or pd.isna(value):
                raise ValueError(f"Missing required feature '{feature_name}' for inference.")
            values.append(float(value))
        return np.asarray(values, dtype=float)

    def _score_frame(self, frame: TelemetryFrame) -> float:
        feature_vector = self._feature_vector(frame)
        scaled_vector = self.scaler.transform([feature_vector])
        x_tensor = torch.FloatTensor(scaled_vector)
        self.model.eval()
        with torch.no_grad():
            recon_x, mu, logvar = self.model(x_tensor)
            score_tensor = compute_anomaly_scores(
                recon_x,
                x_tensor,
                mu,
                logvar,
                kld_weight=self.metadata.kld_weight,
            )
        return float(score_tensor.item())

    def process_packet(
        self,
        payload: bytes,
        timestamp: datetime,
        source: str = "live_station",
    ) -> WatchdogResult:
        self.last_packet_at = timestamp

        try:
            frame = process_frame(self.norad_id, payload, source, timestamp)
            if frame is None:
                self.state = STATE_RECEIVING
                return WatchdogResult(
                    status="decode_failed",
                    state=self.state,
                )

            score = self._score_frame(frame)
            is_anomaly = score > self.metadata.threshold
            self.state = STATE_ALERTING if is_anomaly else STATE_RECEIVING

            if is_anomaly and self.alert_sink is not None:
                self.alert_sink(
                    WatchdogAlert(
                        norad_id=str(self.norad_id),
                        timestamp=timestamp,
                        score=score,
                        threshold=self.metadata.threshold,
                        source=source,
                        features={
                            feature_name: float(getattr(frame, feature_name))
                            for feature_name in self.metadata.feature_names
                        },
                    )
                )

            return WatchdogResult(
                status="ok",
                state=self.state,
                score=score,
                threshold=self.metadata.threshold,
                is_anomaly=is_anomaly,
                frame=frame,
            )
        except Exception as exc:
            self.state = STATE_DEGRADED
            return WatchdogResult(
                status="error",
                state=self.state,
                error=str(exc),
            )

    def check_gap(self, now: datetime) -> str:
        if self.last_packet_at is None:
            self.state = STATE_IDLE
            return self.state

        elapsed = (now - self.last_packet_at).total_seconds()
        if elapsed > self.gap_timeout_seconds:
            self.state = STATE_GAP
        elif self.state != STATE_DEGRADED:
            self.state = STATE_RECEIVING
        return self.state

    def status(self) -> dict:
        return {
            "norad_id": str(self.norad_id),
            "state": self.state,
            "last_packet_at": self.last_packet_at.isoformat() if self.last_packet_at else None,
            "threshold": self.metadata.threshold,
            "inference_mode": self.metadata.inference_mode,
            "feature_names": list(self.metadata.feature_names),
        }

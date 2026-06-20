"""Pure value-conversion helpers shared across the API layer.

Every function here is side-effect-free and imported by the other
dashboard sub-modules (frame_store, scoring, formatters, etc.).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ── Column name tuples ──────────────────────────────────────────────

FEATURE_FIELDS = (
    "batt_voltage",
    "batt_current",
    "batt_a_voltage",
    "batt_b_voltage",
    "batt_a_current",
    "batt_b_current",
    "power_consumption",
    "temp_obc",
    "temp_batt_a",
    "temp_batt_b",
    "temp_panel_z",
    "uptime",
)


QUALITY_FIELDS = (
    "frame_is_complete",
    "missing_raw_fields",
    "missing_raw_field_count",
    "sampling_irregular",
    "dropped_packet_suspect",
    "same_timestamp_collision",
    "seconds_since_prev",
    "seconds_to_next",
    "pass_id",
    "pass_frame_index",
    "pass_frame_count",
    "pass_duration_sec",
    "pass_median_cadence_sec",
    "cadence_reference_sec",
)


# ── Value helpers ────────────────────────────────────────────────────


def now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def timestamp_iso(value: Any) -> str | None:
    """Convert a timestamp-like *value* to ISO-8601, or ``None``."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def json_value(value: Any) -> Any:
    """Make *value* JSON-safe (handle NaN, numpy scalars, Timestamps)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return timestamp_iso(value)
    if isinstance(value, float):
        return float(value)
    return value


def bool_value(value: Any) -> bool | None:
    """Coerce *value* to ``bool`` (handles string truthy variants)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def missing_fields_value(value: Any) -> list[str]:
    """Parse a missing-fields cell into a clean ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            import json

            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [str(value)]

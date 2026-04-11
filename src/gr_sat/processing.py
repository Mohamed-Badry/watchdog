"""
Helpers for telemetry-frame deduplication and processing metadata.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
import math
from typing import Any, Mapping

import pandas as pd

IGNORED_DEDUP_FIELDS = frozenset({"timestamp", "observation_id"})


def _normalize_frame_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp | datetime):
        return pd.Timestamp(value).isoformat()

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except ValueError:
            pass

    if value is pd.NaT:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def frame_payload_fingerprint(
    frame: Mapping[str, Any],
    ignored_fields: set[str] | frozenset[str] = IGNORED_DEDUP_FIELDS,
) -> str:
    normalized = {
        key: _normalize_frame_value(value)
        for key, value in frame.items()
        if key not in ignored_fields and not key.startswith("_")
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_processed_frames(df_processed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if df_processed.empty:
        return df_processed.copy(), {
            "input_rows": 0,
            "deduplicated_rows": 0,
            "exact_duplicates_removed": 0,
            "same_timestamp_multi_payload_groups": 0,
            "same_timestamp_multi_payload_rows": 0,
            "same_observation_multi_payload_rows": 0,
        }

    working = df_processed.copy().reset_index(drop=True)
    working["_row_order"] = working.index
    working["_timestamp_key"] = working["timestamp"].map(_normalize_frame_value)
    working["_payload_fingerprint"] = working.apply(
        lambda row: frame_payload_fingerprint(row.to_dict()),
        axis=1,
    )
    working["_dedupe_key"] = (
        working["_timestamp_key"].astype(str) + "|" + working["_payload_fingerprint"]
    )

    exact_duplicate_mask = working.duplicated(subset=["_dedupe_key"], keep="first")

    per_timestamp_payloads = working.groupby("_timestamp_key")["_payload_fingerprint"].nunique()
    same_timestamp_multi_payload_rows = int(
        working["_timestamp_key"].map(per_timestamp_payloads).gt(1).sum()
    )
    same_timestamp_multi_payload_groups = int((per_timestamp_payloads > 1).sum())

    same_observation_multi_payload_rows = 0
    if "observation_id" in working.columns:
        observed = working[working["observation_id"].notna()]
        if not observed.empty:
            per_observation_payloads = observed.groupby("observation_id")["_payload_fingerprint"].nunique()
            same_observation_multi_payload_rows = int(
                observed["observation_id"].map(per_observation_payloads).gt(1).sum()
            )

    deduplicated = (
        working.loc[~exact_duplicate_mask]
        .sort_values(["timestamp", "_row_order"])
        .drop(columns=["_row_order", "_timestamp_key", "_payload_fingerprint", "_dedupe_key"])
        .reset_index(drop=True)
    )

    stats = {
        "input_rows": int(len(df_processed)),
        "deduplicated_rows": int(len(deduplicated)),
        "exact_duplicates_removed": int(exact_duplicate_mask.sum()),
        "same_timestamp_multi_payload_groups": same_timestamp_multi_payload_groups,
        "same_timestamp_multi_payload_rows": same_timestamp_multi_payload_rows,
        "same_observation_multi_payload_rows": same_observation_multi_payload_rows,
    }
    return deduplicated, stats

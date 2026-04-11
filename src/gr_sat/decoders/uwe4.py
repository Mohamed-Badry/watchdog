"""
UWE-4 (NORAD 43880) Telemetry Decoder.

Uses the satnogs-decoders Kaitai Struct for binary parsing, then maps the
raw fields to SI-unit Golden Features.

UWE-4 "DP0UWH" is a 1U CubeSat from the University of Würzburg. Its beacon
transmits on 435.6 MHz using 9600 bps FSK and contains:
  - Dual battery pack voltages/currents (A + B)
  - 6-panel solar temperatures
  - OBC temperature
  - Power consumption
  - Uptime and subsystem status

Kaitai struct reference:
  satnogsdecoders.decoder.uwe4.Uwe4

Unit conversions (raw Kaitai → SI):
  - Voltages: raw values are in mV → divide by 1000 → Volts
  - Currents: raw values are in mA → divide by 1000 → Amps
  - Temperatures: raw values are already in °C (integers)
  - Power: raw value is in mW → divide by 1000 → Watts
"""

from typing import Dict, Any, Optional
import math

import satnogsdecoders.decoder as dec
from satnogsdecoders.decoder.uwe4 import Uwe4

from gr_sat.telemetry import (
    BaseDecoder,
    DecoderRegistry,
    ProcessingFailure,
    StageOutcome,
)


@DecoderRegistry.register(43880)
class UWE4Decoder(BaseDecoder):
    """
    Decoder for UWE-4 (NORAD 43880).

    Stage 1 (decode): Raw AX.25 payload → all Kaitai fields as-is.
    Stage 2 (adapt): Kaitai fields → SI-unit Golden Features.
    """

    # Fields required to validate a successful decode
    REQUIRED_FIELDS = {"beacon_payload_uptime", "beacon_payload_batt_a_voltage"}
    NUMERIC_FIELD_MAP = {
        "beacon_payload_batt_a_voltage": ("batt_a_voltage", 1000.0),
        "beacon_payload_batt_b_voltage": ("batt_b_voltage", 1000.0),
        "beacon_payload_batt_a_current": ("batt_a_current", 1000.0),
        "beacon_payload_batt_b_current": ("batt_b_current", 1000.0),
        "beacon_payload_power_consumption": ("power_consumption", 1000.0),
        "beacon_payload_obc_temp": ("temp_obc", 1.0),
        "beacon_payload_batt_a_temp": ("temp_batt_a", 1.0),
        "beacon_payload_batt_b_temp": ("temp_batt_b", 1.0),
        "beacon_payload_panel_pos_z_temp": ("temp_panel_z", 1.0),
        "beacon_payload_uptime": ("uptime", 1.0),
    }

    def decode(self, payload: bytes) -> Optional[Dict[str, Any]]:
        return self.decode_with_diagnostics(payload).data

    def decode_with_diagnostics(self, payload: bytes) -> StageOutcome:
        """
        Parse raw bytes using the UWE-4 Kaitai struct.

        Returns all fields exactly as satnogs-decoders produces them
        (AX.25 header fields + beacon payload fields). These are written
        to data/interim/ without modification.
        """
        try:
            struct = Uwe4.from_bytes(payload)
        except Exception as exc:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="decode",
                    code="kaitai_parse_error",
                    message=str(exc),
                )
            )

        try:
            data = dec.get_fields(struct)
        except Exception as exc:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="decode",
                    code="field_extraction_error",
                    message=str(exc),
                )
            )

        if not data:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="decode",
                    code="empty_decode",
                    message="Decoder returned an empty field map.",
                )
            )

        missing_fields = sorted(self.REQUIRED_FIELDS - data.keys())
        if missing_fields:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="decode",
                    code="missing_required_fields",
                    message=", ".join(missing_fields),
                )
            )

        return StageOutcome(data=data)

    def adapt(self, decoded: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.adapt_with_diagnostics(decoded).data

    def adapt_with_diagnostics(self, decoded: Dict[str, Any]) -> StageOutcome:
        """
        Map raw UWE-4 Kaitai fields to SI-unit Golden Features.

        Conversions:
          - beacon_payload_batt_{a,b}_voltage (mV) → batt_{a,b}_voltage (V)
          - beacon_payload_batt_{a,b}_current (mA) → batt_{a,b}_current (A)
          - beacon_payload_batt_{a,b}_temp (°C)    → temp_batt_{a,b} (°C)
          - beacon_payload_obc_temp (°C)            → temp_obc (°C)
          - beacon_payload_panel_pos_z_temp (°C)    → temp_panel_z (°C)
          - beacon_payload_power_consumption (mW)   → power_consumption (W)
          - beacon_payload_uptime (s)               → uptime (s)

        Derived fields:
          - batt_voltage = mean(batt_a_voltage, batt_b_voltage)
          - batt_current = sum(batt_a_current, batt_b_current)
        """
        try:
            return StageOutcome(data=self._adapt_payload(decoded))
        except ValueError as exc:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="adapt",
                    code="invalid_numeric_value",
                    message=str(exc),
                )
            )
        except Exception as exc:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="adapt",
                    code="adapt_exception",
                    message=str(exc),
                )
            )

    def _adapt_payload(self, decoded: Dict[str, Any]) -> Dict[str, Any]:
        get = decoded.get
        adapted: Dict[str, Any] = {
            "src_callsign": get("src_callsign"),
            "dest_callsign": get("dest_callsign"),
        }
        missing_raw_fields = []

        for raw_field, (target_field, scale) in self.NUMERIC_FIELD_MAP.items():
            converted = self._scaled_optional_number(decoded, raw_field, scale)
            if converted is None:
                missing_raw_fields.append(raw_field)
            adapted[target_field] = converted

        adapted["batt_voltage"] = self._mean_if_complete(
            adapted["batt_a_voltage"],
            adapted["batt_b_voltage"],
        )
        adapted["batt_current"] = self._sum_if_complete(
            adapted["batt_a_current"],
            adapted["batt_b_current"],
        )
        if adapted["uptime"] is not None:
            adapted["uptime"] = int(adapted["uptime"])
        adapted["missing_raw_fields"] = "|".join(missing_raw_fields) if missing_raw_fields else None
        adapted["missing_raw_field_count"] = len(missing_raw_fields)
        adapted["frame_is_complete"] = len(missing_raw_fields) == 0
        return adapted

    @staticmethod
    def _mean_if_complete(left: Optional[float], right: Optional[float]) -> Optional[float]:
        if left is None or right is None:
            return None
        return (left + right) / 2.0

    @staticmethod
    def _sum_if_complete(left: Optional[float], right: Optional[float]) -> Optional[float]:
        if left is None or right is None:
            return None
        return left + right

    @staticmethod
    def _scaled_optional_number(
        decoded: Dict[str, Any],
        field_name: str,
        scale: float,
    ) -> Optional[float]:
        value = decoded.get(field_name)
        if value is None:
            return None

        if isinstance(value, float) and math.isnan(value):
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} is not numeric: {value!r}") from exc

        return numeric_value / scale

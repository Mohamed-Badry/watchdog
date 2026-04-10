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

import satnogsdecoders.decoder as dec
from satnogsdecoders.decoder.uwe4 import Uwe4

from gr_sat.telemetry import BaseDecoder, DecoderRegistry


@DecoderRegistry.register(43880)
class UWE4Decoder(BaseDecoder):
    """
    Decoder for UWE-4 (NORAD 43880).

    Stage 1 (decode): Raw AX.25 payload → all Kaitai fields as-is.
    Stage 2 (adapt): Kaitai fields → SI-unit Golden Features.
    """

    # Fields required to validate a successful decode
    REQUIRED_FIELDS = {"beacon_payload_uptime", "beacon_payload_batt_a_voltage"}

    def decode(self, payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse raw bytes using the UWE-4 Kaitai struct.

        Returns all fields exactly as satnogs-decoders produces them
        (AX.25 header fields + beacon payload fields). These are written
        to data/interim/ without modification.
        """
        try:
            struct = Uwe4.from_bytes(payload)
            data = dec.get_fields(struct)

            if not data:
                return None

            # Validate: ensure critical beacon fields are present
            if not self.REQUIRED_FIELDS.issubset(data.keys()):
                return None

            return data

        except Exception:
            return None

    def adapt(self, decoded: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            get = decoded.get

            # --- Individual battery values (mV/mA → V/A) ---
            batt_a_v = get("beacon_payload_batt_a_voltage", 0) / 1000.0
            batt_b_v = get("beacon_payload_batt_b_voltage", 0) / 1000.0
            batt_a_i = get("beacon_payload_batt_a_current", 0) / 1000.0
            batt_b_i = get("beacon_payload_batt_b_current", 0) / 1000.0

            return {
                # Identifiers
                "src_callsign": get("src_callsign"),
                "dest_callsign": get("dest_callsign"),
                # Power (SI units)
                "batt_a_voltage": batt_a_v,
                "batt_b_voltage": batt_b_v,
                "batt_a_current": batt_a_i,
                "batt_b_current": batt_b_i,
                "batt_voltage": (batt_a_v + batt_b_v) / 2.0,
                "batt_current": batt_a_i + batt_b_i,
                "power_consumption": get("beacon_payload_power_consumption", 0) / 1000.0,
                # Thermal (already °C)
                "temp_obc": get("beacon_payload_obc_temp"),
                "temp_batt_a": get("beacon_payload_batt_a_temp"),
                "temp_batt_b": get("beacon_payload_batt_b_temp"),
                "temp_panel_z": get("beacon_payload_panel_pos_z_temp"),
                # Status
                "uptime": get("beacon_payload_uptime"),
            }

        except Exception:
            return None

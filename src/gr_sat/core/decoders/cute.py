"""
CUTE (NORAD 49263) Telemetry Decoder.

Uses the satnogs-decoders Kaitai Struct for binary parsing.
CUTE is a NASA-funded astrophysics CubeSat.
"""

import math
from loguru import logger
import satnogsdecoders.decoder.cute as cute

from gr_sat.core.telemetry import BaseDecoder, DecoderRegistry


@DecoderRegistry.register(49263)
class CuteDecoder(BaseDecoder):
    """
    Decoder for CUTE (NORAD 49263).
    """

    def decode(self, payload: bytes) -> dict[str, float] | None:
        outcome = self.decode_with_diagnostics(payload)
        return outcome.data if outcome.ok else None

    def decode_with_diagnostics(self, payload: bytes):
        import satnogsdecoders.decoder as dec
        from gr_sat.core.telemetry import StageOutcome, ProcessingFailure
        try:
            struct = cute.Cute.from_bytes(payload)
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
                    message="No fields extracted from Kaitai struct.",
                )
            )
        return StageOutcome(data=data)

    def adapt(self, decoded: dict) -> dict[str, float] | None:
        """
        Map the Kaitai struct fields to the unified numerical feature array.
        """
        if not decoded:
            return None

        # The data we want is scattered in various SOH payloads. We must search for it.
        # get_fields flattens the hierarchy, so we can just grab them by name if they exist.
        features = {}
        
        def get_val(key, scale=1.0, is_zynq=False):
            val = decoded.get(key)
            if val is None:
                return None
            try:
                num = float(val)
                if is_zynq:
                    return ((num * 503.975 / 65536.0) - 273.15) if num > 0 else 0.0
                return num * scale
            except (TypeError, ValueError):
                return None

        # Check for CUTE specific flat keys
        if "soh_analogs_battery_voltage" in decoded:
            features["batt_voltage"] = get_val("soh_analogs_battery_voltage", 0.001)
            features["batt_current"] = get_val("soh_analogs_battery_current", 0.001)
            
            v = features["batt_voltage"]
            c = features["batt_current"]
            features["power_consumption"] = (v * abs(c)) if (v is not None and c is not None) else None
            
            features["temp_batt_a"] = get_val("soh_analogs_box1_temp", 0.01)
            features["temp_obc"] = get_val("pld_sw_stat_zynq_temp", is_zynq=True)
            features["temp_panel_z"] = get_val("soh_radio_sdr_tx_temp")
        
        # Check for other variants
        elif "hk_batt_v" in decoded:
            features["batt_voltage"] = get_val("hk_batt_v")
            features["batt_current"] = get_val("hk_batt_i")
            features["power_consumption"] = get_val("hk_bus_v")
            features["temp_batt_a"] = get_val("hk_temp_1")
            features["temp_obc"] = get_val("hk_temp_2")
            features["temp_panel_z"] = get_val("hk_temp_3")
            
        if not features:
            return None

        # Filter out packets that don't actually contain the core telemetry we need
        if features.get("batt_voltage") is None or features["batt_voltage"] == 0:
            return None

        # Replace NaNs/Infs which crash ML models
        for k, v in features.items():
            if v is not None and (math.isnan(v) or math.isinf(v)):
                features[k] = None

        features["frame_is_complete"] = all(
            features.get(f) is not None
            for f in ["batt_voltage", "batt_current", "temp_obc", "temp_batt_a", "temp_panel_z"]
        )

        return features

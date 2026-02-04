from construct import Struct, Int16ub, Int8ub, Int16sb, Float32b, Pass
from gr_sat.telemetry import TelemetryFrame, DecoderRegistry, BaseDecoder
from typing import Dict, Any, Optional

# --- GO-32 (TechSat-1B) Telemetry Definition ---
# Note: This is a REVERSE ENGINEERED / APPROXIMATE structure for demonstration.
# In a real mission, this would be matched exactly to the Interface Control Doc (ICD).
#
# Assumed Structure for Demo:
# - Header: CallSigns (handled by AX.25, we get the payload)
# - Payload:
#   - Sync/ID (2 bytes)
#   - Mode (1 byte)
#   - Battery Voltage (ADC count, 16-bit)
#   - Battery Current (ADC count, 16-bit)
#   - Solar Current (ADC count, 16-bit)
#   - OBC Temp (ADC count, 16-bit)
#   - PA Temp (ADC count, 16-bit)
#   - RSSI (Int8)
#   - Tumble Rate (Float32)

GO32_Struct = Struct(
    "sync" / Int16ub,
    "mode_code" / Int8ub,
    "batt_v_adc" / Int16ub,
    "batt_i_adc" / Int16sb,
    "solar_i_adc" / Int16ub,
    "temp_obc_adc" / Int16sb,
    "temp_pa_adc" / Int16sb,
    "rssi_raw" / Int8ub,
    "spin_rate" / Float32b,
)

@DecoderRegistry.register(25397)
class GO32Decoder(BaseDecoder):
    """
    Decoder for GO-32 (TechSat-1B).
    NORAD ID: 25397
    """
    
    def decode(self, payload: bytes) -> Dict[str, Any]:
        try:
            # We need a minimum length
            if len(payload) < 16:
                return {}

            # Parse
            # We skip the first few bytes if they are AX.25 PID (0xF0) or similar
            # For this demo, we assume the payload passed is exactly the struct.
            # In reality, might need to slice: payload[1:]
            
            # Construct can handle extra bytes at end, but fails if too short.
            data = GO32_Struct.parse(payload)
            
            # --- Calibration / Normalization ---
            # Converting ADC counts to SI Units
            
            # Voltage: 0-1023 -> 0-15V (Example)
            batt_voltage = (data.batt_v_adc / 1024.0) * 15.0
            
            # Current: +/- 1023 -> +/- 2A
            batt_current = (data.batt_i_adc / 1024.0) * 2.0
            
            # Solar: 0-1023 -> 0-5A
            solar_power = ((data.solar_i_adc / 1024.0) * 5.0) * batt_voltage # W ~ V*I
            
            # Temps: ADC to Celsius (Linear NTC approx)
            temp_obc = (data.temp_obc_adc / 10.0)
            temp_pa = (data.temp_pa_adc / 10.0)
            
            # Mode Map
            mode_map = {0: "startup", 1: "nominal", 2: "safe", 3: "transmit"}
            mode = mode_map.get(data.mode_code, "unknown")

            return {
                "batt_voltage": round(batt_voltage, 2),
                "batt_current": round(batt_current, 3),
                "solar_power": round(solar_power, 2),
                "temp_obc": round(temp_obc, 1),
                "temp_pa": round(temp_pa, 1),
                "signal_rssi": float(data.rssi_raw) * -1.0, # usually negative
                "tumble_rate": round(data.spin_rate, 4),
                "mode": mode
            }
            
        except Exception:
            # Struct parsing error (not enough bytes, invalid format)
            return {}


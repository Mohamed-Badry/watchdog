import json
import pandas as pd
from pathlib import Path
from kaitaistruct import KaitaiStream, BytesIO
import sys
from loguru import logger

# Ensure scripts dir is in path to import bugsat1_no_val
sys.path.append("scripts")
from bugsat1_no_val import Bugsat1 
from satnogsdecoders.decoder.ax25frames import Ax25frames

def kaitai_to_dict(struct):
    if isinstance(struct, (int, float, str, bool)) or struct is None:
        return struct
    if isinstance(struct, bytes):
        return struct.hex()
    if isinstance(struct, list):
        return [kaitai_to_dict(item) for item in struct]
    if hasattr(struct, "__dict__"):
        result = {}
        for key in dir(struct):
            if key.startswith("_"): continue
            value = getattr(struct, key)
            if callable(value): continue
            result[key] = kaitai_to_dict(value)
        return result
    return str(struct)

def decode_bugsat_telemetry(hex_str):
    try:
        raw = bytes.fromhex(hex_str)
        if len(raw) < 20: return None
        
        # 1. Parse AX.25
        ax_struct = Ax25frames.from_bytes(raw)
        if not hasattr(ax_struct.ax25_frame, "payload") or not hasattr(ax_struct.ax25_frame.payload, "ax25_info"):
            return None
        payload = ax_struct.ax25_frame.payload.ax25_info
        if not payload: return None
        
        # 2. Parse Bugsat1 from the payload
        info_io = BytesIO(payload)
        info_struct = Bugsat1.Ax25InfoData(KaitaiStream(info_io))
        
        # 3. Extract fields from Telemetry
        if hasattr(info_struct, "beacon_type"):
            telem = info_struct.beacon_type
            data = kaitai_to_dict(telem)
            
            # Calibration (assume mv -> v and amps -> a)
            if "nice_battery_mv" in data:
                data["batt_voltage"] = data["nice_battery_mv"] / 1000.0
            if "battery_amps" in data:
                data["batt_current"] = data["battery_amps"] / 1000.0
            if "cpu_c" in data:
                data["temp_obc"] = data["cpu_c"]
            if "temperature_imu_c" in data:
                data["temp_pa"] = data["temperature_imu_c"]
            if "uptime_s" in data:
                data["uptime"] = data["uptime_s"]
                
            # Callsigns
            data["src_callsign"] = ax_struct.ax25_frame.ax25_header.src_callsign_raw.callsign_ror.callsign.strip()
            data["dest_callsign"] = ax_struct.ax25_frame.ax25_header.dest_callsign_raw.callsign_ror.callsign.strip()
            
            return data
            
    except Exception:
        return None
    return None

def process_40014_data(raw_dir="data/raw/40014", output_file="data/processed/40014_processed.csv"):
    raw_dir = Path(raw_dir)
    all_decoded = []
    
    files = sorted(raw_dir.glob("*.jsonl"))
    logger.info(f"Processing BugSat-1 data in {raw_dir}")
    
    for file in files:
        with open(file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    res = decode_bugsat_telemetry(data.get("frame", ""))
                    if res:
                        res["timestamp"] = data.get("timestamp")
                        res["observation_id"] = data.get("observation_id")
                        all_records_appended = all_decoded.append(res)
                except Exception:
                    continue
                    
    if all_decoded:
        df = pd.DataFrame(all_decoded)
        df = df.dropna(axis=1, how="all")
        
        # Sort and Save
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        
        df.to_csv(output_file, index=False)
        logger.success(f"Successfully processed {len(df)} frames. Saved to {output_file}")
        
        # Standardized View
        standard_cols = ["timestamp", "batt_voltage", "batt_current", "temp_obc", "temp_pa", "uptime"]
        available_std = [c for c in standard_cols if c in df.columns]
        if available_std:
            print(f"\nStandardized Telemetry Samples:\n{df[available_std].head(10)}")
    else:
        logger.warning("No BugSat-1 telemetry frames were successfully decoded.")

if __name__ == "__main__":
    process_40014_data()

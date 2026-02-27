import json
import pandas as pd
from pathlib import Path
from kaitaistruct import KaitaiStream, BytesIO
import satnogsdecoders.decoder as dec
from satnogsdecoders.decoder.uwe4 import Uwe4
from loguru import logger

def decode_uwe4_telemetry(hex_str):
    try:
        raw = bytes.fromhex(hex_str)
        struct = Uwe4.from_bytes(raw)
        data = dec.get_fields(struct)
        
        if not data or 'beacon_payload_uptime' not in data:
            return None
            
        # Map to our "Golden Features" (SI Units)
        normalized = {
            "src_callsign": data.get("src_callsign"),
            "dest_callsign": data.get("dest_callsign"),
            "uptime": data.get("beacon_payload_uptime"),
            
            # Power System (mV -> V, mA -> A)
            "batt_a_voltage": data.get("beacon_payload_batt_a_voltage", 0) / 1000.0,
            "batt_b_voltage": data.get("beacon_payload_batt_b_voltage", 0) / 1000.0,
            "batt_a_current": data.get("beacon_payload_batt_a_current", 0) / 1000.0,
            "batt_b_current": data.get("beacon_payload_batt_b_current", 0) / 1000.0,
            "power_consumption": data.get("beacon_payload_power_consumption", 0) / 1000.0, # mA to A
            
            # Thermal System (°C)
            "temp_obc": data.get("beacon_payload_obc_temp"),
            "temp_batt_a": data.get("beacon_payload_batt_a_temp"),
            "temp_batt_b": data.get("beacon_payload_batt_b_temp"),
            "temp_panel_z": data.get("beacon_payload_panel_pos_z_temp"),
        }
        
        # Aggregate logic: Average voltage, total current
        normalized["batt_voltage"] = (normalized["batt_a_voltage"] + normalized["batt_b_voltage"]) / 2.0
        normalized["batt_current"] = normalized["batt_a_current"] + normalized["batt_b_current"]
        
        return normalized
        
    except Exception:
        return None

def process_uwe4_data(raw_dir="data/raw/43880", output_file="data/processed/43880.csv"):
    raw_dir = Path(raw_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        logger.error(f"Directory {raw_dir} does not exist. Please fetch UWE-4 data first.")
        return
        
    all_decoded = []
    files = sorted(raw_dir.glob("*.jsonl"))
    logger.info(f"Processing UWE-4 data in {raw_dir} ({len(files)} files)")
    
    for file in files:
        with open(file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    res = decode_uwe4_telemetry(data.get("frame", ""))
                    if res:
                        res["timestamp"] = data.get("timestamp")
                        res["observation_id"] = data.get("observation_id")
                        all_decoded.append(res)
                except Exception:
                    continue
                    
    if all_decoded:
        df = pd.DataFrame(all_decoded)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        
        df.to_csv(output_file, index=False)
        logger.success(f"Successfully processed {len(df)} frames. Saved to {output_file}")
        
        print("\nGolden Features (First 5 Rows):")
        cols = ["timestamp", "batt_voltage", "batt_current", "temp_obc", "temp_batt_a", "uptime"]
        print(df[cols].head(5).to_string(index=False))
        
    else:
        logger.warning("No valid UWE-4 frames found.")

if __name__ == "__main__":
    process_uwe4_data()

import json
import pandas as pd
from pathlib import Path
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
            
        return data
        
    except Exception:
        return None

def decode_uwe4_data(raw_dir="data/raw/43880", output_file="data/interim/43880.csv"):
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
        
        print("\nRaw Telemetry (First 5 Rows, up to 10 columns):")
        print(df.iloc[:, :10].head(5).to_string(index=False))
        
    else:
        logger.warning("No valid UWE-4 frames found.")

if __name__ == "__main__":
    decode_uwe4_data()

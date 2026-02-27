import json
import pandas as pd
from pathlib import Path
from kaitaistruct import KaitaiStream, BytesIO
import sys
from loguru import logger

# Add scripts dir to path to import bugsat1_no_val
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

def decode_frame(hex_str):
    try:
        raw = bytes.fromhex(hex_str)
        if len(raw) < 20: return None
        
        # 1. Parse AX.25
        ax_struct = Ax25frames.from_bytes(raw)
        # Check if it has payload
        if not hasattr(ax_struct.ax25_frame, "payload") or not hasattr(ax_struct.ax25_frame.payload, "ax25_info"):
            return None
        payload = ax_struct.ax25_frame.payload.ax25_info
        
        if not payload: return None
        
        # 2. Parse Bugsat1 from the payload
        info_io = BytesIO(payload)
        info_struct = Bugsat1.Ax25InfoData(KaitaiStream(info_io))
        
        # 3. Extract fields
        if hasattr(info_struct, "beacon_type"):
            telem = info_struct.beacon_type
            data = kaitai_to_dict(telem)
            
            # Add some AX.25 metadata
            data["src_callsign"] = ax_struct.ax25_frame.ax25_header.src_callsign_raw.callsign_ror.callsign.strip()
            data["dest_callsign"] = ax_struct.ax25_frame.ax25_header.dest_callsign_raw.callsign_ror.callsign.strip()
            return data
            
    except Exception:
        return None
    return None

def process_all_files(raw_dir, output_file):
    raw_dir = Path(raw_dir)
    all_records = []
    
    files = sorted(raw_dir.glob("*.jsonl"))
    logger.info(f"Processing {len(files)} files...")
    
    for file in files:
        with open(file, "r") as f:
            for line in f:
                try:
                    raw_data = json.loads(line)
                    res = decode_frame(raw_data.get("frame", ""))
                    if res:
                        res["timestamp"] = raw_data.get("timestamp")
                        res["observation_id"] = raw_data.get("observation_id")
                        all_records.append(res)
                except Exception:
                    continue
                    
    if all_records:
        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        
        df.to_csv(output_file, index=False)
        logger.success(f"Saved {len(all_records)} decoded frames to {output_file}")
        
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if not numeric_cols.empty:
            print("\nDecoded Telemetry Summary:")
            print(df[numeric_cols].describe().T[["count", "mean", "min", "max"]].head(20))
    else:
        logger.warning("No frames decoded.")

if __name__ == "__main__":
    process_all_files("data/raw/40014", "data/processed/40014_bugsat_custom.csv")

import json
import pandas as pd
from pathlib import Path
from kaitaistruct import KaitaiStream, BytesIO
from satnogsdecoders.decoder.bugsat1 import Bugsat1
from satnogsdecoders.decoder import get_fields
from loguru import logger
import warnings

# Suppress Kaitai/Struct warnings if any
warnings.filterwarnings("ignore")

def decode_bugsat_frame(hex_frame):
    try:
        raw_bytes = bytes.fromhex(hex_frame)
        io = BytesIO(raw_bytes)
        struct = Bugsat1(KaitaiStream(io))
        
        # Extract fields using satnogsdecoders utility
        fields = get_fields(struct)
        return fields
    except Exception:
        return None

def process_bugsat_data(raw_dir, output_file):
    raw_dir = Path(raw_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_decoded = []
    
    files = sorted(raw_dir.glob("*.jsonl"))
    logger.info(f"Processing {len(files)} files in {raw_dir}")
    
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    frame_hex = data.get('frame')
                    if not frame_hex:
                        continue
                        
                    decoded = decode_bugsat_frame(frame_hex)
                    if decoded:
                        decoded['timestamp'] = data.get('timestamp')
                        decoded['observation_id'] = data.get('observation_id')
                        all_decoded.append(decoded)
                except Exception:
                    continue
                    
    if all_decoded:
        df = pd.DataFrame(all_decoded)
        # Handle cases where some fields might be nested or messy
        # For simplicity, keep it as is for now.
        
        df.to_csv(output_file, index=False)
        logger.success(f"Processed {len(all_decoded)} frames. Saved to {output_file}")
        
        # Display summary of numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            print("\nDecoded Data Summary (Numeric Fields):")
            print(df[numeric_cols].describe().T[['count', 'mean', 'min', 'max']].head(20))
    else:
        logger.warning("No frames were successfully decoded. Check if frames are valid AX.25 BugSat-1 beacons.")

if __name__ == "__main__":
    process_bugsat_data("data/raw/40014", "data/processed/40014_decoded.csv")

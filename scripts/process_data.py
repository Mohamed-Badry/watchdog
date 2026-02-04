import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from loguru import logger
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Import the core logic
from gr_sat.telemetry import process_frame, TelemetryFrame
# Import decoders to trigger registration
import gr_sat.decoders 

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Configure Logger
logger.configure(handlers=[
    {"sink": RichHandler(show_time=False), "format": "{message}"}
])

def load_jsonl(filepath: Path) -> List[Dict]:
    """Reads a JSONL file and returns a list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def process_satellite(norad_id: str):
    """
    Processes all raw data for a specific satellite and saves a Parquet file.
    """
    sat_dir = RAW_DIR / str(norad_id)
    if not sat_dir.exists():
        logger.warning(f"No raw data found for NORAD ID {norad_id}")
        return

    json_files = sorted(list(sat_dir.glob("*.jsonl")))
    if not json_files:
        logger.warning(f"No .jsonl files found in {sat_dir}")
        return

    valid_frames = []
    total_frames = 0
    decoded_count = 0

    logger.info(f"Processing {len(json_files)} files for Satellite {norad_id}...")

    for jp in json_files:
        raw_records = load_jsonl(jp)
        total_frames += len(raw_records)
        
        for record in raw_records:
            # SatNOGS API usually provides 'frame' as a hex string
            hex_payload = record.get('frame')
            timestamp_str = record.get('timestamp')
            
            if not hex_payload or not timestamp_str:
                continue
                
            try:
                # Convert hex to bytes
                payload_bytes = bytes.fromhex(hex_payload)
                
                # Parse timestamp
                # Format: "2024-01-01T12:00:00Z"
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                
                # Use the Shared Core to process
                tf = process_frame(
                    norad_id=int(norad_id), 
                    payload=payload_bytes, 
                    source="satnogs_db", 
                    timestamp=ts
                )
                
                if tf:
                    valid_frames.append(tf.to_dict())
                    decoded_count += 1
                    
            except Exception as e:
                # logger.debug(f"Frame error: {e}")
                continue

    if not valid_frames:
        logger.warning(f"No valid frames decoded for {norad_id} out of {total_frames} raw frames.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(valid_frames)
    
    # Deduplicate (by timestamp)
    initial_len = len(df)
    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    dedup_len = len(df)
    
    # Sort
    df.sort_values('timestamp', inplace=True)
    
    # Save to CSV
    out_file = PROCESSED_DIR / f"{norad_id}.csv"
    df.to_csv(out_file, index=False)
    
    logger.success(f"Saved {dedup_len} frames to {out_file} (discarded {initial_len - dedup_len} dupes). Decode Rate: {decoded_count}/{total_frames} ({decoded_count/total_frames:.1%})")

def main():
    parser = argparse.ArgumentParser(description="Telemetry Processor")
    parser.add_argument("--norad", type=str, help="Specific NORAD ID to process")
    parser.add_argument("--all", action="store_true", help="Process all available satellites")
    
    args = parser.parse_args()
    
    if args.norad:
        process_satellite(args.norad)
    elif args.all:
        # List all directories in data/raw
        sat_dirs = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
        for sat_id in sat_dirs:
            process_satellite(sat_id)
    else:
        logger.info("Please specify --norad <ID> or --all")

if __name__ == "__main__":
    main()

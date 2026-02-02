import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_frequencies(downlink_str):
    if pd.isna(downlink_str):
        return []
    # Find all patterns that look like frequencies (e.g., 437.220)
    # This ignores the range syntax "-" for simplicity and just grabs the numbers
    matches = re.findall(r'(\d{3}\.\d+)', str(downlink_str))
    return [float(m) for m in matches]

def is_target_freq(freqs):
    # Check if any frequency is between 433 and 438
    for f in freqs:
        if 433.0 <= f <= 438.0:
            return True
    return False

def clean_mode(mode_str):
    if pd.isna(mode_str):
        return "Unknown"
    
    # Normalize
    m = str(mode_str).lower()
    
    # Common replacements to group things
    if '9k6' in m or '9600' in m:
        rate = '9k6'
    elif '1k2' in m or '1200' in m:
        rate = '1k2'
    elif '4k8' in m or '4800' in m:
        rate = '4k8'
    elif '19k2' in m or '19200' in m:
        rate = '19k2'
    else:
        rate = 'other_rate'
        
    if 'gmsk' in m:
        mod = 'gmsk'
    elif 'gfsk' in m:
        mod = 'gfsk'
    elif 'afsk' in m:
        mod = 'afsk'
    elif 'fsk' in m:
        mod = 'fsk'
    elif 'bpsk' in m:
        mod = 'bpsk'
    elif 'cw' in m and len(m) < 5: # Just CW
        mod = 'cw'
    else:
        mod = 'other_mod'
        
    return f"{rate}_{mod}"

def main():
    df = pd.read_csv('data/amsat-active-frequencies.csv')
    
    # 1. Filter by Frequency
    df['freqs'] = df['downlink'].apply(parse_frequencies)
    df['in_band'] = df['freqs'].apply(is_target_freq)
    
    target_sats = df[df['in_band']].copy()
    
    print(f"Total Active Satellites: {len(df)}")
    print(f"Satellites in 433-438 MHz: {len(target_sats)}")
    
    # 2. Analyze Modes
    target_sats['clean_mode'] = target_sats['mode'].apply(clean_mode)
    
    mode_counts = target_sats['clean_mode'].value_counts()
    
    print("\n--- Top Modulation/Rates in Band ---")
    print(mode_counts.head(10))
    
    # 3. Recommend
    top_mode = mode_counts.idxmax()
    print(f"\nRecommended Target Category: {top_mode}")
    
    candidates = target_sats[target_sats['clean_mode'] == top_mode]
    print(f"Found {len(candidates)} candidates for {top_mode}. Examples:")
    print(candidates[['name', 'downlink', 'mode']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Config
PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="deep", rc={
    'axes.facecolor': '#fafafa',
    'figure.facecolor': '#fafafa'
})

def plot_satellite(norad_id: str):
    csv_path = PROCESSED_DIR / f"{norad_id}.csv"
    if not csv_path.exists():
        print(f"No processed data found at {csv_path}")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort
    df.sort_values('timestamp', inplace=True)
    
    print(f"Data Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(df[['batt_voltage', 'batt_current', 'temp_obc']].describe())

    # Create Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Power
    sns.lineplot(data=df, x='timestamp', y='batt_voltage', ax=axes[0], color='tab:blue', label='Voltage (V)')
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title(f"UWE-4 (NORAD {norad_id}) - Power System")
    axes[0].legend(loc='upper right')
    
    # 2. Current
    sns.lineplot(data=df, x='timestamp', y='batt_current', ax=axes[1], color='tab:orange', label='Current (A)')
    axes[1].set_ylabel("Current (A)")
    axes[1].legend(loc='upper right')
    
    # 3. Thermal
    sns.lineplot(data=df, x='timestamp', y='temp_obc', ax=axes[2], color='tab:red', label='OBC Temp (°C)')
    if 'temp_batt_a' in df.columns:
        sns.lineplot(data=df, x='timestamp', y='temp_batt_a', ax=axes[2], color='tab:purple', label='Batt A Temp (°C)')
    
    axes[2].set_ylabel("Temp (°C)")
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    
    out_img = FIG_DIR / f"telemetry_{norad_id}.png"
    plt.savefig(out_img, dpi=150)
    print(f"Saved plot to {out_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norad", type=str, default="43880")
    args = parser.parse_args()
    
    plot_satellite(args.norad)

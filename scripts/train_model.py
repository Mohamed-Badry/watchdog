"""
Train Model — The Lab

Trains the Unified Anomaly Detection system (Variational Autoencoder) for a single satellite:
  1. Filters out known physical extremes to train on a "pure" baseline.
  2. Trains StandardScaler.
  3. Trains PyTorch TelemetryVAE (Unified Detector + Diagnoser).
  4. Saves models to models/{norad_id}_scaler.pkl and models/{norad_id}_vae.pt

Usage:
  pixi run python scripts/train_model.py --norad 43880
"""

import argparse
import pandas as pd
from pathlib import Path
import joblib

from loguru import logger
from rich.logging import RichHandler
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from gr_sat.models import TelemetryVAE, vae_loss

logger.configure(handlers=[
    {"sink": RichHandler(show_time=False, markup=True), "format": "{message}"}
])

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")

BASE_FEATURES = [
    "batt_voltage", 
    "batt_current", 
    "temp_batt_a", 
    "temp_batt_b", 
    "temp_panel_z"
]
ALL_FEATURES = BASE_FEATURES

def train_vae(X_train_scaled, norad_id: str):
    logger.info("Training PyTorch Variational Autoencoder (VAE)...")
    
    # 1. Prepare PyTorch Dataset
    X_tensor = torch.FloatTensor(X_train_scaled)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Init Model
    vae = TelemetryVAE(input_dim=len(ALL_FEATURES), hidden_dim=12, latent_dim=3)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # 3. Training Loop
    epochs = 100
    vae.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar, kld_weight=0.05)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 25 == 0:
            logger.debug(f"VAE Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
    # 4. Save
    model_path = MODELS_DIR / f"{norad_id}_vae.pt"
    torch.save(vae.state_dict(), model_path)
    logger.success(f"Saved PyTorch VAE → {model_path}")

def train_for_satellite(norad_id: str):
    data_path = PROCESSED_DIR / f"{norad_id}.csv"
    if not data_path.exists():
        logger.error(f"Processed data not found at {data_path}")
        return

    logger.info(f"Loading data for NORAD {norad_id}...")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    
    orig_len = len(df)
    
    extreme_mask = (df["batt_voltage"] > 5.0) | (df["batt_current"].abs() > 1.0)
    df_clean = df[~extreme_mask].copy()
    
    train_size = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:train_size].copy()
    
    X_train = df_train[ALL_FEATURES].values
    
    logger.info(f"Cleaned extreme rows: {extreme_mask.sum()}")
    logger.info(f"Training on {len(X_train)} frames ({len(X_train) / orig_len * 100:.1f}%)")

    # Train Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, MODELS_DIR / f"{norad_id}_scaler.pkl")

    # Train PyTorch VAE natively
    train_vae(X_train_scaled, norad_id)

    logger.info("[bold green]Training pipeline completed successfully![/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norad", type=str, required=True, help="Specific NORAD ID to process")
    args = parser.parse_args()
    train_for_satellite(args.norad)

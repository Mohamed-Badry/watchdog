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
import numpy as np

from loguru import logger
from rich.logging import RichHandler
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from gr_sat.ml_config import (
    ALL_FEATURES,
    BASE_FEATURES,
    DEFAULT_INFERENCE_MODE,
    DEFAULT_KLD_WEIGHT,
    HIDDEN_DIM,
    LATENT_DIM,
    THRESHOLD_PERCENTILE,
)
from gr_sat.model_artifacts import (
    ModelArtifactMetadata,
    model_artifact_paths,
    save_model_metadata,
    split_chronological,
    threshold_from_scores,
)
from gr_sat.models import TelemetryVAE, compute_anomaly_scores, vae_loss

logger.configure(handlers=[
    {"sink": RichHandler(show_time=False, markup=True), "format": "{message}"}
])

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")


def score_scaled_frames(vae: TelemetryVAE, X_scaled, kld_weight=DEFAULT_KLD_WEIGHT) -> np.ndarray:
    X_tensor = torch.FloatTensor(X_scaled)
    vae.eval()
    with torch.no_grad():
        recon_x, mu, logvar = vae(X_tensor)
        scores = compute_anomaly_scores(
            recon_x,
            X_tensor,
            mu,
            logvar,
            kld_weight=kld_weight,
        )
    return scores.numpy()


def train_vae(X_train_scaled, norad_id: str, epochs: int = 100):
    logger.info("Training PyTorch Variational Autoencoder (VAE)...")
    
    # 1. Prepare PyTorch Dataset
    X_tensor = torch.FloatTensor(X_train_scaled)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Init Model
    vae = TelemetryVAE(
        input_dim=len(ALL_FEATURES),
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # 3. Training Loop
    vae.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(
                recon_x,
                x,
                mu,
                logvar,
                kld_weight=DEFAULT_KLD_WEIGHT,
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 25 == 0:
            logger.debug(f"VAE Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
    return vae


def train_for_satellite(norad_id: str, epochs: int = 100):
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
    
    split = split_chronological(df_clean)
    df_train = split.train
    df_validation = split.validation
    df_test = split.test

    X_train = df_train[ALL_FEATURES].values
    X_validation = df_validation[ALL_FEATURES].values
    
    logger.info(f"Cleaned extreme rows: {extreme_mask.sum()}")
    logger.info(
        "Chronological split: "
        f"train={len(df_train)} | validation={len(df_validation)} | test={len(df_test)}"
    )
    logger.info(f"Training on {len(X_train)} frames ({len(X_train) / orig_len * 100:.1f}%)")

    # Train Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    artifact_paths = model_artifact_paths(MODELS_DIR, norad_id)
    joblib.dump(scaler, artifact_paths.scaler)

    # Train PyTorch VAE natively
    vae = train_vae(X_train_scaled, norad_id, epochs=epochs)
    validation_scores = score_scaled_frames(vae, X_validation_scaled)
    threshold = threshold_from_scores(validation_scores, THRESHOLD_PERCENTILE)

    torch.save(vae.state_dict(), artifact_paths.weights)
    logger.success(f"Saved PyTorch VAE → {artifact_paths.weights}")

    metadata = ModelArtifactMetadata.from_split(
        norad_id=norad_id,
        split=split,
        threshold=threshold,
        feature_names=ALL_FEATURES,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        kld_weight=DEFAULT_KLD_WEIGHT,
        threshold_percentile=THRESHOLD_PERCENTILE,
        inference_mode=DEFAULT_INFERENCE_MODE,
    )
    save_model_metadata(artifact_paths.metadata, metadata)
    logger.success(
        "Saved artifact metadata → "
        f"{artifact_paths.metadata} (threshold={threshold:.6f}, mode={metadata.inference_mode})"
    )

    logger.info("[bold green]Training pipeline completed successfully![/bold green]")
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norad", type=str, required=True, help="Specific NORAD ID to process")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    args = parser.parse_args()
    train_for_satellite(args.norad, epochs=args.epochs)

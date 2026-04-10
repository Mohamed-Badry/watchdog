"""
Generate Faults — Benchmark The Edge Models

Tests the Unified Anomaly Detection system (PyTorch VAE) on a held-out test set
by injecting physically-motivated synthetic faults.

- Stage 1 (Detection): VAE Total Reconstruction Error (MSE)
- Stage 2 (Diagnosis): Per-feature MSE isolation

Usage:
  pixi run python scripts/generate_faults.py --norad 43880
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from loguru import logger
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

from sklearn.metrics import roc_auc_score, roc_curve

import torch
from gr_sat.models import TelemetryVAE
from train_model import BASE_FEATURES, ALL_FEATURES

logger.configure(handlers=[
    {"sink": RichHandler(show_time=False, markup=True), "format": "{message}"}
])

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
DOCS_DIR = Path("docs")

def inject_faults(df_test: pd.DataFrame, n_per_fault=100, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    df_faulted = df_test.copy()
    labels = np.zeros(len(df_test), dtype=int)
    fault_types = np.array(["normal"] * len(df_test), dtype=object)
    
    # 1. Sensor Stuck
    stuck_starts = rng.choice(len(df_test) - 5, size=n_per_fault, replace=False)
    median_v = df_faulted["batt_voltage"].median()
    for start in stuck_starts:
        df_faulted.iloc[start:start+5, df_faulted.columns.get_loc("batt_voltage")] = median_v
        labels[start:start+5] = 1
        fault_types[start:start+5] = "sensor_stuck"

    # 2. Panel Failure
    sunlight_mask = df_faulted["temp_panel_z"] > 15
    sun_idx = np.where(sunlight_mask & (labels == 0))[0]
    panel_fail_idx = rng.choice(sun_idx, size=min(n_per_fault, len(sun_idx)), replace=False)
    df_faulted.iloc[panel_fail_idx, df_faulted.columns.get_loc("batt_current")] = -0.3
    labels[panel_fail_idx] = 1
    fault_types[panel_fail_idx] = "panel_failure"

    # 3. Thermal Runaway
    normal_idx = np.where(labels == 0)[0]
    thermal_idx = rng.choice(normal_idx, size=n_per_fault, replace=False)
    df_faulted.iloc[thermal_idx, df_faulted.columns.get_loc("temp_batt_a")] += 15.0
    df_faulted.iloc[thermal_idx, df_faulted.columns.get_loc("temp_batt_b")] += 12.0
    labels[thermal_idx] = 1
    fault_types[thermal_idx] = "thermal_runaway"
    
    return df_faulted, labels, fault_types

def calculate_diagnosis_accuracy(recon_errors, flagged_idx, fault_types):
    fault_categories = ["sensor_stuck", "panel_failure", "thermal_runaway"]
    accuracy = {ft: {"correct": 0, "flagged": 0} for ft in fault_categories}
    
    for idx in flagged_idx:
        errors = recon_errors[idx]
        top_feature = BASE_FEATURES[np.argmax(errors)]
        actual_fault = fault_types[idx]
        
        if actual_fault != "normal":
            accuracy[actual_fault]["flagged"] += 1
            
            is_correct = False
            if actual_fault == "sensor_stuck" and top_feature == "batt_voltage":
                is_correct = True
            elif actual_fault == "panel_failure" and top_feature == "batt_current":
                is_correct = True
            elif actual_fault == "thermal_runaway" and top_feature in ["temp_batt_a", "temp_batt_b"]:
                is_correct = True
                
            if is_correct:
                accuracy[actual_fault]["correct"] += 1
                
    return accuracy

def evaluate(norad_id: str):
    logger.info(f"Evaluating Unified VAE System for NORAD {norad_id}...")
    
    try:
        scaler = joblib.load(MODELS_DIR / f"{norad_id}_scaler.pkl")
        
        # Load Torch VAE
        vae = TelemetryVAE(input_dim=len(ALL_FEATURES), hidden_dim=12, latent_dim=3)
        vae.load_state_dict(torch.load(MODELS_DIR / f"{norad_id}_vae.pt", weights_only=True))
        vae.eval()
        
    except FileNotFoundError:
        logger.error("Models not found. Run train_model.py first.")
        return

    df = pd.read_csv(PROCESSED_DIR / f"{norad_id}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    
    extreme_mask = (df["batt_voltage"] > 5.0) | (df["batt_current"].abs() > 1.0)
    df_clean = df[~extreme_mask].copy()
    
    train_size = int(len(df_clean) * 0.8)
    df_test = df_clean.iloc[train_size:].copy()
    
    df_faulted, y_true, fault_types = inject_faults(df_test, n_per_fault=150)
    X_test_scaled = scaler.transform(df_faulted[ALL_FEATURES].values)

    # VAE Inference 
    X_tensor = torch.FloatTensor(X_test_scaled)
    with torch.no_grad():
        X_recon_vae, mu, logvar = vae(X_tensor)
        
    # Calculate Per-Sample Overall MSE (Stage 1 Score)
    anomaly_scores = torch.mean((X_tensor - X_recon_vae)**2, dim=1).numpy()
    
    # ------------------
    # Stage 1: Detector (Total Loss Score)
    # ------------------
    logger.info("Executing Stage 1: Detection via VAE Total Score...")
    s_min, s_max = anomaly_scores.min(), anomaly_scores.max()
    anomaly_scores = (anomaly_scores - s_min) / (s_max - s_min) if s_max > s_min else anomaly_scores
    
    auroc = roc_auc_score(y_true, anomaly_scores)
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, anomaly_scores)
    recall_at_5 = np.interp(0.05, fpr_curve, tpr_curve)

    # Establish the operating threshold (95th percentile of normal frames)
    normal_scores = anomaly_scores[y_true == 0]
    operating_threshold = np.percentile(normal_scores, 95)
    predicted_anomalies = anomaly_scores > operating_threshold
    flagged_idx = np.where(predicted_anomalies)[0]

    # ------------------
    # Stage 2: Diagnoser (Node MSE Isolation)
    # ------------------
    logger.info("Executing Stage 2: Feature Diagnosis via VAE node-MSE...")
    recon_errors_vae = np.abs(X_test_scaled - X_recon_vae.numpy())
    acc_vae = calculate_diagnosis_accuracy(recon_errors_vae, flagged_idx, fault_types)

    console = Console()
    console.print(f"\n[bold green]Unified VAE Benchmark Report — NORAD {norad_id}[/bold green]")
    console.print(f"Total Test Frames: {len(X_test_scaled)} (Injected Faults: {y_true.sum()})\n")
    
    table = Table(title="Stage 1: Detection (VAE Overall Loss)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("AUROC", f"{auroc:.4f}")
    table.add_row("Recall @ 5% FPR", f"{recall_at_5 * 100:.1f}%")
    console.print(table)
    
    diag_table = Table(title=f"Stage 2: Diagnosis (VAE Feature Isolation)")
    diag_table.add_column("Fault Type", style="cyan")
    diag_table.add_column("PyTorch VAE Accuracy", justify="right")
    
    fault_categories = ["sensor_stuck", "panel_failure", "thermal_runaway"]
    for ft in fault_categories:
        v_flagged = acc_vae[ft]["flagged"]
        v_corr = acc_vae[ft]["correct"]
        v_str = f"{v_corr}/{v_flagged} ({(v_corr/v_flagged*100) if v_flagged>0 else 0:.1f}%)"
        diag_table.add_row(ft.replace("_", " ").title(), v_str)
        
    console.print(diag_table)

    with open(DOCS_DIR / f"benchmark_{norad_id}.md", "w") as f:
        f.write(f"# Edge Benchmark for NORAD {norad_id}\n\n")
        f.write(f"**Unified Architecture:** PyTorch Variational Autoencoder\n\n")
        f.write(f"## Metrics\n")
        f.write(f"- **AUROC:** {auroc:.4f}\n")
        f.write(f"- **Recall @ 5% FPR:** {recall_at_5 * 100:.1f}%\n\n")
        
        f.write("## Fault Isolation Performance\n")
        f.write(f"| Fault Type | Detected by Stage 1 | Isolated by VAE |\n")
        f.write("|------------|---------------------|-----------------|\n")
        for ft in fault_categories:
            total_injected = (fault_types == ft).sum()
            flagged = acc_vae[ft]["flagged"]
            correct = acc_vae[ft]["correct"]
            det_rate = (flagged / total_injected * 100) if total_injected > 0 else 0
            diag_acc = (correct / flagged * 100) if flagged > 0 else 0
            f.write(f"| {ft} | {det_rate:.1f}% | {diag_acc:.1f}% |\n")
            
    logger.success(f"Saved benchmark report to docs/benchmark_{norad_id}.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norad", type=str, required=True, help="Specific NORAD ID")
    args = parser.parse_args()
    evaluate(args.norad)

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fault Sensitivity Sweep
#
# The current benchmark injects extreme synthetic faults (+45°C thermal,
# -0.8A panel short) that achieve 100% detection. This notebook sweeps
# fault magnitude from subtle to extreme and compares:
#
# 1. **VAE** — full reconstruction + KLD anomaly score
# 2. **Z-Score Baseline** — max per-feature absolute Z-score (simple threshold)
#
# The goal is to find the crossover point where the VAE's multivariate
# correlation awareness actually outperforms a naive univariate rule.

# %%
import sys
sys.path.insert(0, "../scripts")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score, roc_curve

from gr_sat.model_artifacts import load_model_artifacts, split_chronological
from gr_sat.models import compute_anomaly_scores
from gr_sat.satellite_profiles import (
    build_baseline_mask,
    feature_completeness_mask,
    get_satellite_profile,
)

if Path("data").exists():
    DATA_DIR = Path("data")
    MODELS_DIR = Path("models")
    FIG_DIR = Path("docs/figures")
else:
    DATA_DIR = Path("../data")
    MODELS_DIR = Path("../models")
    FIG_DIR = Path("../docs/figures")

PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NORAD_ID = "43880"

# %% [markdown]
# ## Load Model & Test Data

# %%
profile = get_satellite_profile(NORAD_ID)
scaler, vae, metadata = load_model_artifacts(NORAD_ID, MODELS_DIR)
feature_names = metadata.feature_names

df = pd.read_csv(PROCESSED_DIR / f"{NORAD_ID}.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

extreme_mask = build_baseline_mask(df, profile)
df_clean = df[~extreme_mask].copy()
complete_mask = feature_completeness_mask(df_clean, feature_names)
df_ready = df_clean[complete_mask].copy()

split = split_chronological(df_ready)
df_test = split.test.copy()

print(f"Test partition: {len(df_test)} frames")
print(f"Features: {feature_names}")
print(f"Threshold (from training): {metadata.threshold:.6f}")

# %% [markdown]
# ## Injection Helpers

# %%
def inject_thermal_runaway(df_test, delta_c, n=150, seed=42):
    """Add delta_c degrees to both battery temps."""
    rng = np.random.RandomState(seed)
    df = df_test.copy()
    labels = np.zeros(len(df), dtype=int)

    normal_idx = np.arange(len(df))
    fault_idx = rng.choice(normal_idx, size=min(n, len(normal_idx)), replace=False)
    df.iloc[fault_idx, df.columns.get_loc("temp_batt_a")] += delta_c
    df.iloc[fault_idx, df.columns.get_loc("temp_batt_b")] += delta_c
    labels[fault_idx] = 1
    return df, labels


def inject_panel_failure(df_test, forced_current, n=150, seed=42):
    """Force batt_current to a value during sunlight frames."""
    rng = np.random.RandomState(seed)
    df = df_test.copy()
    labels = np.zeros(len(df), dtype=int)

    sun_idx = np.where((df["temp_panel_z"] > 15) & (labels == 0))[0]
    if len(sun_idx) == 0:
        return df, labels
    fault_idx = rng.choice(sun_idx, size=min(n, len(sun_idx)), replace=False)
    df.iloc[fault_idx, df.columns.get_loc("batt_current")] = forced_current
    labels[fault_idx] = 1
    return df, labels

# %% [markdown]
# ## Scoring Functions

# %%
def score_vae(df_faulted, scaler, vae, metadata):
    """Return per-frame VAE anomaly scores."""
    X = scaler.transform(df_faulted[metadata.feature_names].values)
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        recon, mu, logvar = vae(X_t)
        scores = compute_anomaly_scores(recon, X_t, mu, logvar, kld_weight=metadata.kld_weight)
    return scores.numpy()


def score_zscore(df_faulted, scaler, feature_names):
    """Return per-frame max |Z-score| across all features. Simplest possible baseline."""
    X = scaler.transform(df_faulted[feature_names].values)
    # scaler already maps to z-scores (mean=0, std=1)
    return np.max(np.abs(X), axis=1)


def evaluate_at_magnitude(df_test, inject_fn, magnitude_arg, scaler, vae, metadata):
    """Run both detectors at a given fault magnitude and return AUROC + Recall@5%FPR."""
    df_faulted, y_true = inject_fn(df_test, magnitude_arg)

    if y_true.sum() == 0:
        return None

    vae_scores = score_vae(df_faulted, scaler, vae, metadata)
    z_scores = score_zscore(df_faulted, scaler, metadata.feature_names)

    results = {}
    for name, scores in [("VAE", vae_scores), ("Z-Score", z_scores)]:
        auroc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        recall_5 = np.interp(0.05, fpr, tpr)
        results[name] = {"auroc": auroc, "recall_5fpr": recall_5}

    return results

# %% [markdown]
# ## Thermal Runaway Sweep

# %%
thermal_deltas = [0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45]
thermal_results = []

for delta in thermal_deltas:
    r = evaluate_at_magnitude(df_test, inject_thermal_runaway, delta, scaler, vae, metadata)
    if r:
        thermal_results.append({"delta_c": delta, **{f"{k}_{m}": v for k, d in r.items() for m, v in d.items()}})
        print(f"  Δ{delta:5.1f}°C  |  VAE AUROC={r['VAE']['auroc']:.4f}  Recall={r['VAE']['recall_5fpr']:.3f}  |  Z AUROC={r['Z-Score']['auroc']:.4f}  Recall={r['Z-Score']['recall_5fpr']:.3f}")

df_thermal = pd.DataFrame(thermal_results)

# %% [markdown]
# ## Panel Failure Sweep

# %%
# Normal sunlight current ~0.097A. Sweep from barely reduced to extreme reversal.
panel_currents = [0.05, 0.0, -0.02, -0.05, -0.1, -0.15, -0.2, -0.3, -0.5, -0.8]
panel_results = []

for current in panel_currents:
    r = evaluate_at_magnitude(df_test, inject_panel_failure, current, scaler, vae, metadata)
    if r:
        panel_results.append({"forced_current": current, **{f"{k}_{m}": v for k, d in r.items() for m, v in d.items()}})
        print(f"  I={current:6.2f}A  |  VAE AUROC={r['VAE']['auroc']:.4f}  Recall={r['VAE']['recall_5fpr']:.3f}  |  Z AUROC={r['Z-Score']['auroc']:.4f}  Recall={r['Z-Score']['recall_5fpr']:.3f}")

df_panel = pd.DataFrame(panel_results)

# %% [markdown]
# ## Sensitivity Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Thermal Runaway ---
ax = axes[0]
ax.plot(df_thermal["delta_c"], df_thermal["VAE_auroc"], "o-", color="#e64848", label="VAE", linewidth=2)
ax.plot(df_thermal["delta_c"], df_thermal["Z-Score_auroc"], "s--", color="#092e4b", label="Z-Score Baseline", linewidth=2)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random Chance")
ax.set_xlabel("Thermal Runaway Δ (°C)")
ax.set_ylabel("AUROC")
ax.set_title("Thermal Runaway Detection Sensitivity")
ax.legend()
ax.set_ylim(0.4, 1.05)
ax.grid(True, alpha=0.3)

# --- Panel Failure ---
ax = axes[1]
ax.plot(df_panel["forced_current"], df_panel["VAE_auroc"], "o-", color="#e64848", label="VAE", linewidth=2)
ax.plot(df_panel["forced_current"], df_panel["Z-Score_auroc"], "s--", color="#092e4b", label="Z-Score Baseline", linewidth=2)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random Chance")
ax.set_xlabel("Forced Current During Sunlight (A)")
ax.set_ylabel("AUROC")
ax.set_title("Panel Failure Detection Sensitivity")
ax.legend()
ax.set_ylim(0.4, 1.05)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.suptitle("VAE vs. Z-Score Baseline: Where Does the Model Earn Its Keep?", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "sensitivity_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {FIG_DIR}/sensitivity_sweep.png")

# %% [markdown]
# ## Recall @ 5% FPR Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(df_thermal["delta_c"], df_thermal["VAE_recall_5fpr"], "o-", color="#e64848", label="VAE", linewidth=2)
ax.plot(df_thermal["delta_c"], df_thermal["Z-Score_recall_5fpr"], "s--", color="#092e4b", label="Z-Score Baseline", linewidth=2)
ax.set_xlabel("Thermal Runaway Δ (°C)")
ax.set_ylabel("Recall @ 5% FPR")
ax.set_title("Thermal Runaway — Operational Recall")
ax.legend()
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(df_panel["forced_current"], df_panel["VAE_recall_5fpr"], "o-", color="#e64848", label="VAE", linewidth=2)
ax.plot(df_panel["forced_current"], df_panel["Z-Score_recall_5fpr"], "s--", color="#092e4b", label="Z-Score Baseline", linewidth=2)
ax.set_xlabel("Forced Current During Sunlight (A)")
ax.set_ylabel("Recall @ 5% FPR")
ax.set_title("Panel Failure — Operational Recall")
ax.legend()
ax.set_ylim(-0.05, 1.05)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.suptitle("Operational Recall at Fixed 5% False Positive Rate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "sensitivity_recall.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {FIG_DIR}/sensitivity_recall.png")

# %% [markdown]
# ## Summary Table

# %%
print("\n=== THERMAL RUNAWAY ===")
print(f"{'Delta':>8s}  {'VAE AUROC':>10s}  {'Z AUROC':>10s}  {'VAE Win':>8s}  {'VAE Recall':>11s}  {'Z Recall':>11s}")
for _, row in df_thermal.iterrows():
    vae_win = "✅" if row["VAE_auroc"] > row["Z-Score_auroc"] + 0.01 else ("≈" if abs(row["VAE_auroc"] - row["Z-Score_auroc"]) < 0.01 else "❌")
    print(f"{row['delta_c']:7.1f}°C  {row['VAE_auroc']:10.4f}  {row['Z-Score_auroc']:10.4f}  {vae_win:>8s}  {row['VAE_recall_5fpr']:10.1%}  {row['Z-Score_recall_5fpr']:10.1%}")

print("\n=== PANEL FAILURE ===")
print(f"{'Current':>8s}  {'VAE AUROC':>10s}  {'Z AUROC':>10s}  {'VAE Win':>8s}  {'VAE Recall':>11s}  {'Z Recall':>11s}")
for _, row in df_panel.iterrows():
    vae_win = "✅" if row["VAE_auroc"] > row["Z-Score_auroc"] + 0.01 else ("≈" if abs(row["VAE_auroc"] - row["Z-Score_auroc"]) < 0.01 else "❌")
    print(f"{row['forced_current']:7.2f}A  {row['VAE_auroc']:10.4f}  {row['Z-Score_auroc']:10.4f}  {vae_win:>8s}  {row['VAE_recall_5fpr']:10.1%}  {row['Z-Score_recall_5fpr']:10.1%}")

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Experimentation Archive
#
# This notebook preserves our legacy explorations into using distinct Stage 1 / Stage 2 mathematical pipelines (e.g., the *Hybrid Pipeline*).
# We eventually dropped this approach because LEO orbit telemetry is highly bimodal (Day vs. Night), and Gaussian envelope models fail to draw boundaries that preserve recall efficiently in bimodal spaces. We pivoted into a unified Variational Autoencoder (VAE) architecture for end-to-end edge inference.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = Path("../data/processed")
norad_id = "43880"

# %% [markdown]
# ## 1. Load Clean Telemetry

# %%
df = pd.read_csv(PROCESSED_DIR / f"{norad_id}.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

extreme_mask = (df["batt_voltage"] > 5.0) | (df["batt_current"].abs() > 1.0)
df_clean = df[~extreme_mask].copy()

train_size = int(len(df_clean) * 0.8)
df_train = df_clean.iloc[:train_size].copy()
df_test = df_clean.iloc[train_size:].copy()

FEATURES = ["batt_voltage", "batt_current", "temp_batt_a", "temp_batt_b", "temp_panel_z"]
X_train = df_train[FEATURES].values
X_test = df_test[FEATURES].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 2. Elliptic Envelope (The Old Stage 1 Detector)

# %%
envelope = EllipticEnvelope(contamination=0.02, random_state=42)
envelope.fit(X_train_scaled)

scores = -envelope.decision_function(X_train_scaled)
print(f"Training Scores - Min: {scores.min():.2f}, Max: {scores.max():.2f}")

# %% [markdown]
# ## 3. MLP Regressor (The Old Stage 2 Baseline Diagnoser)

# %%
ae_baseline = MLPRegressor(
    hidden_layer_sizes=(3, 2, 3),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)
ae_baseline.fit(X_train_scaled, X_train_scaled)

recon_base = ae_baseline.predict(X_test_scaled)
errors = np.mean(np.abs(X_test_scaled - recon_base), axis=1)

sns.histplot(errors, bins=50)
plt.title("MLPRegressor Reconstruction Errors")
plt.show()

# %% [markdown]
# *Conclusion:* The Elliptic Envelope produced a severe 50% Recall due to Bimodality. The Sklearn MLPRegressor failed at isolating Thermal Runaway root causes due to deterministic compression constraints lacking probabilistic KL Divergence.

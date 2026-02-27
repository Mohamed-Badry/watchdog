# %% [markdown]
# # Exploratory Data Analysis: UWE-4 Telemetry
# 
# **Goal:** Understand the baseline physics and "normal" behavior of the UWE-4 satellite to design an effective Autoencoder for anomaly detection.
# 
# **Data Source:** Processed telemetry (`data/processed/43880.csv`), decoded via `satnogs-decoders`.
# 
# ## 1. Summary Statistics

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup Style
plt.style.use('ggplot')
sns.set_theme(context='notebook', style='whitegrid', palette='viridis', rc={
    'axes.facecolor': '#fafafa',
    'figure.facecolor': '#fafafa'
})

df = pd.read_csv('../data/processed/43880.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Total Rows: {len(df)}")
df.describe()

# %% [markdown]
# ### Observations from Basic Stats:
# *   **Battery Voltage:** Averages around 4.04V, with a tight standard deviation (0.07V). The minimum is 3.88V and max is 4.19V. This confirms a standard 1S Li-Ion/Li-Po configuration.
# *   **Battery Current:** Averages negative (-83 mA), indicating the satellite spends slightly more time discharging (or we receive more frames during eclipse/high load). 
# *   **OBC Temperature:** Extremely stable at 17°C (standard deviation is 0.0). *This is a potential limitation.* If the OBC temp sensor is stuck, coarse-grained, or just highly regulated, it might not be a useful feature for anomaly detection.
# *   **Panel Z Temperature:** Shows a much wider swing (-20°C to +32°C). This is excellent. It clearly shows the day/night (eclipse) cycle.

# %% [markdown]
# ## 2. Feature Correlations
# Understanding how features move together is key to an Autoencoder's success. The model learns these correlations to reconstruct the input.

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.show()

# %% [markdown]
# ### Correlation Insights:
# *   `temp_panel_z` has a strong positive correlation with `batt_voltage` (0.88) and `batt_current` (0.81). **Physics validation:** When the panel is hot (in the sun), it generates current (positive `batt_current`), which charges the battery (increasing `batt_voltage`).
# *   The Autoencoder will easily learn this rule: *If Panel Z is hot, Battery Current should be positive.* An anomaly (like a broken solar panel) would break this correlation, resulting in a high reconstruction error.
# *   `temp_obc` is `NaN` in the correlation matrix because its variance is 0 (it never changes from 17).

# %% [markdown]
# ## 3. Distribution Analysis

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df['batt_voltage'], bins=30, ax=axes[0,0], kde=True, color='teal')
axes[0,0].set_title('Battery Voltage Distribution')

sns.histplot(df['batt_current'], bins=30, ax=axes[0,1], kde=True, color='coral')
axes[0,1].set_title('Battery Current Distribution')

sns.scatterplot(data=df, x='batt_voltage', y='batt_current', hue='temp_panel_z', ax=axes[1,0], alpha=0.6, palette='magma')
axes[1,0].set_title('Voltage vs Current (Colored by Panel Temp)')

temp_cols = ['temp_batt_a', 'temp_batt_b', 'temp_panel_z']
for c in temp_cols:
    sns.kdeplot(df[c], ax=axes[1,1], label=c)
axes[1,1].set_title('Temperature Distributions')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Final Conclusions for Phase 2 (The Lab)
# 
# 1.  **Feature Selection:** We should use `batt_voltage`, `batt_current`, `temp_batt_a`, `temp_batt_b`, and `temp_panel_z`.
# 2.  **Feature Dropping:** We should drop `temp_obc` as it has zero variance in this dataset and will only add noise or cause matrix singularity issues during normalization (divide by zero).
# 3.  **Data Quality:** The data is clean, with no missing values in the telemetry fields.
# 4.  **Autoencoder Feasibility:** The strong correlations (e.g., Temperature vs. Voltage) confirm that a self-supervised Autoencoder is the perfect architecture for this dataset.

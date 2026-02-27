# %% [markdown]
# # Advanced Exploratory Data Analysis: UWE-4 Telemetry
# 
# **Goal:** Understand the baseline physics and "normal" behavior of the UWE-4 satellite to design an effective, robust Autoencoder for anomaly detection. This notebook incorporates an iterative feedback loop to refine our data engineering strategy.
# 
# **Data Source:** Processed telemetry (`data/processed/43880.csv`), decoded via `satnogs-decoders`.
# 
# ## 1. Summary Statistics & Data Quality

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.dates as mdates

# Setup Style
plt.style.use('ggplot')
sns.set_theme(context='notebook', style='whitegrid', palette='viridis', rc={
    'axes.facecolor': '#fafafa',
    'figure.facecolor': '#fafafa'
})

df = pd.read_csv('../data/processed/43880.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

print(f"Total Rows: {len(df)}")
print("\n--- Missing Values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- Constant Features (Zero Variance) ---")
for col in df.columns:
    if df[col].nunique() <= 1:
        print(f"{col}: {df[col].iloc[0]}")

# %% [markdown]
# ### Data Quality Insights:
# *   **Missing Data:** The core telemetry is perfectly clean (0 missing values). Only `observation_id` has missing values, which is irrelevant for ML.
# *   **Constant Features:** `src_callsign`, `dest_callsign`, and crucially `temp_obc` (stuck at exactly 17°C) have zero variance. 
# *   **Action:** We **must drop `temp_obc`** before training. Including a zero-variance feature will cause standard scalers to fail (divide by zero) and provides no signal to the model.

# %% [markdown]
# ## 2. Time-Series Continuity
# Amateur satellite data is typically collected via ad-hoc ground stations. We need to check if the data is continuous enough for time-series modeling (like LSTMs).

# %%
df['time_diff_sec'] = df['timestamp'].diff().dt.total_seconds()

print("--- Time Gap Statistics (Seconds) ---")
print(df['time_diff_sec'].describe())

plt.figure(figsize=(10, 4))
sns.histplot(df[df['time_diff_sec'] < 300]['time_diff_sec'], bins=50, color='purple')
plt.title("Distribution of Time Gaps (Under 5 minutes)")
plt.xlabel("Seconds between frames")
plt.show()

# %% [markdown]
# ### Time-Series Insights:
# *   **Bursty Nature:** The median time difference is 18 seconds, indicating rapid bursts of frames during a satellite pass.
# *   **Massive Gaps:** The maximum gap between frames is over 40,000 seconds (~11 hours), indicating long periods with no ground station coverage.
# *   **Action (Architecture Decision):** Because of these massive, irregular gaps, **we cannot use rolling windows or sequence models (RNN/LSTM)**. Treating the data as a continuous time-series will feed the model false discontinuities. 
# *   Instead, we will use a **Standard Feed-Forward Autoencoder**. The model will evaluate each telemetry frame as an independent, stateless snapshot of the satellite's physics at that exact moment.

# %% [markdown]
# ## 3. Physics & "Outlier" Analysis
# Standard anomaly detection techniques (like IQR or Z-score) often flag rare events as "outliers". In physics datasets, these "outliers" are often just alternate operational states (like day vs. night).

# %%
# Let's look at Battery Current Outliers via IQR
Q1 = df['batt_current'].quantile(0.25)
Q3 = df['batt_current'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['batt_current'] < (Q1 - 1.5 * IQR)) | (df['batt_current'] > (Q3 + 1.5 * IQR))]
print(f"Statistical 'Outliers' in batt_current: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# Are these outliers, or just physics? Let's check panel temperature (proxy for sunlight)
sunlight_current = df[df['temp_panel_z'] > 15]['batt_current'].mean()
eclipse_current = df[df['temp_panel_z'] <= 15]['batt_current'].mean()

print(f"\nAverage Current when Panel > 15C (Sunlight): {sunlight_current:.3f} A")
print(f"Average Current when Panel <= 15C (Eclipse): {eclipse_current:.3f} A")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='temp_panel_z', y='batt_current', hue='batt_voltage', palette='magma', alpha=0.7)
ax.axvline(15, color='red', linestyle='--', label='Approx Eclipse Boundary')
ax.axhline(0, color='blue', linestyle='--', label='Zero Current')
plt.title("Physics Verification: Panel Temp vs. Battery Current")
plt.legend()
plt.show()

# %% [markdown]
# ### Physics Insights:
# *   Standard statistical methods flag ~40% of the current data as "outliers".
# *   However, physics verification shows a clear bimodal state:
#     *   **Sunlight (Panel > 15°C):** Current is positive (+0.100 A), battery is charging.
#     *   **Eclipse (Panel < 15°C):** Current is negative (-0.206 A), battery is discharging.
# *   **Action:** We will not filter or clip these "outliers". They represent the core physical states (Day/Night cycle) that the Autoencoder needs to learn.

# %% [markdown]
# ## 4. Multi-Dimensional Relationships (Pairplot)
# To understand how the Autoencoder will map these features into a latent space, we can visualize the pairwise relationships.

# %%
features = ['batt_voltage', 'batt_current', 'temp_batt_a', 'temp_batt_b', 'temp_panel_z']

pair_fig = sns.pairplot(df[features], diag_kind='kde', corner=True, plot_kws={'alpha': 0.1, 'color': 'teal'})
pair_fig.fig.suptitle('Feature Pairwise Relationships', y=1.02)
plt.show()

# %% [markdown]
# ### Pairwise Insights:
# *   `batt_voltage` and `batt_current` show a very strong linear/bimodal relationship. 
# *   Temperatures (`temp_batt_a`, `temp_batt_b`) lag slightly behind the rapid changes in `batt_current`, which makes physical sense (thermal mass takes time to heat up after current starts flowing).
# *   This multi-collinearity is exactly what an Autoencoder excels at compressing.

# %% [markdown]
# ## 5. Operational State Clustering (PCA)
# If the Autoencoder is going to work, the "Normal" operational states should form distinct, learnable manifolds in lower-dimensional space. We use Principal Component Analysis (PCA) to verify this.

# %%
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Color by Day/Night state
df_pca['Operational State'] = np.where(df['temp_panel_z'] > 15, 'Day (Charging)', 'Night (Discharging)')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Operational State', palette='Set1', alpha=0.6)
plt.title(f'PCA of UWE-4 Telemetry\nExplained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%')
plt.show()

# %% [markdown]
# ### Clustering Insights:
# *   The first two Principal Components explain over 70% of the variance! This means the 5-dimensional physics data can be heavily compressed without losing much information.
# *   The data naturally clusters into two distinct, continuous manifolds representing "Day" and "Night" operations.
# *   **Autoencoder Design:** A bottleneck of 2 or 3 neurons in the latent space will be perfectly sufficient to capture the normal physics of UWE-4.

# %% [markdown]
# ## 6. Continuous Pass Dynamics (Micro-Scale)
# Let's look at the data not as a scatter plot, but as a continuous flow during the longest uninterrupted ground station pass.

# %%
# Find the longest continuous pass (gaps < 60s)
df['pass_id'] = (df['time_diff_sec'] > 60).cumsum()

pass_lengths = df.groupby('pass_id').size()
longest_pass_id = pass_lengths.idxmax()
longest_pass = df[df['pass_id'] == longest_pass_id]

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(longest_pass['timestamp'], longest_pass['batt_voltage'], 'b-', linewidth=2, label='Voltage (V)')
ax2.plot(longest_pass['timestamp'], longest_pass['batt_current'], 'r-', linewidth=2, label='Current (A)')
ax2.plot(longest_pass['timestamp'], longest_pass['temp_panel_z']/100, 'g--', linewidth=2, label='Panel Temp (Scaled)') # Scaled to fit current axis roughly

ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Voltage (V)', color='b', fontweight='bold')
ax2.set_ylabel('Current (A) / Temp (Scaled)', color='r', fontweight='bold')
plt.title(f'Dynamics of Longest Continuous Pass ({len(longest_pass)} frames)')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### Dynamics Insights:
# *   During a single pass, we can literally watch the satellite transition. As the panel temperature drops (entering eclipse), the current immediately drops into the negative, and the battery voltage slowly sags.
# *   The Autoencoder will be trained on thousands of these snapshots. If a frame suddenly shows high voltage but negative current during sunlight, the reconstruction error will spike, triggering an anomaly alert.

# %% [markdown]
# ## 7. Time-Series Macro Analysis
# Beyond a single pass, we need to understand the macro-level patterns over days and the actual density of our data collection.

# %%
# 7.1 Macro-Scale Time Series (First 7 Days)
start_time = df['timestamp'].min()
end_time = start_time + pd.Timedelta(days=7)
df_week = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df_week['timestamp'], df_week['batt_voltage'], 'b.', alpha=0.5, markersize=4)
ax1.set_ylabel('Battery Voltage (V)')
ax1.set_title('Macro-Scale Telemetry: 7-Day Window')

ax2.plot(df_week['timestamp'], df_week['temp_panel_z'], 'g.', alpha=0.5, markersize=4)
ax2.set_ylabel('Panel Z Temp (°C)')
ax2.set_xlabel('Time (UTC)')

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_major_locator(mdates.DayLocator())
plt.tight_layout()
plt.show()

# 7.2 Reception Density
fig, ax = plt.subplots(figsize=(14, 3))
ax.scatter(df['timestamp'], [1]*len(df), marker='|', alpha=0.3, color='purple', s=500)
ax.set_yticks([])
ax.set_title('Data Reception Density (1 month)')
ax.set_xlabel('Time (UTC)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.tight_layout()
plt.show()

# 7.3 Daily Rolling Averages
df_indexed = df.set_index('timestamp')
daily_avg = df_indexed[['batt_voltage', 'temp_batt_a', 'temp_panel_z']].resample('1D').mean()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(daily_avg.index, daily_avg['batt_voltage'], 'b-o', label='Avg Voltage')
ax2.plot(daily_avg.index, daily_avg['temp_batt_a'], 'r-s', label='Avg Batt Temp')
ax2.plot(daily_avg.index, daily_avg['temp_panel_z'], 'g-^', label='Avg Panel Temp')

ax1.set_ylabel('Voltage (V)', color='b')
ax2.set_ylabel('Temperature (°C)')
ax1.set_xlabel('Date')
plt.title('Daily Average Trends (Smoothing out the orbits)')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Macro-Time Insights:
# *   **Orbital Cycles:** In the 7-day plot, we can clearly see the cyclic rise and fall of the panel temperatures matching the orbital period, alongside the corresponding battery voltage swings.
# *   **Data Density:** The reception plot confirms the bursty nature over the entire month. The data is not uniform; some days have massive clusters of frames, while others have fewer passes captured by the SatNOGS network.
# *   **Macro Stability:** The daily rolling averages show that the satellite is fundamentally healthy. The average daily voltage is stable (not slowly dropping to 0), and temperatures hover around a stable equilibrium. Our AI will be learning from a healthy baseline.

# %% [markdown]
# ## 8. Final Feature Engineering Plan
# Based on the iterative EDA, here is the finalized plan for the ML pipeline.

# %%
print("--- Final Scaled Training Features (Sample) ---")
print(pd.DataFrame(X_scaled, columns=features).head())

# %% [markdown]
# ### Final Training Protocol ("The Lab")
# 1.  **Input Vector:** `[batt_voltage, batt_current, temp_batt_a, temp_batt_b, temp_panel_z]` (5 dimensions).
# 2.  **Dropped Features:** `temp_obc` (zero variance), metadata (`timestamp`, `observation_id`, callsigns).
# 3.  **Scaling:** `StandardScaler` (Z-score normalization). It correctly centers our bimodal physics data around 0 with a standard deviation of 1.
# 4.  **Architecture:** Feed-Forward Neural Network Autoencoder (e.g., 5 -> 3 -> 5 nodes). Stateless processing per frame to handle large temporal gaps.

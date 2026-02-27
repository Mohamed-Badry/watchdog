import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
df = pd.read_csv("data/processed/43880.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("--- Basic Stats ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Correlations ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].corr())

# Try to plot some interesting distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(df['batt_voltage'], bins=30, ax=axes[0,0], kde=True)
axes[0,0].set_title('Battery Voltage Distribution')

sns.histplot(df['batt_current'], bins=30, ax=axes[0,1], kde=True)
axes[0,1].set_title('Battery Current Distribution')

sns.scatterplot(data=df, x='batt_voltage', y='batt_current', ax=axes[1,0], alpha=0.5)
axes[1,0].set_title('Voltage vs Current')

# Thermal distribution
temp_cols = [c for c in df.columns if 'temp' in c]
for c in temp_cols:
    sns.kdeplot(df[c], ax=axes[1,1], label=c)
axes[1,1].set_title('Temperature Distributions')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('docs/figures/eda_uwe4.png')
print("\nSaved EDA plot to docs/figures/eda_uwe4.png")

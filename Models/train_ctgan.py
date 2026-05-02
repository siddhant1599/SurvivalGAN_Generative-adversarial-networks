"""
CTGAN Baseline Training Script
================================
Baseline comparison model for SurvivalGAN project.
Paper: Norcliffe et al. (2023) - SurvivalGAN

Dataset: MSK-IMPACT 50k
Split: 80% train / 20% test (matches preprocessing.py)
"""

import pandas as pd
from ctgan import CTGAN

# ── 0. LOAD DATA ──────────────────────────────────────────────────────────────
df_train = pd.read_csv("data_sets/survival_gan_train.csv")
df_test  = pd.read_csv("data_sets/survival_gan_test.csv")

print(f"Train shape: {df_train.shape}")
print(f"Test shape:  {df_test.shape}")
print(f"Train event rate: {df_train['status'].mean():.1%}")
print(f"\nColumns: {df_train.columns.tolist()}")

# ── 1. DEFINE DISCRETE COLUMNS ────────────────────────────────────────────────
# Categorical columns — label encoded in preprocessing.py
# Integer columns — treated as discrete by CTGAN
# status — event indicator 0 or 1

discrete_columns = [
    # Categorical (label encoded)
    "Cancer Type",
    "Genetic Ancestry",
    "Disease Status",
    "FACETS QC",
    "MSI Type",
    "Sex",
    "Whole Genome Doubling Status (FACETS)",
    # Integer valued
    "Age at Diagnosis",
    "Mutation Count",
    "Sample coverage",
    "Number of Other Cancer Types",
    # Target
    "status",
]

# Keep only columns that actually exist in the dataframe
discrete_columns = [c for c in discrete_columns if c in df_train.columns]
print(f"\nDiscrete columns ({len(discrete_columns)}): {discrete_columns}")

# Continuous columns — everything else except time
continuous_columns = [
    c for c in df_train.columns
    if c not in discrete_columns and c != "time"
]
print(f"Continuous columns ({len(continuous_columns)}): {continuous_columns}")

# ── 2. TRAIN CTGAN ────────────────────────────────────────────────────────────
print("\nTraining CTGAN...")
print("This will take 30-60 minutes on AWS...")

ctgan = CTGAN(
    epochs=500,
    batch_size=500,
    verbose=True       # prints loss every epoch so you can see progress
)

ctgan.fit(df_train, discrete_columns=discrete_columns)
print("\nCTGAN training complete!")

# ── 3. GENERATE SYNTHETIC PATIENTS ───────────────────────────────────────────
# Generate same number as test set
n_synthetic = len(df_test)
print(f"\nGenerating {n_synthetic} synthetic patients...")

synthetic_ctgan = ctgan.sample(n_synthetic)
print(f"Synthetic shape: {synthetic_ctgan.shape}")

# ── 4. QUICK SANITY CHECK ─────────────────────────────────────────────────────
print("\n── Sanity Check ──")
print(f"Real    event rate:    {df_test['status'].mean():.1%}")
print(f"Synthetic event rate:  {synthetic_ctgan['status'].mean():.1%}")

print(f"\nReal    time — min: {df_test['time'].min():.1f}  max: {df_test['time'].max():.1f}")
print(f"Synthetic time — min: {synthetic_ctgan['time'].min():.1f}  max: {synthetic_ctgan['time'].max():.1f}")

print(f"\nReal    Age mean: {df_test['Age at Diagnosis'].mean():.1f}")
print(f"Synthetic Age mean: {synthetic_ctgan['Age at Diagnosis'].mean():.1f}")

print(f"\nReal    TMB mean: {df_test['TMB Score'].mean():.2f}")
print(f"Synthetic TMB mean: {synthetic_ctgan['TMB Score'].mean():.2f}")

# ── 5. SAVE ───────────────────────────────────────────────────────────────────
output_path = "data_sets/synthetic_ctgan.csv"
synthetic_ctgan.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}")

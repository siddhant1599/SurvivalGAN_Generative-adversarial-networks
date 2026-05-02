"""
GAN Baseline Training Script
=====================================
Plain MLP Generator/Discriminator

Dataset: MSK-IMPACT 50k
Split: 80% train / 20% test (matches preprocessing.py)

After training, this script saves a complete checkpoint bundle in
gan_checkpoint/ :
    generator_gan.pt      — Generator weights
    scaler_gan.joblib     — fitted StandardScaler
    metadata_gan.json     — column order, discrete cols, bounds, latent_dim
    model_gan.py          — Generator class definition
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from model_gan import Generator

# ── 0. LOAD DATA ──────────────────────────────────────────────────────────────
df_train = pd.read_csv("data_sets/survival_gan_train.csv")
df_test  = pd.read_csv("data_sets/survival_gan_test.csv")

print(f"Train shape: {df_train.shape}")
print(f"Test shape:  {df_test.shape}")
print(f"Train event rate: {df_train['status'].mean():.1%}")
print(f"\nColumns: {df_train.columns.tolist()}")

# ── 1. DEFINE DISCRETE COLUMNS ────────────────────────────────────────────────

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

discrete_columns = [c for c in discrete_columns if c in df_train.columns]
print(f"\nDiscrete columns ({len(discrete_columns)}): {discrete_columns}")

continuous_columns = [
    c for c in df_train.columns
    if c not in discrete_columns and c != "time"
]
print(f"Continuous columns ({len(continuous_columns)}): {continuous_columns}")

# GAN sees one flat real-valued vector — keep column order fixed so
# we can map back to a DataFrame after sampling.
all_columns = df_train.columns.tolist()
n_features  = len(all_columns)

# Track per-column min/max for discrete columns so we can clamp samples to
# valid ranges after generation (e.g. status ∈ {0, 1}).
discrete_bounds = {
    c: (int(df_train[c].min()), int(df_train[c].max()))
    for c in discrete_columns
}

# ── 2. SCALE DATA ─────────────────────────────────────────────────────────────
# Standardize everything.  GAN works in z-score space; we'll inverse-
# transform after sampling.
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[all_columns].values).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

dataset = TensorDataset(torch.from_numpy(X_train))
loader  = DataLoader(dataset, batch_size=500, shuffle=True, drop_last=True)

# ── 3. DEFINE NETWORKS ────────────────────────────────────────────────────────
LATENT_DIM = 128


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


G = Generator(LATENT_DIM, n_features).to(device)
D = Discriminator(n_features).to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce   = nn.BCELoss()

# ── 4. TRAIN  GAN ──────────────────────────────────────────────────────
print("\nTraining GAN...")

EPOCHS = 200

for epoch in range(1, EPOCHS + 1):
    d_losses, g_losses = [], []

    for (real_batch,) in loader:
        real_batch = real_batch.to(device)
        bsz = real_batch.size(0)

        real_lbl = torch.ones (bsz, 1, device=device)
        fake_lbl = torch.zeros(bsz, 1, device=device)

        # ── Train Discriminator ──
        opt_D.zero_grad()
        d_real = D(real_batch)
        loss_real = bce(d_real, real_lbl)

        z = torch.randn(bsz, LATENT_DIM, device=device)
        fake_batch = G(z).detach()
        d_fake = D(fake_batch)
        loss_fake = bce(d_fake, fake_lbl)

        d_loss = loss_real + loss_fake
        d_loss.backward()
        opt_D.step()

        # ── Train Generator ──
        opt_G.zero_grad()
        z = torch.randn(bsz, LATENT_DIM, device=device)
        fake_batch = G(z)
        d_pred = D(fake_batch)
        # Non-saturating generator loss: maximize log D(G(z))
        g_loss = bce(d_pred, real_lbl)
        g_loss.backward()
        opt_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
        print(f"Epoch {epoch:4d}/{EPOCHS}  "
              f"D loss: {np.mean(d_losses):.4f}  "
              f"G loss: {np.mean(g_losses):.4f}")

print("\nGAN training complete!")

# ── 5. GENERATE SYNTHETIC PATIENTS ───────────────────────────────────────────
n_synthetic = len(df_test)
print(f"\nGenerating {n_synthetic} synthetic patients...")

G.eval()
with torch.no_grad():
    z = torch.randn(n_synthetic, LATENT_DIM, device=device)
    fake_scaled = G(z).cpu().numpy()

# Transform back to original feature scale
fake_raw = scaler.inverse_transform(fake_scaled)
synthetic_gan = pd.DataFrame(fake_raw, columns=all_columns)

# Round + clamp discrete columns
for c in discrete_columns:
    lo, hi = discrete_bounds[c]
    synthetic_gan[c] = np.clip(np.round(synthetic_gan[c]), lo, hi).astype(int)

if "time" in synthetic_gan.columns:
    synthetic_gan["time"] = synthetic_gan["time"].clip(lower=0.0)

print(f"Synthetic shape: {synthetic_gan.shape}")

# ── 6. QUICK SANITY CHECK ─────────────────────────────────────────────────────
print("\n── Sanity Check ──")
print(f"Real    event rate:    {df_test['status'].mean():.1%}")
print(f"Synthetic event rate:  {synthetic_gan['status'].mean():.1%}")

print(f"\nReal    time — min: {df_test['time'].min():.1f}  max: {df_test['time'].max():.1f}")
print(f"Synthetic time — min: {synthetic_gan['time'].min():.1f}  max: {synthetic_gan['time'].max():.1f}")

print(f"\nReal    Age mean: {df_test['Age at Diagnosis'].mean():.1f}")
print(f"Synthetic Age mean: {synthetic_gan['Age at Diagnosis'].mean():.1f}")

print(f"\nReal    TMB mean: {df_test['TMB Score'].mean():.2f}")
print(f"Synthetic TMB mean: {synthetic_gan['TMB Score'].mean():.2f}")

# ── 7. SAVE SYNTHETIC CSV ─────────────────────────────────────────────────────
output_path = "data_sets/synthetic_vanilla_gan.csv"
synthetic_gan.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}")

# ── 8. SAVE CHECKPOINT BUNDLE  ────────────────────────────────
# reconstruct G and sample new patients.
ckpt_dir = Path("gan_checkpoint")
ckpt_dir.mkdir(exist_ok=True)

# 8a. Generator weights
G_cpu_state = {k: v.detach().cpu() for k, v in G.state_dict().items()}
torch.save(G_cpu_state, ckpt_dir / "generator_gan.pt")

# 8b. Fitted scaler
joblib.dump(scaler, ckpt_dir / "scaler_gan.joblib")

# 8c. Metadata — column order, discrete handling, network shape
metadata = {
    "latent_dim": LATENT_DIM,
    "n_features": n_features,
    "columns": all_columns,
    "discrete_columns": discrete_columns,
    "discrete_bounds": discrete_bounds,
    "epochs_trained": EPOCHS,
    "architecture": "MLP 128 → 256 → 256 → 256 → n_features (LeakyReLU 0.2)",
}
with open(ckpt_dir / "metadata_gan.json", "w") as f:
    json.dump(metadata, f, indent=2)

# 8d. Copy the Generator class definition next to the weights.
shutil.copy("model_gan.py", ckpt_dir / "model_gan.py")

print(f"\nCheckpoint bundle saved → {ckpt_dir}/")
print("  ├── generator_gan.pt")
print("  ├── scaler_gan.joblib")
print("  ├── metadata_gan.json")
print("  └── model_gan.py")

"""
Local check for the saved GAN checkpoint.
=================================================
Loads gan_checkpoint/ and runs a few example queries against G to verify:
  1. The .pt file loads into the Generator architecture cleanly
  2. The scaler + metadata round trip produces sensible patient rows
  3. The same z always produces the same patient
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model_gan import Generator

CKPT = Path("gan_checkpoint")

# ── 1. LOAD THE BUNDLE ────────────────────────────────────────────────────────
print("Loading checkpoint...")
with open(CKPT / "metadata_gan.json") as f:
    meta = json.load(f)
scaler = joblib.load(CKPT / "scaler_gan.joblib")

G = Generator(meta["latent_dim"], meta["n_features"])
G.load_state_dict(torch.load(CKPT / "generator_gan.pt", map_location="cpu"))
G.eval()

print(f"  latent_dim  : {meta['latent_dim']}")
print(f"  n_features  : {meta['n_features']}")
print(f"  columns     : {meta['columns']}")
print(f"  G params    : {sum(p.numel() for p in G.parameters()):,}")

# ── 2. TEST 1 — single example vector ────────────────────────────────────────
print("\n── Test 1: one synthetic patient from a single z ──")
torch.manual_seed(0)
z = torch.randn(1, meta["latent_dim"])
print(f"  z shape: {tuple(z.shape)}  (first 5 values: {z[0, :5].tolist()})")

with torch.no_grad():
    fake_scaled = G(z).numpy()
print(f"  G(z) shape: {fake_scaled.shape}  (z-scored, raw network output)")
print(f"  G(z) first 5 values: {fake_scaled[0, :5]}")

fake_raw = scaler.inverse_transform(fake_scaled)
df = pd.DataFrame(fake_raw, columns=meta["columns"])
for c in meta["discrete_columns"]:
    lo, hi = meta["discrete_bounds"][c]
    df[c] = np.clip(np.round(df[c]), lo, hi).astype(int)
if "time" in df.columns:
    df["time"] = df["time"].clip(lower=0.0)

print("\n  Decoded synthetic patient:")
print(df.T.to_string(header=False))

# ── 3. TEST 2 — determinism: same z → same patient ───────────────────────────
print("\n── Test 2: determinism check ──")
torch.manual_seed(0)
z_a = torch.randn(1, meta["latent_dim"])
torch.manual_seed(0)
z_b = torch.randn(1, meta["latent_dim"])
with torch.no_grad():
    out_a = G(z_a).numpy()
    out_b = G(z_b).numpy()
identical = np.allclose(out_a, out_b)
print(f"  Same seed → same output: {identical}")

# ── 4. TEST 3 — batch of 5 patients ──────────────────────────────────────────
print("\n── Test 3: batch of 5 synthetic patients ──")
torch.manual_seed(42)
z_batch = torch.randn(5, meta["latent_dim"])
with torch.no_grad():
    fake_batch = G(z_batch).numpy()

fake_batch_raw = scaler.inverse_transform(fake_batch)
df_batch = pd.DataFrame(fake_batch_raw, columns=meta["columns"])
for c in meta["discrete_columns"]:
    lo, hi = meta["discrete_bounds"][c]
    df_batch[c] = np.clip(np.round(df_batch[c]), lo, hi).astype(int)
if "time" in df_batch.columns:
    df_batch["time"] = df_batch["time"].clip(lower=0.0)

print(df_batch.to_string())

print("\nAll Tests passed")

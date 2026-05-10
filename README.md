# SurvivalGAN - Generative adversarial network

Extension of **SurvivalGAN** (Norcliffe et al., 2023) on the
**MSK-IMPACT 50K** dataset, benchmarked against **CTGAN** and a
plain **GAN** baseline.

> See `Research_Paper.pdf` for the full write-up.
> Authors: Siddhant Saxena, Zixuan Zhao, Simbarashe Mpofu

The repository ships:
- A reproducible preprocessing pipeline for the raw MSK-IMPACT archive.
- Three trainable generative models for survival data (GAN, CTGAN, SurvivalGAN).
- A unified evaluation harness producing marginal, joint, and survival specific metrics.
- A Flask + Angular reference app that serves the trained GAN generator.

---

## Repository layout

```
survivalGAN/
├── Models/                       # All ML code, datasets, checkpoints, eval results
│   ├── preprocessing.py          # Raw MSK-IMPACT → train/test split + metadata JSON
│   ├── model_gan.py              # GAN Generator class (shared by train + serve)
│   ├── train_gan.py              # GAN training + checkpoint bundle writer
│   ├── train_ctgan.py            # CTGAN baseline (sdv/ctgan)
│   ├── survGAN.py                # SurvivalGAN core: WGAN-GP + DeepHit + XGB TTE
│   ├── train_survGAN_aws.py      # Driver around survGAN.SurvivalPipeline (EC2-friendly)
│   ├── test_gan.py               # Local sanity check for the saved GAN bundle
│   ├── evaluate_synthetic_data.py# Synthcity based eval (stats / detection / survival)
│   ├── datasets_raw/             # MSK-IMPACT raw export
│   ├── datasets_cleaned/         # Train/test splits + per-method synthetic CSVs
│   ├── gan_checkpoint/           # GAN deployable bundle (served by backend)
│   ├── eval_results_gan/         # GAN metrics + figures
│   ├── eval_results_ctgan/       # CTGAN metrics + figures
│   └── eval_results_survGan/     # SurvivalGAN metrics + figures
├── Backend/                      # Flask inference server for the  GAN
│   ├── app.py
│   └── requirements.txt
├── Frontend/                     # Angular 14 web UI
└── Research_Paper.pdf            # Manuscript
```

---

## Dataset: MSK-IMPACT 50K

The MSK-IMPACT 50K release records overall survival
together with demographic, histopathological, and genomic variables for 48,179
unique cancer patients across 64 cancer types.

`Models/preprocessing.py` reduces the raw 45 column / 54,331 sample export to a
modeling ready cohort:

| Stage | What happens | Result |
|---|---|---|
| Missing removal + dedup | drop rows missing OS labels; keep one (preferably *Primary*) sample per patient | 43,673 patients |
| Feature selection | drop IDs, HLA, >40% missing columns; bucket *Cancer Type* to top-15 + *Other* | 19 columns |
| Cleaning | clip *Pathologist Tumor Purity* to `[0, 100]`, median-impute continuous, `Unknown` for categorical, label-encode categoricals, **preserve integer dtypes** | clean cohort |
| Split | 80 / 20 train / test, both with the same 38.6 % event rate | 34,938 train / 8,735 test |

Two artifacts are emitted alongside the CSVs:
- `survival_gan_train.csv`, `survival_gan_test.csv`
- `survival_gan_column_metadata.json` — integer columns + per-column min/max bounds
  used at generation time to enforce physical constraints.

The 17 retained covariates split into ten continuous (e.g. *Age at Diagnosis*,
*Ploidy*, *FACETS Estimated Purity*, *TMB Score*, *MSI Score*, *Mutation Count*,
*Sample coverage*) and seven categorical (*Sex*, *Cancer Type*, *Disease Status*,
*MSI Type*, *Genetic Ancestry*, *FACETS QC*, *Whole Genome Doubling Status*),
with `time` (months) and `status` (event) as targets.

---

## Models

### 1. GAN Baseline — `train_gan.py` + `model_gan.py`

Plain MLP generator and discriminator on the standardized full feature vector
(covariates + `time` + `status`). Sanity baseline; not survival aware.

| | Spec |
|---|---|
| Generator | `Linear(latent=128, 256) → LeakyReLU → ×3 → Linear(256, n_features)` |
| Discriminator | `Linear → LeakyReLU → Dropout(0.3)` ×2 → `Linear(256, 1) → Sigmoid` |
| Loss | Binary cross-entropy (non-saturating G loss) |
| Optimizer | Adam, lr 2e-4, β = (0.5, 0.999) |
| Epochs / batch | 200 / 500 |
| Decoding | `inverse_transform` → round + clip discrete columns → `time = max(0, time)` |

Outputs a self-contained checkpoint bundle in `gan_checkpoint/`:
```
generator_gan.pt     # Generator weights
scaler_gan.joblib    # fitted StandardScaler
metadata_gan.json    # column order, discrete cols, bounds, latent_dim
model_gan.py         # Generator class definition
```

### 2. CTGAN — `train_ctgan.py`

Tabular only baseline using the `ctgan` library with mode specific normalization
for continuous columns and a conditional vector for categorical columns.
Trained for 500 epochs, batch size 500, with the same discrete column list as
above (`Cancer Type`, `Sex`, `MSI Type`, `status`, etc.). Survival naive — it
sees `time` and `status` as ordinary tabular columns.

### 3. SurvivalGAN — `survGAN.py` + `train_survGAN_aws.py`

A three-model architecture specialized for survival data:

```
            ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐
 z ~ N(0,I) │  1. Conditional    │  │  2. Survival fn    │  │  3. Time regressor  │
 C, E from  │      WGAN-GP       │→ │      DeepHit       │→ │       XGBoost       │ → (x, t, E)
 sampler    │   G(z, C) → x_e    │  │   S(t | x) at N_H  │  │  (x, S(·), E) → t   │
            └────────────────────┘  └────────────────────┘  └─────────────────────┘
```

- **Conditional WGAN-GP** — generates covariates `x` conditioned on event status
  `e` and a class label `C = f(x)` from a Bayesian Gaussian Mixture Model
  (`encoder_max_clusters = 5` in our config, vs. paper's 10, to mitigate
  spurious multi modality on the heterogeneous pan cancer cohort).
- **DeepHit** survival head — `S(t | x)` evaluated at 100 time horizons.
  Uses our local `LocalSurvivalFunctionTTE` wrapper (replaces synthcity's buggy
  `SurvivalFunctionTimeToEvent` plugin and adds log time clamping).
- **XGBoost time regressor** — predicts `log T` from `[x, S(·|x), e]`. Bumped
  to `n_estimators=500`, `max_depth=6` (vs. 200 / 5 in the paper) to scale to
  the larger MSK-IMPACT cohort.

`train_survGAN_aws.py` is a thin CLI wrapper around `survGAN.SurvivalPipeline`
that adds: file logging, GPU memory reporting, pickled checkpoint, post
generation precision matching, and physical constraint enforcement.

| Component | Parameter | Ours | Paper |
|---|---|---|---|
| DeepHit | epochs / lr / hidden / dropout | 2000 / 1e-3 / 300 / 0.02 | match ✓ |
| XGB TTE | estimators / max-depth / booster | 500 / 6 / gbtree × | 200 / 5 / Dart |
| WGAN-GP | iters / batch / G width / dropout | 3000 / 256 / 256 / 0.0 × | 1500 / 500 / 250 / 0.1 |
| WGAN-GP | gradient penalty λ / encoder clusters | 10 ✓ / 5 × | 10 / 10 |

---

## Evaluation — `evaluate_synthetic_data.py`

A single CLI runs all three tiers of metrics for any synthetic CSV against the
real train/test splits:

- **Marginal fidelity** — Jensen-Shannon distance, inverse KL.
- **Joint fidelity** — Wasserstein distance, MMD, PRDC (precision/recall/density/
  coverage), α-precision / β-recall / authenticity, detection AUC under
  XGBoost & MLP attackers.
- **Survival-specific** — Optimism, Kaplan–Meier divergence, short-sightedness,
  censoring-rate difference, time-Wasserstein on event vs. censored arms.
- **Downstream utility** — Cox PH and XGBoost-AFT C-index under TRTR / TSTR /
  TSTS, plus Spearman correlation between feature importances on real vs.
  synthetic training data.

Each method directory (`eval_results_gan/`, `eval_results_ctgan/`,
`eval_results_survGan/`) contains `comparison.csv`,
`per_method/synthetic/`, and `report/figures/`,
`report/tables/`.

### Headline results (held-out test set, `n = 8,735`)

| Metric | SurvivalGAN | CTGAN | Better |
|---|---|---|---|
| Jensen-Shannon (marg. mean) | **0.006** | 0.012 | SurvivalGAN |
| Wasserstein (joint) | **0.096** | 0.163 | SurvivalGAN |
| α-precision (OC) | **0.922** | 0.718 | SurvivalGAN |
| Authenticity (OC) | 0.513 | **0.557** | CTGAN |
| KM divergence | **0.042** | 0.091 | SurvivalGAN |
| Short-sightedness | **0.000** | 0.058 | SurvivalGAN |
| XGBoost C-index TSTR | **0.628** | 0.539 | SurvivalGAN |
| Feat-importance Spearman ρ | **0.733** | 0.027 | SurvivalGAN |

SurvivalGAN dominates every survival distribution metric and preserves the
covariate outcome dependency that downstream survival learners exploit
(ρ = 0.73 vs. 0.03 for CTGAN), confirming Norcliffe et al.'s thesis that
survival structure should be modelled **explicitly** outside the GAN.
β-recall (≈0.45) remains the open weakness — the generator covers
high density regions well but under-samples the tails.

---

## Quick start

### Reproduce the ML pipeline

Run from the `Models/` directory. Expects MSK-IMPACT raw CSV under
`Models/datasets_raw/`. A GPU is recommended for SurvivalGAN.

```bash
cd Models

# 1. Clean + split
python preprocessing.py

# 2a. Train vanilla GAN (writes gan_checkpoint/ + synthetic_vanilla_gan.csv)
python train_gan.py

# 2b. Train CTGAN baseline
python train_ctgan.py

# 2c. Train SurvivalGAN (long; tmux on EC2 recommended)
python train_survGAN_aws.py \
    --input  datasets_cleaned/survival_gan_train.csv \
    --output datasets_cleaned/synthetic_survgan.csv \
    --n-iter 3000 \
    --synthetic-count 8735 \
    --checkpoint-dir survgan_ckpt \
    --metadata-json datasets_cleaned/survival_gan_column_metadata.json

# 3. Evaluate any synthetic CSV against the real splits
python evaluate_synthetic_data.py \
    --real-train datasets_cleaned/survival_gan_train.csv \
    --real-test  datasets_cleaned/survival_gan_test.csv  \
    --synthetic  datasets_cleaned/synthetic_survgan.csv  \
    --time-column time --status-column status            \
    --outdir eval_results_survGan

# 4. Sanity-check the saved vanilla GAN bundle locally
python test_gan.py
```

### Serve the GAN

```bash
cd Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py     # http://127.0.0.1:5000
```

Endpoints:
- `GET  /api/health`   → `{ok, n_features}`
- `POST /api/generate` → samples `z ~ N(0, I)`, decodes to a single synthetic
  patient row with the full column dictionary.

### Run the Angular frontend

```bash
cd Frontend
npm install
ng serve         # http://localhost:4200
```
`proxy.conf.json` forwards `/api/*` to the Flask server, so no CORS setup is
needed.

---

## Dependencies

ML training & eval (install into a single Python ≥ 3.10 venv):

```
torch>=2.0
numpy pandas scikit-learn joblib tqdm
xgboost lifelines
ctgan                       # CTGAN baseline
synthcity                   # DeepHit + metrics
matplotlib scipy
```

Backend additionally needs `flask>=2.3` and `flask-cors>=4.0`
(see `Backend/requirements.txt`). Frontend pins are in
`Frontend/package.json` (Angular 14).

---

## Citation

If you use this code or the trained checkpoints, please cite the accompanying
manuscript (`Research_Paper.pdf`):

> Norcliffe, A. *et al.* **SurvivalGAN: Generating Time-to-Event Data for
> Survival Analysis.** AISTATS, 2023.

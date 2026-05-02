
"""
MSK-IMPACT 50k → Survival GAN Cleaning Pipeline

CHANGES vs previous version (2026-04):
  1. PRESERVE INTEGER DTYPE. After SimpleImputer's median imputation, columns
     that were integer-valued in the raw data (Mutation Count, Sample coverage,
     Number of Other Cancer Types, Age at Diagnosis) are rounded and cast back
     to int64. Previously they silently became float64, which caused the
     downstream precision-match in train_survGAN_aws.py to fail (its integer
     restoration checks `np.issubdtype(real.dtype, np.integer)`, which is
     False for float64). Result in the old pipeline: synthetic Mutation Count
     values like `2.4631227` instead of `2`, and *all* 34938 synthetic rows
     failed the integer check.

  2. CLIP PATHOLOGIST PURITY TO [0, 100] BEFORE IMPUTATION. The raw column
     has a data-entry artefact (max value 9000, clearly meant to be 90).
     Without clipping, this single outlier inflated the median imputation and
     the BayesianGMM fit, causing the synthetic column to run up to 9997.
     We now clip to its semantic range [0, 100] percent.

  3. WRITE metadata JSON alongside the CSVs. It records:
       - which columns are integer-valued
       - per-column min/max bounds for clipping at generation time
     `train_survGAN_aws.py` reads this file to enforce physical constraints
     on the synthetic output.

  4. Previously-existing changes retained:
     - 80/20 train/test split (no val — GANs have no val-loss early stopping).
     - Drop high-cardinality HLA + Primary Site columns before encoding.
     - Bucket Cancer Type to top-15 + "Other".
     - Patient-level de-duplication; leakage-safe imputation fit on train only.
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# The random seed
random_seed = 42

# Set random seed in numpy
np.random.seed(random_seed)

# ── 0. LOADING DATA ───────────────────────────────────────────────────────────
RAW_PATH = "data sets/msk_impact 50K.csv"

df = pd.read_csv(RAW_PATH, low_memory=False)
df.columns = df.columns.str.strip()
print(f"Raw shape: {df.shape}")


# ── 1. REQUIRE SURVIVAL LABELS ────────────────────────────────────────────────
# Both OS fields must be present — rows without them are unusable for survival.
df = df.dropna(subset=["Overall Survival (Months)", "Overall Survival Status"])

# Parsing event indicator: "1:DECEASED" → 1 | "0:LIVING" → 0
df["status"] = df["Overall Survival Status"].str.startswith("1").astype(int)
df["time"]   = df["Overall Survival (Months)"].astype(float)

# Drop raw OS columns — replaced by time / status
df = df.drop(columns=["Overall Survival (Months)", "Overall Survival Status"])

# Dropping biologically impossible values (must have time > 0)
df = df[(df["time"] > 0) & (df["time"] < 600)].reset_index(drop=True)
print(f"After survival filter: {df.shape}")


# ── 2. DE-DUPLICATE TO PATIENT LEVEL ─────────────────────────────────────────
# Some patients have multiple samples. Keep the PRIMARY sample when available,
# otherwise the first sample encountered.
def pick_sample(group):
    primary = group[group["Sample Type"] == "Primary"]
    return primary.iloc[0] if not primary.empty else group.iloc[0]

df = df.groupby("Patient ID", group_keys=False).apply(pick_sample)
df = df.reset_index(drop=True)
print(f"After patient-level de-duplication: {df.shape}")


# ── 3. SPLITTING THE DATA ─────────────────────────────────────────────────────
df_train, df_test = train_test_split(
    df, train_size=0.8, random_state=random_seed, stratify=df["status"]
)

df_train = df_train.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

print(pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=["# rows", "# columns"]))
print(pd.DataFrame([[df_test.shape[0],  df_test.shape[1]]],  columns=["# rows", "# columns"]))

for name, split in [("train", df_train), ("test", df_test)]:
    print(f"  {name} event rate: {split['status'].mean():.1%}")


# ── 4. HANDLING IDENTIFIERS + PROBLEMATIC HIGH-CARDINALITY COLS ──────────────
drop_id_cols = [
    "Patient ID", "Sample ID", "Study ID",
    "Sample Type",                          # used for dedup, not a feature
    "Gene Panel",                           # assay label, not a patient feature
    "FACETS Suite Version",                 # version string
    "Somatic Status",                       # near-constant (~100% Matched)
    "Purity Estimate from Mutations",       # binary flag, low info
    "HLA Genotype Available",               # derived flag
    "Number of Samples Per Patient",        # dedup artifact
    "Oncotree Code",                        # redundant with Cancer Type
    "Cancer Type Detailed",                 # too high cardinality
    "HLA-A Allele 1 Genotype",
    "HLA-A Allele 2 Genotype",
    "HLA-B Allele 1 Genotype",
    "HLA-B Allele 2 Genotype",
    "HLA-C Allele 1 Genotype",
    "HLA-C Allele 2 Genotype",
    "Primary Site",
]

df_train = df_train.drop(columns=drop_id_cols, errors="ignore")
df_test  = df_test.drop(columns=drop_id_cols,  errors="ignore")


# ── 4b. BUCKET RARE CANCER TYPES — TOP-15 FIT ON TRAIN ───────────────────────
TOP_N_CANCER = 15
if "Cancer Type" in df_train.columns:
    top_types = df_train["Cancer Type"].value_counts().head(TOP_N_CANCER).index.tolist()

    def bucket_cancer(s):
        return s.where(s.isin(top_types), "Other")

    df_train["Cancer Type"] = bucket_cancer(df_train["Cancer Type"])
    df_test["Cancer Type"]  = bucket_cancer(df_test["Cancer Type"])
    n_other_train = (df_train["Cancer Type"] == "Other").sum()
    n_other_test  = (df_test["Cancer Type"]  == "Other").sum()
    print(f"\nCancer Type: kept top-{TOP_N_CANCER} types, bucketed "
          f"{n_other_train} train / {n_other_test} test rows as 'Other'")


# ── 4c. CLIP PATHOLOGIST PURITY TO ITS SEMANTIC RANGE [0, 100] ──────────────  (NEW)
# Pathologist Estimated Tumor Purity is a percentage. The raw column contains
# at least one value of 9000 (almost certainly a data-entry error for 90).
# Without this clip, the outlier inflates the median imputation and the
# BayesianGMM fit downstream, producing synthetic values up to 9997.
#
# Done BEFORE imputation so that the outlier doesn't poison the median either.
# Applied to both splits independently — this is a semantic bound, not a
# learned parameter, so there's no leakage.
if "Pathologist Estimated Tumor Purity" in df_train.columns:
    for split_name, split in [("train", df_train), ("test", df_test)]:
        col = "Pathologist Estimated Tumor Purity"
        # convert first so clip works on numeric (string column in raw)
        split[col] = pd.to_numeric(split[col], errors="coerce")
        n_before_clip = split[col].notna().sum()
        n_oob = ((split[col] < 0) | (split[col] > 100)).sum()
        if n_oob > 0:
            print(f"  Pathologist Purity ({split_name}): clipping {n_oob} "
                  f"out-of-range values into [0, 100]")
        split[col] = split[col].clip(lower=0.0, upper=100.0)


# ── 5. HANDLING MISSING VALUES ────────────────────────────────────────────────

def nan_checker(df):
    df_nan = pd.DataFrame(
        [[var, df[var].isna().sum() / df.shape[0], str(df[var].dtype)]
         for var in df.columns if df[var].isna().sum() > 0],
        columns=["var", "proportion", "dtype"]
    )
    df_nan = df_nan.sort_values(by="proportion", ascending=False).reset_index(drop=True)
    return df_nan

df_nan = nan_checker(df_train)
print("\nMissing values in train:")
print(df_nan)
print(pd.DataFrame(df_nan["dtype"].unique(), columns=["dtype"]))


# ── 5a. DROP COLUMNS WITH > 40% MISSING (based on train) ─────────────────────
threshold = 0.40
cols_to_drop = df_nan[df_nan["proportion"] > threshold]["var"].tolist()
print(f"\nColumns dropped (>40% missing): {cols_to_drop}")

df_train = df_train.drop(columns=cols_to_drop, errors="ignore")
df_test  = df_test.drop(columns=cols_to_drop,  errors="ignore")


# ── 5b. TYPE COERCIONS ────────────────────────────────────────────────────────
# Columns stored as String in the raw MSK CSV that are semantically numeric.
# Both of these need `pd.to_numeric` coercion before any downstream numeric
# operation (imputation, clipping, integer casting). Without this, any raw
# non-numeric sentinel (e.g. "NA", "Unknown", ">89" in age de-identification)
# keeps the column at dtype=object and breaks everything that follows.
NUMERIC_STRING_COLS = [
    "Pathologist Estimated Tumor Purity",
    "Age at Diagnosis",
]

def coerce_types(split):
    split = split.copy()
    for col in NUMERIC_STRING_COLS:
        if col in split.columns:
            split[col] = pd.to_numeric(split[col], errors="coerce")
    if "Whole Genome Doubling Status (FACETS)" in split.columns:
        split["Whole Genome Doubling Status (FACETS)"] = (
            split["Whole Genome Doubling Status (FACETS)"]
            .astype(str)
            .replace("nan", np.nan)
        )
    return split

df_train = coerce_types(df_train)
df_test  = coerce_types(df_test)


# ── 5b-ii. RECORD INTEGER-VALUED COLUMNS (BEFORE IMPUTATION) ────────────  (NEW)
# We need to remember which columns are naturally integer-valued, so that
# after SimpleImputer (which casts to float64) we can round + cast back.
# A column is considered integer-valued if every non-null entry equals its
# rounded value. Status/time are excluded — they have explicit handling.
INT_CANDIDATES = {
    "Age at Diagnosis",
    "Mutation Count",
    "Number of Other Cancer Types",
    "Sample coverage",
}
integer_columns = []
for col in df_train.columns:
    if col in ("time", "status"):
        continue
    if col not in INT_CANDIDATES:
        continue
    s = pd.to_numeric(df_train[col], errors="coerce").dropna()
    if len(s) > 0 and (s == s.round()).all():
        integer_columns.append(col)

print(f"\nInteger-valued columns to preserve: {integer_columns}")


# ── 5c. RE-CHECKING MISSING AFTER DROPPING HIGH-MISSINGNESS COLS ────────────
df_nan = nan_checker(df_train)
print("\nMissing values after dropping >40% columns:")
print(df_nan)


# ── 5d. IMPUTE NUMERIC COLUMNS — FIT ON TRAIN ONLY ───────────────────────────
df_miss_num = df_nan[df_nan["dtype"].isin(["float64", "int64", "Float64", "Int64", "float32"])].reset_index(drop=True)
print("\nNumeric columns to impute:")
print(df_miss_num)

if len(df_miss_num) > 0:
    num_vars = df_miss_num["var"].tolist()
    si_num = SimpleImputer(missing_values=np.nan, strategy="median")
    df_train[num_vars] = si_num.fit_transform(df_train[num_vars])
    df_test[num_vars]  = si_num.transform(df_test[num_vars])


# ── 5e. IMPUTE CATEGORICAL COLUMNS — FIT ON TRAIN ONLY ───────────────────────
df_miss_cat = df_nan[df_nan["dtype"].isin(["object", "str", "string"])].reset_index(drop=True)
print("\nCategorical columns to impute:")
print(df_miss_cat)

if len(df_miss_cat) > 0:
    cat_vars = df_miss_cat["var"].tolist()
    si_cat = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Unknown")
    df_train[cat_vars] = si_cat.fit_transform(df_train[cat_vars])
    df_test[cat_vars]  = si_cat.transform(df_test[cat_vars])


# ── 5f. CLIP OUTLIERS — USING TRAIN QUANTILES ONLY ───────────────────────────
# TMB Score, Mutation Count, MSI Score have extreme right tails and (in the
# raw data) negative sentinels (-1) for missing. Fit bounds on train only.
# Pathologist Purity already handled in §4c with a semantic bound.
clip_cols = ["TMB Score", "Mutation Count", "MSI Score"]
p99_bounds = {}   # remembered for metadata JSON
for col in clip_cols:
    if col not in df_train.columns:
        continue
    lo  = 0.0
    hi  = float(df_train[col].quantile(0.99))
    for split in [df_train, df_test]:
        split[col] = split[col].clip(lower=lo, upper=hi)
    p99_bounds[col] = (lo, hi)
    print(f"  Clipped '{col}' to [0, {hi:.2f}] using train p99")


# ── 5g. RESTORE INTEGER DTYPE ─────────────────────────────────────────────  (NEW)
# After imputation + clipping, integer columns (e.g., Mutation Count) are
# float64. Round to nearest integer and cast back to int64.

for col in integer_columns:
    if col not in df_train.columns:
        continue
    # Compute a safe fallback value from train only (leakage-safe).
    s_train_numeric = pd.to_numeric(df_train[col], errors="coerce")
    fallback = float(s_train_numeric.median()) if s_train_numeric.notna().any() else 0.0
    for split_name, split in [("train", df_train), ("test", df_test)]:
        s = pd.to_numeric(split[col], errors="coerce")
        n_nan = int(s.isna().sum())
        if n_nan > 0:
            print(f"  WARN: '{col}' has {n_nan} NaN/non-numeric values in "
                  f"{split_name} before int cast — filling with train median "
                  f"{fallback}")
            s = s.fillna(fallback)
        split[col] = s.round().astype(np.int64)
    print(f"  Restored '{col}' to int64")


# ── 6. CATEGORICAL VARIABLE CHECKER ──────────────────────────────────────────

def cat_var_checker(df, target_cols):
    df_cat = pd.DataFrame(
        [[var, df[var].nunique(dropna=False)]
         for var in df.columns
         if var not in target_cols and df[var].dtype in ["object", "str", "string"]],
        columns=["var", "nunique"]
    )
    df_cat = df_cat.sort_values(by="nunique", ascending=False).reset_index(drop=True)
    return df_cat

target = ["time", "status"]
df_cat = cat_var_checker(df_train, target_cols=target)
print("\nCategorical variables:")
print(df_cat)


# ── 7. ENCODING ───────────────────────────────────────────────────────────────
cat_cols = df_cat["var"].tolist()

encoders = {}
for col in cat_cols:
    for split in [df_train, df_test]:
        split[col] = split[col].astype(str)

    le = LabelEncoder()
    le.fit(df_train[col])
    encoders[col] = le

    train_classes = set(le.classes_)
    fallback = le.classes_[0]

    for split in [df_train, df_test]:
        split[col] = split[col].apply(
            lambda x: x if x in train_classes else fallback
        )
        split[col] = le.transform(split[col])

    print(f"  Label encoded '{col}' → {len(le.classes_)} classes")


# ── 8. FINAL QC ───────────────────────────────────────────────────────────────
print("\n── Final QC ──")
for name, split in [("train", df_train), ("test", df_test)]:
    assert split.isnull().sum().sum() == 0,       f"{name}: NaNs remaining!"
    assert (split["time"] > 0).all(),              f"{name}: non-positive times!"
    assert split["status"].isin([0, 1]).all(),     f"{name}: bad status codes!"
    print(f"  {name}: {split.shape} | event rate: {split['status'].mean():.1%} ✓")


# ── 9. SAVE CSVs ──────────────────────────────────────────────────────────────
df_train.to_csv("data sets/survival_gan_train.csv", index=False)
df_test.to_csv("data sets/survival_gan_test.csv",   index=False)

print("\nSaved:")
print("  data sets/survival_gan_train.csv  (for GAN training)")
print("  data sets/survival_gan_test.csv   (held out for downstream eval)")
print(f"\nFinal columns ({len(df_train.columns)}): {df_train.columns.tolist()}")


# ── 10. WRITE COLUMN METADATA JSON
# ── 10. WRITE COLUMN METADATA JSON ───────────────────────────────────────  (NEW)
# This file is consumed by train_survGAN_aws.py to enforce physical
# constraints on synthetic output (clip to bounds, cast integers).
#
# Per column:
#   - "dtype":    "int"  or  "float"
#   - "bounds":   [lo, hi]  (either may be null if unbounded on that side)
#
# Integer columns: round + cast synthetic values to int, then clip.
# Bounded columns: clip synthetic values to [lo, hi].
#
# The bounds are a combination of:
#   - Training-data [min, max] (always recorded)
#   - Semantic bounds for fields with known physical meaning (override)
# Semantic bounds are stricter where applicable and prevent the GAN from
# generating biologically impossible values like negative mutation counts
# or genome fractions > 1.

def _col_bounds(df_train, col, semantic=None):
    """Return the tighter of (observed min, observed max) and semantic bounds."""
    obs_lo = float(df_train[col].min())
    obs_hi = float(df_train[col].max())
    if semantic is None:
        return [obs_lo, obs_hi]
    sem_lo, sem_hi = semantic
    lo = obs_lo if sem_lo is None else max(obs_lo, sem_lo)
    hi = obs_hi if sem_hi is None else min(obs_hi, sem_hi)
    return [lo, hi]


# Known semantic bounds (what the column *means* in the real world).
# These override observed bounds where they are stricter.
SEMANTIC_BOUNDS = {
    "Fraction Genome Altered":             (0.0, 1.0),    # literally a fraction
    "FACETS Estimated Purity":             (0.0, 1.0),    # proportion
    "Pathologist Estimated Tumor Purity":  (0.0, 100.0),  # percentage
    "Ploidy (FACETS)":                     (0.0, None),   # non-negative
    "MSI Score":                           (0.0, None),   # already clipped
    "TMB Score":                           (0.0, None),   # already clipped
    "Mutation Count":                      (0, None),     # already clipped
    "time":                                (1e-3, None),  # > 0
    "Age at Diagnosis":                    (0, 120),      # human age range
}

col_metadata = {}
for col in df_train.columns:
    dtype_kind = str(df_train[col].dtype)
    is_int = col in integer_columns or pd.api.types.is_integer_dtype(df_train[col])
    bounds = _col_bounds(df_train, col, semantic=SEMANTIC_BOUNDS.get(col))
    col_metadata[col] = {
        "dtype": "int" if is_int else "float",
        "bounds": bounds,
    }


meta = {
    "source_train_csv": "data sets/survival_gan_train.csv",
    "source_test_csv":  "data sets/survival_gan_test.csv",
    "time_column":      "time",
    "status_column":    "status",
    "integer_columns":  integer_columns + [
        c for c in df_train.columns
        if c not in integer_columns
        and pd.api.types.is_integer_dtype(df_train[c])
        and c != "status"   # status is already integer by construction
    ] + ["status"],
    "columns":          col_metadata,
}

meta_path = Path("data sets/survival_gan_column_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nWrote column metadata → {meta_path}")
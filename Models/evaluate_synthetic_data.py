"""
SurvivalGAN evaluation
====================================================

The CLI surface is BACKWARD-COMPATIBLE: the same `--real-train`, `--real-test`,
`--synthetic` / `--synthetic-dir`, `--time-column`, `--status-column`,
`--outdir`.
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
import warnings as _warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, spearmanr
from scipy.integrate import trapezoid as _scipy_trapezoid

# Headless backend so this works on EC2 with no X server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Synthcity — main metric runner.
from synthcity.metrics import Metrics
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader

# lifelines — KM, Nelson-Aalen, used both for plots and for our custom metrics.
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# Optional dependencies — degrade gracefully when missing.
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("eval")

# Suppress lifelines warnings about constant-extension of the survival curve
# beyond the latest event time — that's exactly what the paper specifies.
_warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================================== #
#  PUBLICATION STYLE                                                          #
# =========================================================================== #

PLOT_STYLE = {
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
}

# Method colors — consistent across all figures
METHOD_COLORS = {
    "real":      "#222222",
    "real_train":"#222222",
    "real_test": "#666666",
    "survgan":   "#1f77b4",
    "ctgan":     "#ff7f0e",
    "tvae":      "#2ca02c",
    "adsgan":    "#d62728",
    "nflows":    "#9467bd",
    "privbayes": "#8c564b",
    "_default":  "#17becf",
}

def _color(name: str) -> str:
    """Resolve a method name to a colour. Robust to case + path stems."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    for k, v in METHOD_COLORS.items():
        if k in key:
            return v
    return METHOD_COLORS["_default"]


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


# =========================================================================== #
#  SECTION 1 — TIERED METRICS CONFIGURATION                                   #
# =========================================================================== #

# Tier 1: cheap metrics — O(n) or small classifiers. Safe on full data.
METRICS_CHEAP: Dict[str, List[str]] = {
    "stats": [
        "jensenshannon_dist",
    ],
    "detection": [
        "detection_xgb",
        "detection_mlp",
    ],
    "performance": [
        # synthcity's performance.* on a SurvivalAnalysisDataLoader returns
        # C-Index for {gt, syn_id, syn_ood} AND survival-specific metrics
        # (optimism, km_distance, sightedness, brier_score) automatically.
        "linear_model",
        "xgb",
        "feat_rank_distance",
    ],
}

# Tier 2: heavy metrics — O(n²) memory. Bootstrap on subsamples.
METRICS_HEAVY: Dict[str, List[str]] = {
    "stats": [
        "max_mean_discrepancy",
        "wasserstein_dist",
        "prdc",
        "alpha_precision",
        "inv_kl_divergence",
    ],
}

# Tier 3: privacy — opt-in.
METRICS_PRIVACY: Dict[str, List[str]] = {
    "privacy": [
        "k_anonymization",
        "distinct_l_diversity",
        "identifiability_score",
    ],
}

METRICS_CONFIG = {**METRICS_CHEAP, **METRICS_HEAVY, **METRICS_PRIVACY}


# Dashboard: terminal table + headline numbers in summary CSV.
DASHBOARD_KEYS = [
    ("stats.jensenshannon_dist.marginal.mean", "JSD (lower=better)"),
    ("stats.prdc.precision.mean", "Precision"),
    ("stats.prdc.recall.mean", "Recall"),
    ("stats.alpha_precision.delta_precision_alpha_OC.mean", "α-Precision"),
    ("detection.detection_xgb.mean.mean", "Detection AUC XGB (0.5=best)"),
    ("detection.detection_mlp.mean.mean", "Detection AUC MLP (0.5=best)"),
    ("performance.xgb.syn_ood.mean", "Downstream C-Index (TSTR, syn→real)"),
    ("performance.feat_rank_distance.corr.mean", "Feat. importance corr"),
    # Survival-specific (custom keys we compute ourselves; see Section 3)
    ("custom.optimism", "Optimism (0=best)"),
    ("custom.km_divergence", "KM Divergence (0=best)"),
    ("custom.short_sightedness", "Short-sightedness (0=best)"),
]

KEY_ALIASES = {
    "JSD (lower=better)": [
        "stats.jensenshannon_dist.marginal.mean",
        "stats.jensenshannon_dist.mean",
        "stats.jensenshannon_dist.joint.mean",
    ],
    "Precision": [
        "stats.prdc.precision.mean",
        "performance.prdc.precision.mean",
    ],
    "Recall": [
        "stats.prdc.recall.mean",
        "performance.prdc.recall.mean",
    ],
    "α-Precision": [
        "stats.alpha_precision.delta_precision_alpha_OC.mean",
        "stats.alpha_precision.delta_precision_alpha_naive.mean",
        "stats.alpha_precision.authenticity_OC.mean",
    ],
    "Detection AUC XGB (0.5=best)": [
        "detection.detection_xgb.mean.mean",
        "detection.detection_xgb.mean",
    ],
    "Detection AUC MLP (0.5=best)": [
        "detection.detection_mlp.mean.mean",
        "detection.detection_mlp.mean",
    ],
    "Downstream C-Index (TSTR, syn→real)": [
        "performance.xgb.syn_ood.mean",
        "performance.xgb.syn_ood",
        "custom.xgb_tstr",
        "custom.cox_tstr",
    ],
    "Feat. importance corr": [
        "performance.feat_rank_distance.corr.mean",
        "performance.feat_rank_distance.corr",
        "custom.feat_imp_spearman",
    ],
    "Optimism (0=best)": ["custom.optimism"],
    "KM Divergence (0=best)": ["custom.km_divergence"],
    "Short-sightedness (0=best)": ["custom.short_sightedness"],
}


def _first_match(metrics: Dict, candidates: List[str]):
    for k in candidates:
        if k in metrics and metrics[k] is not None:
            return metrics[k]
    return None


# =========================================================================== #
#  SECTION 2 — DATA PREP HELPERS                                              #
# =========================================================================== #

def load_and_align(real_path: Path, syn_path: Path,
                   time_col: str, status_col: str,
                   real_time_range: Optional[Tuple[float, float]] = None,
                   clip_time_to_real: bool = True,
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load real + synthetic, intersect columns, coerce status to int."""
    real = pd.read_csv(real_path)
    syn = pd.read_csv(syn_path)

    common = [c for c in real.columns if c in syn.columns]
    missing_in_syn = [c for c in real.columns if c not in syn.columns]
    if missing_in_syn:
        log.warning(f"  Synthetic {syn_path.name} missing columns: {missing_in_syn}")

    real = real[common].copy()
    syn = syn[common].copy()

    if status_col in syn.columns:
        syn[status_col] = syn[status_col].round().clip(0, 1).astype(int)
    if status_col in real.columns:
        real[status_col] = real[status_col].round().clip(0, 1).astype(int)

    if time_col in syn.columns:
        syn[time_col] = syn[time_col].clip(lower=1e-3)
    if time_col in real.columns:
        real[time_col] = real[time_col].clip(lower=1e-3)

    if clip_time_to_real and real_time_range is not None and time_col in syn.columns:
        tmin, tmax = real_time_range
        tmin = max(float(tmin), 1e-3)
        tmax = max(float(tmax), tmin * 1.01)
        n_low = int((syn[time_col] < tmin).sum())
        n_high = int((syn[time_col] > tmax).sum())
        if n_low + n_high > 0:
            log.warning(
                f"  Clipping synthetic {time_col} to real range "
                f"[{tmin:.3f}, {tmax:.3f}]: {n_low} below, {n_high} above "
                f"(out of {len(syn)})"
            )
        syn[time_col] = syn[time_col].clip(lower=tmin, upper=tmax)

    return real, syn, common


def match_size(a: pd.DataFrame, b: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(a), len(b))
    rng = np.random.default_rng(seed)
    a_idx = rng.choice(len(a), size=n, replace=False) if len(a) > n else np.arange(n)
    b_idx = rng.choice(len(b), size=n, replace=False) if len(b) > n else np.arange(n)
    return a.iloc[a_idx].reset_index(drop=True), b.iloc[b_idx].reset_index(drop=True)


def dl_from_df(df: pd.DataFrame, time_col: str, status_col: str) -> SurvivalAnalysisDataLoader:
    return SurvivalAnalysisDataLoader(
        df,
        target_column=status_col,
        time_to_event_column=time_col,
        time_horizons=np.linspace(df[time_col].min(), df[time_col].max(), 5).tolist(),
    )


def detect_categorical(df: pd.DataFrame, max_unique: int = 20) -> List[str]:
    """Return the list of columns we treat as categorical for plotting.

    Heuristic: object dtype, OR low-cardinality numeric (<= max_unique unique
    values). This mirrors the BinEncoder logic in survGAN.py.
    """
    out: List[str] = []
    for c in df.columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_bool_dtype(s):
            out.append(c)
            continue
        try:
            n_unique = s.nunique(dropna=True)
            if n_unique <= max_unique:
                out.append(c)
        except Exception:
            pass
    return out


# =========================================================================== #
#  SECTION 3 — METRIC EVALUATION                                              #
# =========================================================================== #

def _flatten_metrics(results) -> Dict[str, float]:
    """Flatten synthcity's results DataFrame into {<index>.<column>: value}."""
    out: Dict[str, float] = {}
    if results is None:
        return out
    if hasattr(results, "to_dict"):
        d = results.to_dict()
        for col, row in d.items():
            for metric_name, val in row.items():
                out[f"{metric_name}.{col}"] = val
    else:
        out = dict(results)
    return out


def _mem_log(tag: str) -> None:
    if psutil is None:
        return
    try:
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        log.info(f"  [mem] {tag}: RSS={rss_gb:.2f} GB")
    except Exception:
        pass


def _subsample_pair(real: pd.DataFrame, syn: pd.DataFrame,
                    n: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    def _take(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) >= n:
            idx = rng.choice(len(df), size=n, replace=False)
        else:
            idx = rng.choice(len(df), size=n, replace=True)
        return df.iloc[idx].reset_index(drop=True)
    return _take(real), _take(syn)


def _aggregate_bootstrap(runs: List[Dict[str, float]]) -> Dict[str, float]:
    if not runs:
        return {}
    per_key: Dict[str, List[float]] = {}
    for r in runs:
        for k, v in r.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                per_key.setdefault(k, []).append(float(v))

    out: Dict[str, float] = {}
    for k, vals in per_key.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = float(arr.mean())
        if k.endswith(".mean"):
            base = k[: -len(".mean")]
            out[f"{base}.boot_mean"] = float(arr.mean())
            out[f"{base}.boot_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[f"{base}.boot_ci_lo"] = float(np.quantile(arr, 0.025))
            out[f"{base}.boot_ci_hi"] = float(np.quantile(arr, 0.975))
            out[f"{base}.boot_n"] = int(len(arr))
    return out


# --------------------------------------------------------------------------- #
#  Custom survival metrics — Optimism, KM Divergence, Short-sightedness.      #
#                                                                             #
#  These are implemented locally so the report doesn't depend on synthcity's  #
#  internal naming. Definitions follow Norcliffe et al. (AISTATS 2023):       #
#                                                                             #
#      Optimism       = (1/T) ∫₀ᵀ (S_syn(t) − S_real(t)) dt                   #
#      KM Divergence  = (1/T) ∫₀ᵀ |S_syn(t) − S_real(t)| dt                   #
#      Short-sighted  = max(0, (T_real − T_syn) / T_real)                     #
#                                                                             #
#  T is taken to be max(T_real, T_syn). Beyond either curve's last observed   #
#  time, lifelines extends the survival function as a constant — that         #
#  matches the paper's "constant rate of events" extrapolation closely        #
#  enough for our purposes.                                                   #
# --------------------------------------------------------------------------- #

def compute_survival_specific_metrics(
    real_t: np.ndarray, real_e: np.ndarray,
    syn_t: np.ndarray, syn_e: np.ndarray,
    n_grid: int = 500,
) -> Dict[str, float]:
    """Compute the three SurvivalGAN-paper survival-specific metrics."""
    real_t = np.asarray(real_t, dtype=float)
    real_e = np.asarray(real_e, dtype=int)
    syn_t  = np.asarray(syn_t,  dtype=float)
    syn_e  = np.asarray(syn_e,  dtype=int)

    if len(real_t) == 0 or len(syn_t) == 0:
        return {"optimism": np.nan, "km_divergence": np.nan, "short_sightedness": np.nan}

    kmf_r = KaplanMeierFitter().fit(real_t, real_e)
    kmf_s = KaplanMeierFitter().fit(syn_t, syn_e)

    T_real = float(real_t.max())
    T_syn  = float(syn_t.max())
    T = max(T_real, T_syn)
    if T <= 0:
        return {"optimism": np.nan, "km_divergence": np.nan, "short_sightedness": np.nan}

    grid = np.linspace(0.0, T, n_grid)

    # lifelines' .survival_function_at_times returns a Series with `times` index.
    s_r = kmf_r.survival_function_at_times(grid).values.astype(float)
    s_s = kmf_s.survival_function_at_times(grid).values.astype(float)

    diff = s_s - s_r
    optimism = float(_scipy_trapezoid(diff, grid) / T)
    km_div   = float(_scipy_trapezoid(np.abs(diff), grid) / T)
    short_s  = float(max(0.0, (T_real - T_syn) / T_real)) if T_real > 0 else 0.0

    return {
        "optimism": optimism,
        "km_divergence": km_div,
        "short_sightedness": short_s,
    }


def compute_marginal_extras(
    real: pd.DataFrame, syn: pd.DataFrame,
    time_col: str, status_col: str,
) -> Dict[str, float]:
    """Cheap marginal sanity metrics computed locally (no synthcity)."""
    out: Dict[str, float] = {}

    if status_col in real.columns and status_col in syn.columns:
        out["censoring_rate_diff"] = float(
            abs(real[status_col].mean() - syn[status_col].mean())
        )

    if time_col in real.columns and time_col in syn.columns:
        # 1-D Wasserstein on the time-axis split by event indicator.
        for ev_label, mask_real, mask_syn in [
            ("event", real[status_col] == 1, syn[status_col] == 1),
            ("censor", real[status_col] == 0, syn[status_col] == 0),
        ]:
            r = real.loc[mask_real, time_col].values
            s = syn.loc[mask_syn, time_col].values
            if len(r) > 0 and len(s) > 0:
                try:
                    out[f"time_wass_{ev_label}"] = float(wasserstein_distance(r, s))
                except Exception:
                    pass
    return out


# --------------------------------------------------------------------------- #
#  Native downstream performance — bypasses synthcity                         #
#                                                                             #
#  synthcity's `performance.xgb` / `performance.linear_model` /               #
#  `performance.feat_rank_distance` evaluators silently return empty results  #
#  when their internal cross-validation chokes on something (mixed dtypes,    #
#  high-cardinality categoricals, optional-dep version mismatch). The result  #
#  is that "Downstream C-Index" shows n/a in the dashboard with no traceback. #
#                                                                             #
#  We compute the same TRTR / TSTR / TSTS C-Indices natively here using       #
#  lifelines + xgboost — the same primitives the rest of the script uses,    #
#  so it always works whenever the rest of the script works.                  #
# --------------------------------------------------------------------------- #

def _prep_xy(df: pd.DataFrame, time_col: str,
             status_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Numericize features, fill NaN; return (X, t, e)."""
    drop = [c for c in (time_col, status_col) if c in df.columns]
    X = df.drop(columns=drop).copy()
    for c in X.columns:
        if X[c].dtype == object or pd.api.types.is_bool_dtype(X[c]):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    t = df[time_col].values.astype(float)
    e = df[status_col].values.astype(int)
    return X, t, e


def compute_downstream_performance(
    real_train: pd.DataFrame, real_test: pd.DataFrame, syn: pd.DataFrame,
    time_col: str, status_col: str,
    max_n: int = 20_000, seed: int = 42,
) -> Dict[str, float]:
    """TRTR / TSTR / TSTS C-Index for Cox and XGB.

    Returns keys:
      custom.cox_trtr, custom.cox_tstr, custom.cox_tsts
      custom.xgb_trtr, custom.xgb_tstr, custom.xgb_tsts
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    out: Dict[str, float] = {}
    rng = np.random.default_rng(seed)

    def _cap(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df
        return df.iloc[rng.choice(len(df), n, replace=False)].reset_index(drop=True)

    real_train = _cap(real_train, max_n)
    syn        = _cap(syn,        max_n)

    Xrt, trt, ert = _prep_xy(real_train, time_col, status_col)
    Xte, tte, ete = _prep_xy(real_test,  time_col, status_col)
    Xs,  ts,  es  = _prep_xy(syn,        time_col, status_col)

    common = [c for c in Xrt.columns if c in Xte.columns and c in Xs.columns]
    if not common:
        log.warning("  [downstream] no common feature columns across real/syn; skipped")
        return out
    Xrt, Xte, Xs = Xrt[common], Xte[common], Xs[common]

    # ---- Cox proportional hazards ----
    try:
        def _fit_cox(X, t, e):
            df = X.copy()
            df["__t"] = t
            df["__e"] = e
            cph = CoxPHFitter(penalizer=0.01)  # mild ridge for stability on real data
            cph.fit(df, duration_col="__t", event_col="__e", show_progress=False)
            return cph

        def _cox_score(cph, X):
            # negative partial hazard ≈ "predicted survival time"
            return -cph.predict_partial_hazard(X).values

        t0 = time.time()
        cph_r = _fit_cox(Xrt, trt, ert)
        out["custom.cox_trtr"] = float(concordance_index(tte, _cox_score(cph_r, Xte), ete))
        cph_s = _fit_cox(Xs, ts, es)
        out["custom.cox_tstr"] = float(concordance_index(tte, _cox_score(cph_s, Xte), ete))
        out["custom.cox_tsts"] = float(concordance_index(ts,  _cox_score(cph_s, Xs),  es))
        log.info(
            f"  [downstream] Cox  TRTR={out['custom.cox_trtr']:.4f}  "
            f"TSTR={out['custom.cox_tstr']:.4f}  "
            f"TSTS={out['custom.cox_tsts']:.4f}  ({time.time()-t0:.1f}s)"
        )
    except Exception as exc:
        log.warning(f"  [downstream] Cox failed: {type(exc).__name__}: {exc}")

    # ---- XGB time regressor (predicting log(t)) ----
    if _HAVE_XGB:
        try:
            def _fit_xgb(X, t):
                m = xgb.XGBRegressor(
                    n_estimators=300, max_depth=4, n_jobs=4,
                    verbosity=0, tree_method="hist",
                )
                m.fit(X.values, np.log(np.clip(t, 1e-3, None)))
                return m

            t0 = time.time()
            m_r = _fit_xgb(Xrt, trt)
            out["custom.xgb_trtr"] = float(concordance_index(tte, m_r.predict(Xte.values), ete))
            m_s = _fit_xgb(Xs, ts)
            out["custom.xgb_tstr"] = float(concordance_index(tte, m_s.predict(Xte.values), ete))
            out["custom.xgb_tsts"] = float(concordance_index(ts,  m_s.predict(Xs.values),  es))
            log.info(
                f"  [downstream] XGB  TRTR={out['custom.xgb_trtr']:.4f}  "
                f"TSTR={out['custom.xgb_tstr']:.4f}  "
                f"TSTS={out['custom.xgb_tsts']:.4f}  ({time.time()-t0:.1f}s)"
            )
        except Exception as exc:
            log.warning(f"  [downstream] XGB failed: {type(exc).__name__}: {exc}")

    return out


def run_metrics(real: pd.DataFrame, syn: pd.DataFrame,
                time_col: str, status_col: str,
                label: str,
                run_cheap: bool = True,
                run_heavy: bool = True,
                run_privacy: bool = False,
                cheap_max_n: Optional[int] = 50_000,
                heavy_n_sub: int = 5_000,
                heavy_n_bootstrap: int = 3,
                partial_save_path: Optional[Path] = None,
                seed: int = 42) -> Dict:
    """Tiered synthcity metrics + custom survival-specific metrics."""
    log.info(f"  [{label}] start  real_n={len(real)}  syn_n={len(syn)}")
    _mem_log(f"{label} start")
    merged: Dict[str, float] = {}

    def _persist():
        if partial_save_path is not None:
            try:
                with open(partial_save_path, "w") as f:
                    json.dump(merged, f, indent=2, default=str)
            except Exception:
                log.debug("Partial save failed", exc_info=True)

    # --- Custom survival metrics (cheap, run first so we always have them) ---
    try:
        if time_col in real.columns and status_col in real.columns:
            sm = compute_survival_specific_metrics(
                real[time_col].values, real[status_col].values,
                syn[time_col].values,  syn[status_col].values,
            )
            for k, v in sm.items():
                merged[f"custom.{k}"] = v
            log.info(
                f"  [{label}] custom survival metrics: "
                f"optim={sm['optimism']:+.4f}  "
                f"km_div={sm['km_divergence']:.4f}  "
                f"short={sm['short_sightedness']:.4f}"
            )
        extras = compute_marginal_extras(real, syn, time_col, status_col)
        for k, v in extras.items():
            merged[f"custom.{k}"] = v
        _persist()
    except Exception:
        log.error("  Custom survival metrics failed:")
        log.error(traceback.format_exc())

    # --- Tier 1: cheap synthcity ---
    if run_cheap:
        rt, st = real, syn
        if cheap_max_n is not None and (len(rt) > cheap_max_n or len(st) > cheap_max_n):
            rng = np.random.default_rng(seed)
            rt = rt.iloc[rng.choice(len(rt), min(len(rt), cheap_max_n), replace=False)].reset_index(drop=True)
            st = st.iloc[rng.choice(len(st), min(len(st), cheap_max_n), replace=False)].reset_index(drop=True)
            log.info(f"  [{label}] cheap tier capped to n={cheap_max_n}")

        real_dl = dl_from_df(rt, time_col, status_col)
        syn_dl = dl_from_df(st, time_col, status_col)

        for family, metric_names in METRICS_CHEAP.items():
            for metric in metric_names:
                t0 = time.time()
                try:
                    results = Metrics.evaluate(
                        real_dl, syn_dl, metrics={family: [metric]}
                    )
                    flat = _flatten_metrics(results)
                    merged.update(flat)
                    log.info(
                        f"  [{label}] cheap  {family}.{metric:<22}  "
                        f"ok in {time.time()-t0:5.1f}s  (+{len(flat)} keys)"
                    )
                except Exception as e:
                    log.error(
                        f"  [{label}] cheap  {family}.{metric} failed: "
                        f"{type(e).__name__}: {e}"
                    )
                gc.collect()
                _persist()
        _mem_log(f"{label} after cheap")

    # --- Tier 2: heavy / bootstrapped ---
    if run_heavy:
        n_sub = min(heavy_n_sub, len(real), len(syn))
        if n_sub < heavy_n_sub:
            log.warning(
                f"  [{label}] heavy tier: requested n_sub={heavy_n_sub} but "
                f"data only has n={n_sub}"
            )

        for family, metric_names in METRICS_HEAVY.items():
            for metric in metric_names:
                t0 = time.time()
                boot_runs: List[Dict[str, float]] = []
                for b in range(heavy_n_bootstrap):
                    try:
                        rt, st = _subsample_pair(real, syn, n=n_sub, seed=seed + b)
                        results = Metrics.evaluate(
                            dl_from_df(rt, time_col, status_col),
                            dl_from_df(st, time_col, status_col),
                            metrics={family: [metric]},
                        )
                        flat = _flatten_metrics(results)
                        boot_runs.append(flat)
                    except Exception as e:
                        log.error(
                            f"  [{label}] heavy  {family}.{metric} bootstrap "
                            f"#{b+1}/{heavy_n_bootstrap} failed: "
                            f"{type(e).__name__}: {e}"
                        )
                    finally:
                        gc.collect()

                if boot_runs:
                    agg = _aggregate_bootstrap(boot_runs)
                    merged.update(agg)
                    log.info(
                        f"  [{label}] heavy  {family}.{metric:<22}  "
                        f"ok in {time.time()-t0:5.1f}s  "
                        f"(n_sub={n_sub}, B={len(boot_runs)})"
                    )
                else:
                    log.error(
                        f"  [{label}] heavy  {family}.{metric} — all bootstraps failed"
                    )
                _persist()
        _mem_log(f"{label} after heavy")

    # --- Tier 3: privacy ---
    if run_privacy:
        n_sub = min(heavy_n_sub, len(real), len(syn))
        for family, metric_names in METRICS_PRIVACY.items():
            for metric in metric_names:
                t0 = time.time()
                boot_runs: List[Dict[str, float]] = []
                for b in range(heavy_n_bootstrap):
                    try:
                        rt, st = _subsample_pair(real, syn, n=n_sub, seed=seed + b)
                        results = Metrics.evaluate(
                            dl_from_df(rt, time_col, status_col),
                            dl_from_df(st, time_col, status_col),
                            metrics={family: [metric]},
                        )
                        boot_runs.append(_flatten_metrics(results))
                    except Exception as e:
                        log.error(
                            f"  [{label}] priv   {family}.{metric} bootstrap "
                            f"#{b+1} failed: {type(e).__name__}: {e}"
                        )
                    finally:
                        gc.collect()
                if boot_runs:
                    merged.update(_aggregate_bootstrap(boot_runs))
                    log.info(
                        f"  [{label}] priv   {family}.{metric:<22}  "
                        f"ok in {time.time()-t0:5.1f}s"
                    )
                _persist()
        _mem_log(f"{label} after privacy")

    if merged:
        sample = [k for k in merged.keys() if k.endswith(".mean") or k.startswith("custom.")][:8]
        log.info(f"  [{label}] example surfaced keys: {sample}")
    else:
        log.error(f"  [{label}] NO METRICS SUCCEEDED — see tracebacks above")

    _persist()
    return merged


# =========================================================================== #
#  SECTION 4 — PER-METHOD VISUALISATIONS                                      #
# =========================================================================== #

def plot_km_with_metrics(real_train: pd.DataFrame, real_test: pd.DataFrame,
                         syn: pd.DataFrame, time_col: str, status_col: str,
                         out_path: Path, method_name: str,
                         metrics: Optional[Dict] = None) -> None:
    """KM curves with 95% CI bands, plus annotated survival metrics.

    Replaces the previous plot_km_curves. Bands let the reader judge whether
    the synthetic curve is meaningfully different or just noisy.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for df, label, color, ls in [
        (real_train, "Real (train)", METHOD_COLORS["real_train"], "-"),
        (real_test,  "Real (test)",  METHOD_COLORS["real_test"],  "--"),
        (syn,        f"Synthetic ({method_name})", _color(method_name), "-"),
    ]:
        if len(df) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], df[status_col], label=label)
        kmf.plot_survival_function(
            ax=ax, color=color, linestyle=ls, ci_show=True, ci_alpha=0.15,
        )

    ax.set_title(f"Kaplan–Meier curves: {method_name}", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability $S(t)$")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", frameon=True)

    # Annotate the three SurvivalGAN paper metrics
    if metrics:
        ann = []
        for k, label_s in [
            ("custom.optimism", "Optimism"),
            ("custom.km_divergence", "KM Div."),
            ("custom.short_sightedness", "Short-sight."),
        ]:
            v = metrics.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                ann.append(f"{label_s:13s} = {v:+.4f}")
        if ann:
            txt = "\n".join(ann)
            ax.text(
                0.02, 0.04, txt, transform=ax.transAxes,
                fontfamily="monospace", fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="lightgray"),
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved KM plot → {out_path}")


def plot_distributions_grid(real: pd.DataFrame, syn: pd.DataFrame,
                            out_path: Path, method_name: str,
                            time_col: str, status_col: str,
                            max_cols: int = 16) -> None:
    """Per-column distribution overlays. Auto-detects categorical vs continuous.

    Replaces plot_histograms. Categorical columns get bar charts; continuous
    columns get density-normalised histograms. Time and status are always
    included first.
    """
    _apply_style()

    # Order: time + status first (always informative for survival data),
    # then by largest mean |difference| to surface the most interesting cols.
    cols = list(real.columns)
    head = [c for c in (time_col, status_col) if c in cols]
    rest = [c for c in cols if c not in head][:max_cols - len(head)]
    cols = head + rest

    cat_cols = set(detect_categorical(real))

    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes).flatten()

    real_color = METHOD_COLORS["real"]
    syn_color = _color(method_name)

    for i, c in enumerate(cols):
        ax = axes[i]
        r = real[c].dropna()
        s = syn[c].dropna()
        if len(r) == 0 or len(s) == 0:
            ax.set_title(f"{c}\n(empty)", fontsize=9)
            ax.axis("off")
            continue

        if c in cat_cols:
            # Bar chart of normalised counts side-by-side.
            r_counts = r.value_counts(normalize=True).sort_index()
            s_counts = s.value_counts(normalize=True).sort_index()
            cats = sorted(set(r_counts.index) | set(s_counts.index),
                          key=lambda x: str(x))
            x = np.arange(len(cats))
            w = 0.4
            r_y = [r_counts.get(k, 0.0) for k in cats]
            s_y = [s_counts.get(k, 0.0) for k in cats]
            ax.bar(x - w/2, r_y, width=w, color=real_color, label="Real", alpha=0.85)
            ax.bar(x + w/2, s_y, width=w, color=syn_color,  label="Synthetic", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([str(k)[:10] for k in cats], rotation=40,
                               ha="right", fontsize=7)
        else:
            try:
                lo, hi = np.percentile(np.concatenate([r, s]), [1, 99])
            except Exception:
                lo, hi = float(min(r.min(), s.min())), float(max(r.max(), s.max()))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(r.min()), float(r.max() + 1e-9)
            bins = np.linspace(lo, hi, 30)
            ax.hist(r, bins=bins, alpha=0.55, label="Real", color=real_color, density=True)
            ax.hist(s, bins=bins, alpha=0.55, label="Synthetic", color=syn_color, density=True)

        ax.set_title(c, fontsize=10)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Marginal distributions: {method_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved distributions plot → {out_path}")


def _encode_for_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Cast df to numeric for Spearman: ordinal-encode object columns."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or pd.api.types.is_bool_dtype(out[c]):
            out[c] = pd.Categorical(out[c]).codes.astype(float)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def plot_correlation_panel(real: pd.DataFrame, syn: pd.DataFrame,
                           out_path: Path, method_name: str,
                           max_cols: int = 25) -> None:
    """Three-panel Spearman correlation heatmap: real / synthetic / |diff|.

    Joint dependencies are invisible to per-column metrics — this panel
    surfaces them at a glance.
    """
    _apply_style()
    cols = list(real.columns)[:max_cols]
    R = _encode_for_corr(real[cols]).corr(method="spearman")
    S = _encode_for_corr(syn[cols]).corr(method="spearman")
    D = (S - R).abs()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap_corr = "RdBu_r"
    cmap_diff = "magma"

    for ax, M, title, cmap, vmin, vmax in [
        (axes[0], R, "Real (Spearman ρ)",      cmap_corr, -1, 1),
        (axes[1], S, "Synthetic (Spearman ρ)", cmap_corr, -1, 1),
        (axes[2], D, "|ρ_syn − ρ_real|",       cmap_diff,  0, 1),
    ]:
        im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=70, fontsize=7, ha="right")
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Correlation structure: {method_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved correlation panel → {out_path}")


def plot_time_split_histograms(real: pd.DataFrame, syn: pd.DataFrame,
                               time_col: str, status_col: str,
                               out_path: Path, method_name: str) -> None:
    """Two panels: time | E=1 (events) and time | E=0 (censoring).

    A faithful generator must reproduce both — getting the marginal of t
    right while mixing up event / censoring assignments still produces
    biased downstream models. Decoupling them shows the failure.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    real_color = METHOD_COLORS["real"]
    syn_color = _color(method_name)

    for ax, e_val, title in [
        (axes[0], 1, "Time | event = 1"),
        (axes[1], 0, "Time | event = 0 (censored)"),
    ]:
        r = real.loc[real[status_col] == e_val, time_col]
        s = syn.loc[syn[status_col] == e_val, time_col]
        if len(r) == 0 or len(s) == 0:
            ax.set_title(f"{title}\n(empty)", fontsize=10)
            ax.axis("off")
            continue
        try:
            lo, hi = np.percentile(np.concatenate([r, s]), [1, 99])
        except Exception:
            lo = float(min(r.min(), s.min()))
            hi = float(max(r.max(), s.max()))
        bins = np.linspace(lo, hi, 40)
        ax.hist(r, bins=bins, alpha=0.55, label=f"Real (n={len(r)})",
                color=real_color, density=True)
        ax.hist(s, bins=bins, alpha=0.55,
                label=f"Synthetic (n={len(s)})", color=syn_color, density=True)
        # Wasserstein-distance annotation, since this is the metric we report
        try:
            wd = wasserstein_distance(r.values, s.values)
            ax.text(0.97, 0.95, f"Wasserstein = {wd:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85,
                              edgecolor="lightgray"))
        except Exception:
            pass
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right" if e_val == 1 else "best")

    fig.suptitle(f"Time-by-event distributions: {method_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved time-split histograms → {out_path}")


def plot_feature_importance(real: pd.DataFrame, syn: pd.DataFrame,
                            time_col: str, status_col: str,
                            out_path: Path, method_name: str,
                            max_n: int = 10_000, seed: int = 42) -> Optional[float]:
    """Train two XGB time-regressors (one on real, one on syn), scatter the
    feature importances. Returns Spearman ρ for record. None if XGB missing.
    """
    if not _HAVE_XGB:
        log.warning("  xgboost not available; skipping feature-importance plot")
        return None
    _apply_style()

    feat_cols = [c for c in real.columns if c not in (time_col, status_col)]
    if not feat_cols:
        return None

    # Subsample for speed
    rng = np.random.default_rng(seed)
    if len(real) > max_n:
        real = real.iloc[rng.choice(len(real), max_n, replace=False)].reset_index(drop=True)
    if len(syn) > max_n:
        syn = syn.iloc[rng.choice(len(syn), max_n, replace=False)].reset_index(drop=True)

    def _prep(df):
        X = df[feat_cols].copy()
        # XGB needs numeric; ordinal-encode object cols
        for c in X.columns:
            if X[c].dtype == object or pd.api.types.is_bool_dtype(X[c]):
                X[c] = pd.Categorical(X[c]).codes.astype(float)
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        # Predict log(time) regression — same loss family as the SurvivalGAN
        # time regressor (no need for survival framing here, we're just
        # extracting feature relevance to time).
        y = np.log(np.clip(df[time_col].values, 1e-3, None))
        return X, y

    try:
        Xr, yr = _prep(real)
        Xs, ys = _prep(syn)
        m_r = xgb.XGBRegressor(n_estimators=200, max_depth=4, n_jobs=4,
                               verbosity=0, tree_method="hist")
        m_s = xgb.XGBRegressor(n_estimators=200, max_depth=4, n_jobs=4,
                               verbosity=0, tree_method="hist")
        m_r.fit(Xr, yr)
        m_s.fit(Xs, ys)
        imp_r = pd.Series(m_r.feature_importances_, index=feat_cols)
        imp_s = pd.Series(m_s.feature_importances_, index=feat_cols)
    except Exception as e:
        log.error(f"  feature-importance training failed: {type(e).__name__}: {e}")
        return None

    rho, _ = spearmanr(imp_r, imp_s)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(imp_r, imp_s, s=40, alpha=0.7,
               color=_color(method_name), edgecolors="black", linewidths=0.5)
    lim = max(imp_r.max(), imp_s.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="y = x")

    # Label the top-K features by mean importance
    K = min(8, len(feat_cols))
    top_feat = (imp_r + imp_s).nlargest(K).index
    for f in top_feat:
        ax.annotate(f, (imp_r[f], imp_s[f]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, alpha=0.85)

    ax.set_xlabel("Importance — XGB trained on REAL")
    ax.set_ylabel("Importance — XGB trained on SYNTHETIC")
    ax.set_title(f"Feature importance correspondence: {method_name}\n"
                 f"Spearman ρ = {rho:.3f}", fontweight="bold")
    ax.set_xlim(-0.005, lim)
    ax.set_ylim(-0.005, lim)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved feature-importance plot → {out_path}  (ρ={rho:.3f})")
    return float(rho)


def plot_tsne_embedding(real: pd.DataFrame, syn: pd.DataFrame,
                        time_col: str, status_col: str,
                        out_path: Path, method_name: str,
                        n_per_side: int = 2500, seed: int = 42) -> None:
    """2-D t-SNE of the covariate space (excluding time / status), real vs syn.

    Mirrors Appendix E.4 of the SurvivalGAN paper. Uses PCA-50 → t-SNE for
    speed on tabular data with mixed types.
    """
    if not _HAVE_SKLEARN:
        log.warning("  sklearn not available; skipping t-SNE")
        return
    _apply_style()

    feat_cols = [c for c in real.columns if c not in (time_col, status_col)]
    if not feat_cols:
        return

    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(real), min(n_per_side, len(real)), replace=False)
    s_idx = rng.choice(len(syn),  min(n_per_side, len(syn)),  replace=False)
    Xr = real.iloc[r_idx][feat_cols].copy()
    Xs = syn.iloc[s_idx][feat_cols].copy()

    for X in (Xr, Xs):
        for c in X.columns:
            if X[c].dtype == object or pd.api.types.is_bool_dtype(X[c]):
                X[c] = pd.Categorical(X[c]).codes.astype(float)
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
    Xr = Xr.fillna(Xr.median(numeric_only=True)).fillna(0.0).values
    Xs = Xs.fillna(Xs.median(numeric_only=True)).fillna(0.0).values

    try:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(np.vstack([Xr, Xs]))
        if X_all.shape[1] > 50:
            X_all = PCA(n_components=50, random_state=seed).fit_transform(X_all)
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=seed, learning_rate="auto").fit_transform(X_all)
    except Exception as e:
        log.error(f"  t-SNE failed: {type(e).__name__}: {e}")
        return

    n_r = len(Xr)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(emb[:n_r, 0], emb[:n_r, 1], s=8, alpha=0.5,
               color=METHOD_COLORS["real"], label=f"Real (n={n_r})", linewidths=0)
    ax.scatter(emb[n_r:, 0], emb[n_r:, 1], s=8, alpha=0.5,
               color=_color(method_name),
               label=f"Synthetic (n={len(Xs)})", linewidths=0)
    ax.set_title(f"t-SNE of covariate space: {method_name}", fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved t-SNE plot → {out_path}")


# =========================================================================== #
#  SECTION 5 — TERMINAL DASHBOARD                                             #
# =========================================================================== #

def ascii_hist(values: np.ndarray, bins: int = 20, width: int = 40, label: str = "") -> str:
    counts, edges = np.histogram(values, bins=bins)
    mx = counts.max() if counts.max() > 0 else 1
    out_lines = [f"  {label}  range=[{edges[0]:.2f}, {edges[-1]:.2f}]  n={len(values)}"]
    for i, c in enumerate(counts):
        bar = "█" * int(width * c / mx)
        out_lines.append(f"    {edges[i]:7.2f} | {bar} {c}")
    return "\n".join(out_lines)


def print_dashboard(method_name: str,
                    real_train: pd.DataFrame, real_test: pd.DataFrame, syn: pd.DataFrame,
                    metrics_train: Dict, metrics_test: Dict,
                    time_col: str, status_col: str) -> None:
    line = "=" * 78
    print(f"\n{line}")
    print(f"  RESULTS — {method_name}")
    print(line)

    def _fmt(metrics: Dict, aliases: List[str]) -> str:
        v = _first_match(metrics, aliases)
        if not isinstance(v, (int, float)):
            return "n/a"
        for a in aliases:
            if a.endswith(".mean"):
                std_key = a[:-len(".mean")] + ".boot_std"
                std = metrics.get(std_key)
                if isinstance(std, (int, float)) and std > 0:
                    return f"{v:.4f} ±{std:.4f}"
        return f"{v:.4f}"

    print(f"\n  {'Metric':<40} {'vs Train':>18} {'vs Test':>18}")
    print(f"  {'-'*40} {'-'*18} {'-'*18}")
    for key, display in DASHBOARD_KEYS:
        aliases = KEY_ALIASES.get(display, [key])
        s_tr = _fmt(metrics_train, aliases)
        s_te = _fmt(metrics_test, aliases)
        print(f"  {display:<40} {s_tr:>18} {s_te:>18}")

    # Verdict on detection AUC
    det_auc = _first_match(metrics_test, KEY_ALIASES["Detection AUC XGB (0.5=best)"])
    if det_auc is not None:
        if det_auc < 0.6:
            verdict = "✓ LOOKS GOOD — classifier can barely tell real from synthetic"
        elif det_auc < 0.75:
            verdict = "~ BORDERLINE — some distribution shift, usable for augmentation"
        else:
            verdict = "✗ CONCERNING — classifier easily distinguishes."
        print(f"\n  VERDICT: {verdict}")

    print(f"\n  Time distribution (column '{time_col}'):")
    print(ascii_hist(real_train[time_col].values, bins=15, label="real-train"))
    print(ascii_hist(syn[time_col].values, bins=15, label="synthetic "))

    print(f"\n  Event rate ({status_col}==1):")
    if status_col in real_train.columns:
        print(f"    real-train: {real_train[status_col].mean():.3f}")
    if status_col in real_test.columns:
        print(f"    real-test:  {real_test[status_col].mean():.3f}")
    if status_col in syn.columns:
        print(f"    synthetic:  {syn[status_col].mean():.3f}")
    print(line + "\n")


# =========================================================================== #
#  SECTION 6 — CROSS-METHOD TABLES                                            #
# =========================================================================== #

# Table specifications — list of (column_label, list_of_metric_keys) tuples.
# The first matching key from comparison.csv (with "test__" prefix) wins.

TABLE_COVARIATE = [
    ("JSD (↓)",            ["test__stats.jensenshannon_dist.marginal.mean"]),
    ("Wasserstein (↓)",    ["test__stats.wasserstein_dist.joint.mean",
                            "test__stats.wasserstein_dist.mean"]),
    ("MMD (↓)",            ["test__stats.max_mean_discrepancy.joint.mean",
                            "test__stats.max_mean_discrepancy.mean"]),
    ("Precision (↑)",      ["test__stats.prdc.precision.mean"]),
    ("Recall (↑)",         ["test__stats.prdc.recall.mean"]),
    ("Density (↑)",        ["test__stats.prdc.density.mean"]),
    ("Coverage (↑)",       ["test__stats.prdc.coverage.mean"]),
    ("α-Precision (↑)",    ["test__stats.alpha_precision.delta_precision_alpha_OC.mean",
                            "test__stats.alpha_precision.delta_precision_alpha_naive.mean"]),
    ("β-Recall (↑)",       ["test__stats.alpha_precision.delta_coverage_beta_OC.mean",
                            "test__stats.alpha_precision.delta_coverage_beta_naive.mean"]),
    ("Inv-KL (↑)",         ["test__stats.inv_kl_divergence.marginal.mean",
                            "test__stats.inv_kl_divergence.mean"]),
]

TABLE_SURVIVAL = [
    ("Optimism (→0)",      ["test__custom.optimism"]),
    ("KM Divergence (↓)",  ["test__custom.km_divergence"]),
    ("Short-sighted (↓)",  ["test__custom.short_sightedness"]),
    ("|Δ Cens. rate| (↓)", ["test__custom.censoring_rate_diff"]),
    ("W(t|E=1) (↓)",       ["test__custom.time_wass_event"]),
    ("W(t|E=0) (↓)",       ["test__custom.time_wass_censor"]),
]

TABLE_DETECTION = [
    ("Det. AUC XGB (→0.5)", ["test__detection.detection_xgb.mean.mean",
                             "test__detection.detection_xgb.mean"]),
    ("Det. AUC MLP (→0.5)", ["test__detection.detection_mlp.mean.mean",
                             "test__detection.detection_mlp.mean"]),
    ("Det. AUC GMM (→0.5)", ["test__detection.detection_gmm.mean.mean",
                             "test__detection.detection_gmm.mean"]),
]

TABLE_DOWNSTREAM = [
    ("C-Index TRTR (XGB) (↑)", ["test__performance.xgb.gt.mean",
                                "test__performance.xgb.gt",
                                "test__custom.xgb_trtr"]),
    ("C-Index TSTR (XGB) (↑)", ["test__performance.xgb.syn_ood.mean",
                                "test__performance.xgb.syn_ood",
                                "test__custom.xgb_tstr"]),
    ("C-Index TSTS (XGB) (↑)", ["test__performance.xgb.syn_id.mean",
                                "test__performance.xgb.syn_id",
                                "test__custom.xgb_tsts"]),
    ("C-Index TRTR (Cox) (↑)", ["test__performance.linear_model.gt.mean",
                                "test__performance.linear_model.gt",
                                "test__custom.cox_trtr"]),
    ("C-Index TSTR (Cox) (↑)", ["test__performance.linear_model.syn_ood.mean",
                                "test__performance.linear_model.syn_ood",
                                "test__custom.cox_tstr"]),
    ("Feat-rank ρ (↑)",        ["test__performance.feat_rank_distance.corr.mean",
                                "test__custom.feat_imp_spearman"]),
]


def _pick(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row.index:
            v = row[c]
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v)
    return None


def build_table(comp: pd.DataFrame, spec: List[Tuple[str, List[str]]],
                out_path: Path, name: str) -> pd.DataFrame:
    """Pivot comparison.csv columns into a clean publication table."""
    out_rows: List[Dict] = []
    for _, r in comp.iterrows():
        row: Dict = {"Method": r.get("method", "?"),
                     "n_synthetic": int(r.get("n_synthetic", 0))}
        for label, keys in spec:
            v = _pick(r, keys)
            row[label] = v
        out_rows.append(row)
    df = pd.DataFrame(out_rows)
    df.to_csv(out_path, index=False, float_format="%.5f")
    log.info(f"  Wrote {name} → {out_path}  ({len(df)} methods × {len(spec)} metrics)")
    return df


# =========================================================================== #
#  SECTION 7 — CROSS-METHOD VISUALISATIONS                                    #
# =========================================================================== #

def plot_cindex_brier(comp: pd.DataFrame, out_path: Path) -> None:
    """Two-panel bar chart: C-Index (TSTR vs TRTR baseline) and Brier."""
    _apply_style()
    methods = comp["method"].tolist()
    if not methods:
        return

    cind_trtr = [_pick(comp.iloc[i], ["test__performance.xgb.gt.mean",
                                       "test__performance.xgb.gt",
                                       "test__custom.xgb_trtr"])
                 for i in range(len(comp))]
    cind_tstr = [_pick(comp.iloc[i], ["test__performance.xgb.syn_ood.mean",
                                       "test__performance.xgb.syn_ood",
                                       "test__custom.xgb_tstr"])
                 for i in range(len(comp))]
    brier     = [_pick(comp.iloc[i], ["test__performance.xgb.brier_score.mean",
                                       "test__performance.xgb.brier_score"])
                 for i in range(len(comp))]
    boot_std  = [_pick(comp.iloc[i], ["test__performance.xgb.syn_ood.boot_std"])
                 for i in range(len(comp))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(methods))
    w = 0.35

    # Panel 1 — C-Index
    ax = axes[0]
    bars_trtr = ax.bar(x - w/2, _zerofill(cind_trtr), width=w,
                        color=METHOD_COLORS["real"],
                        label="TRTR (real → real)", alpha=0.85)
    err = [s if s is not None else 0 for s in boot_std]
    bars_tstr = ax.bar(x + w/2, _zerofill(cind_tstr), width=w,
                        color=[_color(m) for m in methods],
                        yerr=err, capsize=3, alpha=0.9,
                        label="TSTR (syn → real)")
    ax.axhline(0.5, color="gray", lw=1, ls=":", label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Concordance Index (C-Index)")
    ax.set_title("Downstream survival prediction (XGB)", fontweight="bold")
    ax.set_ylim(0.45, max(1.0, max(_zerofill(cind_trtr) + _zerofill(cind_tstr) + [0.5]) * 1.05))
    ax.legend(loc="lower right")

    # Panel 2 — Brier
    ax = axes[1]
    if any(b is not None for b in brier):
        ax.bar(x, _zerofill(brier), color=[_color(m) for m in methods], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Integrated Brier Score (lower = better)")
        ax.set_title("Calibration (XGB)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Brier score not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.axis("off")

    fig.suptitle("Cross-method downstream performance",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved C-Index/Brier comparison → {out_path}")


def _zerofill(xs: List[Optional[float]]) -> List[float]:
    return [0.0 if (x is None or not np.isfinite(x)) else float(x) for x in xs]


def plot_radar_summary(comp: pd.DataFrame, out_path: Path) -> None:
    """Radar plot summarising normalized performance across families.

    Each axis is a metric, normalized so that 1.0 = best across methods, 0 =
    worst. Lower-is-better metrics are inverted before normalising.
    """
    _apply_style()
    methods = comp["method"].tolist()
    if not methods:
        return

    # (label, key_candidates, lower_is_better)
    AXES = [
        ("Cov. fidelity\n(JSD)",
         ["test__stats.jensenshannon_dist.marginal.mean"], True),
        ("Coverage\n(Recall)",
         ["test__stats.prdc.recall.mean"], False),
        ("Sample quality\n(α-Prec)",
         ["test__stats.alpha_precision.delta_precision_alpha_OC.mean"], False),
        ("Indistinguish.\n(1−|AUC−.5|·2)",
         ["test__detection.detection_xgb.mean.mean",
          "test__detection.detection_xgb.mean"], False),  # special-cased below
        ("Survival fidelity\n(KM Div)",
         ["test__custom.km_divergence"], True),
        ("Calibration\n(|Optimism|)",
         ["test__custom.optimism"], True),  # absolute val taken below
        ("Downstream\n(TSTR C-Index)",
         ["test__performance.xgb.syn_ood.mean",
          "test__performance.xgb.syn_ood",
          "test__custom.xgb_tstr",
          "test__custom.cox_tstr"], False),
        ("Feat. Imp. ρ",
         ["test__performance.feat_rank_distance.corr.mean",
          "test__custom.feat_imp_spearman"], False),
    ]

    # Pull values
    vals: List[List[Optional[float]]] = []
    for _, r in comp.iterrows():
        row_vals: List[Optional[float]] = []
        for label, keys, lower_better in AXES:
            v = _pick(r, keys)
            if v is None:
                row_vals.append(None)
                continue
            if "Indistinguish" in label:
                # 1 − 2·|AUC−0.5|: 1 = perfectly indistinguishable, 0 = perfectly distinguishable.
                v = max(0.0, 1.0 - 2.0 * abs(v - 0.5))
            elif "Optimism" in label:
                v = abs(v)
            row_vals.append(v)
        vals.append(row_vals)

    # Normalise each axis to [0, 1] across methods, with lower_better inverted.
    arr = np.array([[v if v is not None else np.nan for v in r] for r in vals], dtype=float)
    norm = np.zeros_like(arr)
    for j, (label, _, lower_better) in enumerate(AXES):
        col = arr[:, j]
        valid = np.isfinite(col)
        if valid.sum() < 1:
            continue
        c_min, c_max = np.nanmin(col), np.nanmax(col)
        if c_max == c_min:
            norm[:, j] = 0.5
            continue
        # Indistinguishability and Optimism are already "higher better" after
        # the transforms above, so don't double-invert them.
        if "Indistinguish" in label or "Optimism" in label:
            norm[:, j] = (col - c_min) / (c_max - c_min)
        elif lower_better:
            norm[:, j] = 1.0 - (col - c_min) / (c_max - c_min)
        else:
            norm[:, j] = (col - c_min) / (c_max - c_min)
        norm[~valid, j] = 0.0  # treat missing as worst

    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)
    for i, m in enumerate(methods):
        v = norm[i].tolist() + [norm[i, 0]]
        ax.plot(angles, v, "o-", lw=2, color=_color(m), label=m, markersize=4)
        ax.fill(angles, v, alpha=0.10, color=_color(m))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a[0] for a in AXES], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Normalized performance by metric family\n(1.0 = best across methods, on each axis)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved radar summary → {out_path}")


def plot_metric_heatmap(comp: pd.DataFrame, out_path: Path) -> None:
    """Method × metric heatmap, color-coded green→red on per-column z-scores.

    Lower-is-better metrics are inverted so that GREEN = best universally.
    """
    _apply_style()
    methods = comp["method"].tolist()
    if not methods:
        return

    spec = [
        ("JSD",        ["test__stats.jensenshannon_dist.marginal.mean"], True),
        ("Wasserst.",  ["test__stats.wasserstein_dist.joint.mean",
                        "test__stats.wasserstein_dist.mean"], True),
        ("MMD",        ["test__stats.max_mean_discrepancy.joint.mean",
                        "test__stats.max_mean_discrepancy.mean"], True),
        ("Recall",     ["test__stats.prdc.recall.mean"], False),
        ("Coverage",   ["test__stats.prdc.coverage.mean"], False),
        ("α-Prec",     ["test__stats.alpha_precision.delta_precision_alpha_OC.mean"], False),
        ("Det.AUC",    ["test__detection.detection_xgb.mean.mean",
                        "test__detection.detection_xgb.mean"], None),  # special: target=0.5
        ("Optimism",   ["test__custom.optimism"], None),  # special: target=0
        ("KM Div",     ["test__custom.km_divergence"], True),
        ("Short-S.",   ["test__custom.short_sightedness"], True),
        ("TSTR C",     ["test__performance.xgb.syn_ood.mean",
                        "test__performance.xgb.syn_ood",
                        "test__custom.xgb_tstr"], False),
        ("Feat ρ",     ["test__performance.feat_rank_distance.corr.mean",
                        "test__custom.feat_imp_spearman"], False),
    ]

    raw = np.full((len(methods), len(spec)), np.nan)
    for i, (_, r) in enumerate(comp.iterrows()):
        for j, (_, keys, _) in enumerate(spec):
            v = _pick(r, keys)
            if v is not None:
                raw[i, j] = v

    # Normalise per-column to a "score" where higher = better.
    score = np.zeros_like(raw)
    for j, (label, _, lb) in enumerate(spec):
        col = raw[:, j]
        valid = np.isfinite(col)
        if valid.sum() < 1:
            continue
        if label == "Det.AUC":
            v = -np.abs(col - 0.5)
        elif label == "Optimism":
            v = -np.abs(col)
        elif lb is True:
            v = -col
        else:
            v = col
        v_min, v_max = np.nanmin(v), np.nanmax(v)
        if v_max == v_min:
            score[:, j] = 0.0
        else:
            score[:, j] = (v - v_min) / (v_max - v_min)
        score[~valid, j] = np.nan

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(spec)), 0.55 * max(3, len(methods)) + 1.5))
    cmap = "RdYlGn"
    im = ax.imshow(score, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate raw values
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            v = raw[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="black")

    ax.set_xticks(np.arange(len(spec)))
    ax.set_xticklabels([s[0] for s in spec], rotation=40, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Method × metric overview (green = best, red = worst per column)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Per-column rank score")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved metric heatmap → {out_path}")


# =========================================================================== #
#  SECTION 8 — SUBGROUP ANALYSIS (RQ3)                                        #
# =========================================================================== #

SUBGROUP_COL_CANDIDATES = ["Cancer Type", "cancer_type", "CancerType", "cancer"]


def find_subgroup_column(df: pd.DataFrame) -> Optional[str]:
    for c in SUBGROUP_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def run_subgroup_analysis(syn_paths: List[Path], real_test: pd.DataFrame,
                          time_col: str, status_col: str,
                          out_root: Path, top_k: int = 12,
                          min_n_per_group: int = 50) -> Optional[pd.DataFrame]:
    """Per-cancer-type metrics for each method.

    For each (method, cancer_type) pair where both real-test and synthetic
    have ≥ min_n_per_group rows, compute Optimism, KM Div, Short-sightedness,
    JSD over time, and the censoring-rate diff. Returns a tidy long-format
    DataFrame; also writes table5_subgroup.csv.
    """
    sub_col = find_subgroup_column(real_test)
    if sub_col is None:
        log.info("Subgroup analysis: no Cancer Type column in real_test — skipped.")
        return None

    # Identify top-K cancer types by real-test frequency
    top_groups = real_test[sub_col].value_counts().head(top_k).index.tolist()
    log.info(f"Subgroup analysis: using {len(top_groups)} most-frequent groups from "
             f"'{sub_col}': {top_groups}")

    rows: List[Dict] = []
    for sp in syn_paths:
        method = sp.stem
        try:
            syn = pd.read_csv(sp)
        except Exception as e:
            log.error(f"  subgroup: failed to load {sp.name}: {e}")
            continue

        if sub_col not in syn.columns:
            log.warning(f"  subgroup: '{sub_col}' missing in {sp.name}; skipping {method}")
            continue
        if status_col in syn.columns:
            syn[status_col] = syn[status_col].round().clip(0, 1).astype(int)
        if time_col in syn.columns:
            syn[time_col] = syn[time_col].clip(lower=1e-3)

        for g in top_groups:
            r_g = real_test[real_test[sub_col] == g]
            s_g = syn[syn[sub_col] == g]
            if len(r_g) < min_n_per_group or len(s_g) < min_n_per_group:
                rows.append({
                    "method": method, "cancer_type": str(g),
                    "n_real": len(r_g), "n_syn": len(s_g),
                    "optimism": np.nan, "km_divergence": np.nan,
                    "short_sightedness": np.nan,
                    "censoring_rate_diff": np.nan,
                    "real_event_rate": float(r_g[status_col].mean()) if len(r_g) > 0 else np.nan,
                    "syn_event_rate":  float(s_g[status_col].mean()) if len(s_g) > 0 else np.nan,
                })
                continue

            sm = compute_survival_specific_metrics(
                r_g[time_col].values, r_g[status_col].values,
                s_g[time_col].values, s_g[status_col].values,
            )
            rows.append({
                "method": method, "cancer_type": str(g),
                "n_real": len(r_g), "n_syn": len(s_g),
                "optimism": sm["optimism"],
                "km_divergence": sm["km_divergence"],
                "short_sightedness": sm["short_sightedness"],
                "censoring_rate_diff": float(abs(r_g[status_col].mean() - s_g[status_col].mean())),
                "real_event_rate": float(r_g[status_col].mean()),
                "syn_event_rate":  float(s_g[status_col].mean()),
            })

    df = pd.DataFrame(rows)
    out_path = out_root / "report" / "tables" / "table5_subgroup.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.5f")
    log.info(f"  Wrote subgroup table → {out_path}  ({len(df)} rows)")
    return df


def plot_subgroup_km_grid(syn_path: Path, real_test: pd.DataFrame,
                          time_col: str, status_col: str,
                          out_path: Path, focus_method: str,
                          top_k: int = 12, min_n: int = 50) -> None:
    """Grid of KM curves, one panel per cancer type, for ONE focus method."""
    sub_col = find_subgroup_column(real_test)
    if sub_col is None:
        return
    _apply_style()

    syn = pd.read_csv(syn_path)
    if sub_col not in syn.columns:
        log.info(f"  subgroup-KM: '{sub_col}' missing in {syn_path.name}; skipped")
        return
    if status_col in syn.columns:
        syn[status_col] = syn[status_col].round().clip(0, 1).astype(int)
    if time_col in syn.columns:
        syn[time_col] = syn[time_col].clip(lower=1e-3)

    groups = real_test[sub_col].value_counts().head(top_k).index.tolist()
    ncols = 4
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for i, g in enumerate(groups):
        ax = axes[i]
        r = real_test[real_test[sub_col] == g]
        s = syn[syn[sub_col] == g]
        if len(r) < min_n or len(s) < min_n:
            ax.text(0.5, 0.5,
                    f"{g}\n(n_real={len(r)}, n_syn={len(s)})\nbelow min_n={min_n}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        try:
            kmf_r = KaplanMeierFitter().fit(r[time_col], r[status_col], label="Real")
            kmf_s = KaplanMeierFitter().fit(s[time_col], s[status_col], label="Synthetic")
            kmf_r.plot_survival_function(ax=ax, color=METHOD_COLORS["real"],
                                         ci_show=True, ci_alpha=0.15)
            kmf_s.plot_survival_function(ax=ax, color=_color(focus_method),
                                         ci_show=True, ci_alpha=0.15)
        except Exception:
            ax.text(0.5, 0.5, "fit failed", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        ax.set_title(f"{g}  (n={len(r)}/{len(s)})", fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower left")

    for j in range(len(groups), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Per-cancer-type KM curves — focus method: {focus_method}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved subgroup KM grid → {out_path}")


def plot_subgroup_metric_heatmap(subgroup_df: pd.DataFrame, out_path: Path,
                                 metric: str = "km_divergence") -> None:
    """Methods (rows) × cancer types (cols) heatmap of one subgroup metric."""
    _apply_style()
    if subgroup_df is None or subgroup_df.empty:
        return
    pivot = subgroup_df.pivot_table(index="method", columns="cancer_type",
                                     values=metric, aggfunc="mean")
    if pivot.empty:
        return

    # Sort columns by their mean to put the easiest groups on the left
    col_order = pivot.mean(axis=0).sort_values().index.tolist()
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * pivot.shape[1] + 3),
                                     0.55 * max(3, pivot.shape[0]) + 1.5))
    cmap = "viridis_r"
    vmax = np.nanpercentile(pivot.values, 95) if np.isfinite(pivot.values).any() else 1.0
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                color = "white" if v > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color=color)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Per-cancer-type {metric} (lower = better)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  Saved subgroup metric heatmap → {out_path}")


# =========================================================================== #
#  SECTION 9 — PREFLIGHT (unchanged)                                          #
# =========================================================================== #

def preflight_check(syn: pd.DataFrame, real_train: pd.DataFrame,
                    time_col: str, status_col: str,
                    strict: bool = False) -> bool:
    print("\n  --- Preflight sanity check ---")
    errors: List[str] = []
    warnings: List[str] = []

    for c in (time_col, status_col):
        if c not in syn.columns:
            errors.append(f"column '{c}' missing from synthetic CSV")
    if errors:
        for e in errors:
            print(f"    ✗ {e}")
        return False

    na_t = int(syn[time_col].isna().sum())
    na_s = int(syn[status_col].isna().sum())
    if na_t > 0.01 * len(syn):
        warnings.append(f"{na_t}/{len(syn)} synthetic time values are NaN")
    if na_s > 0.01 * len(syn):
        warnings.append(f"{na_s}/{len(syn)} synthetic status values are NaN")

    if time_col in real_train.columns:
        r_lo = float(real_train[time_col].min())
        r_hi = float(real_train[time_col].max())
        s = pd.to_numeric(syn[time_col], errors="coerce").dropna()
        s_lo, s_hi = float(s.min()), float(s.max())
        n_below = int((s < r_lo * 0.5).sum())
        n_above = int((s > r_hi * 2.0).sum())
        print(f"    time: real=[{r_lo:.3f}, {r_hi:.3f}]  syn=[{s_lo:.3f}, {s_hi:.3f}]")
        if n_below + n_above > 0.1 * len(s):
            errors.append(
                f"{n_below} syn times < {r_lo*0.5:.3f} and {n_above} > "
                f"{r_hi*2:.3f} (>10% out of range — TTE model is broken)"
            )
        elif n_below + n_above > 0:
            warnings.append(
                f"{n_below + n_above} synthetic times slightly out of real range"
            )
        s_q05, s_q95 = s.quantile([0.05, 0.95])
        if s_q95 - s_q05 < 0.05 * (r_hi - r_lo):
            errors.append(
                f"synthetic time column is collapsed: "
                f"q05..q95 = [{s_q05:.3f}, {s_q95:.3f}] spans less than "
                f"5% of real range"
            )

    s_vals = pd.to_numeric(syn[status_col], errors="coerce").dropna().unique()
    s_vals_rounded = set(int(round(v)) for v in s_vals if np.isfinite(v))
    if not s_vals_rounded.issubset({0, 1}):
        warnings.append(f"status column has non-binary values after rounding: {s_vals_rounded}")

    if time_col in real_train.columns:
        r_ev = real_train[status_col].mean() if status_col in real_train else None
        s_ev = pd.to_numeric(syn[status_col], errors="coerce").round().clip(0, 1).mean()
        if r_ev is not None:
            print(f"    event rate: real={r_ev:.3f}  syn={s_ev:.3f}")
            if abs(r_ev - s_ev) > 0.15:
                warnings.append(
                    f"event rate differs by > 0.15 (real={r_ev:.3f}, syn={s_ev:.3f})"
                )

    for w in warnings:
        print(f"    ⚠  {w}")
    for e in errors:
        print(f"    ✗  {e}")

    if errors:
        print("    Preflight FAILED — data is broken. Fix before running metrics.")
        return False
    if warnings and strict:
        print("    Preflight failed under --strict-preflight.")
        return False
    if warnings:
        print("    Preflight passed with warnings — proceeding.")
    else:
        print("    ✓ Preflight passed.")
    return True


# =========================================================================== #
#  SECTION 10 — PER-METHOD ORCHESTRATION                                      #
# =========================================================================== #

def evaluate_one_method(syn_path: Path,
                        real_train: pd.DataFrame, real_test: pd.DataFrame,
                        outdir: Path, time_col: str, status_col: str,
                        seed: int = 42,
                        clip_time_to_real: bool = True,
                        run_cheap: bool = True,
                        run_heavy: bool = True,
                        run_privacy: bool = False,
                        cheap_max_n: int = 50_000,
                        heavy_n_sub: int = 5_000,
                        heavy_n_bootstrap: int = 3,
                        eval_vs_train: bool = True,
                        eval_vs_test: bool = True,
                        strict_preflight: bool = False,
                        preflight_only: bool = False,
                        skip_plots: bool = False,
                        skip_tsne: bool = False,
                        skip_corr: bool = False) -> Dict:
    method = syn_path.stem
    log.info(f"\n{'='*60}\nEvaluating method: {method}\n{'='*60}")
    method_out = outdir / "per_method" / method
    method_out.mkdir(parents=True, exist_ok=True)

    real_time_range = None
    if time_col in real_train.columns:
        real_time_range = (
            float(real_train[time_col].min()),
            float(real_train[time_col].max()),
        )

    _, syn, common = load_and_align(
        syn_path, syn_path, time_col, status_col,
        real_time_range=real_time_range,
        clip_time_to_real=clip_time_to_real,
    )
    real_train_aligned = real_train[[c for c in real_train.columns if c in syn.columns]].copy()
    real_test_aligned = real_test[[c for c in real_test.columns if c in syn.columns]].copy()

    ok = preflight_check(syn, real_train_aligned, time_col, status_col,
                          strict=strict_preflight)
    if preflight_only:
        log.info("--preflight-only set; skipping metrics.")
        return {"method": method, "n_synthetic": len(syn), "preflight_ok": ok}
    if not ok and strict_preflight:
        log.error(f"Preflight failed for {method} under --strict-preflight — skipping metrics.")
        return {"method": method, "n_synthetic": len(syn), "preflight_ok": False}

    m_train: Dict = {}
    m_test: Dict = {}

    if eval_vs_train:
        rt, st = match_size(real_train_aligned, syn, seed=seed)
        m_train = run_metrics(
            rt, st, time_col, status_col, label="vs train",
            run_cheap=run_cheap, run_heavy=run_heavy, run_privacy=run_privacy,
            cheap_max_n=cheap_max_n, heavy_n_sub=heavy_n_sub,
            heavy_n_bootstrap=heavy_n_bootstrap,
            partial_save_path=method_out / "metrics_vs_train.json",
            seed=seed,
        )

    if eval_vs_test:
        re, se = match_size(real_test_aligned, syn, seed=seed + 1)
        m_test = run_metrics(
            re, se, time_col, status_col, label="vs test",
            run_cheap=run_cheap, run_heavy=run_heavy, run_privacy=run_privacy,
            cheap_max_n=cheap_max_n, heavy_n_sub=heavy_n_sub,
            heavy_n_bootstrap=heavy_n_bootstrap,
            partial_save_path=method_out / "metrics_vs_test.json",
            seed=seed + 1,
        )

    # ---- Native downstream performance (independent of synthcity) ----
    # Always run when both real-train and real-test are available — this is
    # cheap (~30s for 20k rows) and guarantees the dashboard/tables get
    # populated even when synthcity's performance.* family silently returns
    # empty results (which it does on some env/version combos).
    if eval_vs_test and len(real_train_aligned) > 0:
        try:
            ds = compute_downstream_performance(
                real_train_aligned, real_test_aligned, syn,
                time_col, status_col, seed=seed,
            )
            m_test.update(ds)
        except Exception:
            log.error("Native downstream performance failed:")
            log.error(traceback.format_exc())

    # ---- Plots (replace the legacy km_curves.png + histograms.png) ----
    if not skip_plots:
        try:
            plot_km_with_metrics(real_train_aligned, real_test_aligned, syn,
                                 time_col, status_col,
                                 method_out / "km_curves.png", method,
                                 metrics=m_test or m_train)
        except Exception:
            log.error("KM plot failed:"); log.error(traceback.format_exc())

        try:
            plot_distributions_grid(real_train_aligned, syn,
                                    method_out / "distributions.png", method,
                                    time_col=time_col, status_col=status_col)
        except Exception:
            log.error("Distributions plot failed:"); log.error(traceback.format_exc())

        try:
            plot_time_split_histograms(real_train_aligned, syn, time_col, status_col,
                                       method_out / "time_split_histograms.png", method)
        except Exception:
            log.error("Time-split plot failed:"); log.error(traceback.format_exc())

        if not skip_corr:
            try:
                plot_correlation_panel(real_train_aligned, syn,
                                       method_out / "correlation_heatmap.png", method)
            except Exception:
                log.error("Correlation plot failed:"); log.error(traceback.format_exc())

        try:
            rho = plot_feature_importance(real_train_aligned, syn, time_col, status_col,
                                          method_out / "feature_importance.png", method,
                                          seed=seed)
            if rho is not None:
                # Surface our own ρ alongside synthcity's, in case they disagree
                m_test["custom.feat_imp_spearman"] = rho
        except Exception:
            log.error("Feat-importance plot failed:"); log.error(traceback.format_exc())

        if not skip_tsne:
            try:
                plot_tsne_embedding(real_train_aligned, syn, time_col, status_col,
                                    method_out / "tsne_embedding.png", method, seed=seed)
            except Exception:
                log.error("t-SNE plot failed:"); log.error(traceback.format_exc())

    # ---- Dashboard ----
    print_dashboard(method, real_train_aligned, real_test_aligned, syn,
                    m_train, m_test, time_col, status_col)

    # ---- Persist final metrics ----
    with open(method_out / "metrics_vs_train.json", "w") as f:
        json.dump(m_train, f, indent=2, default=str)
    with open(method_out / "metrics_vs_test.json", "w") as f:
        json.dump(m_test, f, indent=2, default=str)

    # Flatten for cross-method comparison
    row = {"method": method, "n_synthetic": len(syn)}
    for k, v in m_train.items():
        row[f"train__{k}"] = v
    for k, v in m_test.items():
        row[f"test__{k}"] = v
    return row


# =========================================================================== #
#  SECTION 11 — REPORT BUILDER (cross-method)                                 #
# =========================================================================== #

def build_report(comp: pd.DataFrame, syn_paths: List[Path],
                 real_test: pd.DataFrame, time_col: str, status_col: str,
                 outdir: Path, focus_method: Optional[str] = None) -> None:
    """Build the cross-method report directory: tables + figures + subgroup."""
    rep_root = outdir / "report"
    fig_dir = rep_root / "figures"
    tab_dir = rep_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tables ----
    log.info("\n" + "=" * 60)
    log.info("Building cross-method report tables")
    log.info("=" * 60)
    build_table(comp, TABLE_COVARIATE,  tab_dir / "table1_covariate_quality.csv", "Table 1 (covariate)")
    build_table(comp, TABLE_SURVIVAL,   tab_dir / "table2_survival_quality.csv",  "Table 2 (survival)")
    build_table(comp, TABLE_DETECTION,  tab_dir / "table3_detection.csv",         "Table 3 (detection)")
    build_table(comp, TABLE_DOWNSTREAM, tab_dir / "table4_downstream.csv",        "Table 4 (downstream)")

    # ---- Cross-method figures ----
    try:
        plot_cindex_brier(comp, fig_dir / "cindex_brier_comparison.png")
    except Exception:
        log.error("C-Index/Brier comparison failed:"); log.error(traceback.format_exc())

    try:
        plot_radar_summary(comp, fig_dir / "radar_summary.png")
    except Exception:
        log.error("Radar summary failed:"); log.error(traceback.format_exc())

    try:
        plot_metric_heatmap(comp, fig_dir / "metric_heatmap.png")
    except Exception:
        log.error("Metric heatmap failed:"); log.error(traceback.format_exc())

    # ---- Subgroup analysis ----
    log.info("\nRunning subgroup (per-cancer-type) analysis")
    sub_df = run_subgroup_analysis(syn_paths, real_test, time_col, status_col, outdir)

    if sub_df is not None and not sub_df.empty:
        try:
            plot_subgroup_metric_heatmap(sub_df, fig_dir / "subgroup_metric_heatmap.png",
                                         metric="km_divergence")
        except Exception:
            log.error("Subgroup heatmap failed:"); log.error(traceback.format_exc())

        # Pick a focus method for the subgroup KM grid (default: best TSTR C-Index)
        if focus_method is None:
            tstr = comp.set_index("method").get("test__performance.xgb.syn_ood.mean")
            if tstr is not None and tstr.notna().any():
                focus_method = tstr.idxmax()
            else:
                focus_method = comp["method"].iloc[0]
        focus_path = next((p for p in syn_paths if p.stem == focus_method), syn_paths[0])
        try:
            plot_subgroup_km_grid(focus_path, real_test, time_col, status_col,
                                  fig_dir / "subgroup_km_grid.png", focus_method)
        except Exception:
            log.error("Subgroup KM grid failed:"); log.error(traceback.format_exc())


# =========================================================================== #
#  SECTION 12 — MAIN + CLI                                                    #
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication-ready evaluator for synthetic survival data")
    p.add_argument("--real-train", required=True, help="Path to real training CSV")
    p.add_argument("--real-test", required=True, help="Path to real held-out test CSV")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--synthetic", help="Path to a single synthetic CSV")
    grp.add_argument("--synthetic-dir", help="Directory of synthetic CSVs (one per method)")

    p.add_argument("--outdir", default="eval_results_gan")
    p.add_argument("--time-column", default="time")
    p.add_argument("--status-column", default="status")
    p.add_argument("--seed", type=int, default=42)

    # Tier / memory controls
    p.add_argument("--no-heavy", dest="run_heavy", action="store_false")
    p.set_defaults(run_heavy=True)
    p.add_argument("--no-cheap", dest="run_cheap", action="store_false")
    p.set_defaults(run_cheap=True)
    p.add_argument("--privacy", dest="run_privacy", action="store_true")
    p.set_defaults(run_privacy=False)

    p.add_argument("--cheap-max-n", type=int, default=50_000)
    p.add_argument("--heavy-n-sub", type=int, default=5_000)
    p.add_argument("--heavy-n-bootstrap", type=int, default=3)

    p.add_argument("--vs-train-only", action="store_true")
    p.add_argument("--vs-test-only", action="store_true")

    # Plot controls (additive, all default ON to maintain backwards-compat)
    p.add_argument("--skip-plots", action="store_true",
                   help="Skip ALL per-method plots (metrics only).")
    p.add_argument("--skip-tsne", action="store_true",
                   help="Skip t-SNE plots (slow on >10k samples).")
    p.add_argument("--skip-corr", action="store_true",
                   help="Skip correlation panels (slow with many cols).")
    p.add_argument("--no-report", dest="build_report", action="store_false",
                   help="Skip the cross-method report (tables + summary figures).")
    p.set_defaults(build_report=True)
    p.add_argument("--focus-method",
                   help="Method name (filename stem) to use for the per-cancer-type "
                        "KM grid in the report. Defaults to the method with the "
                        "highest TSTR C-Index.")

    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--strict-preflight", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(outdir / "eval.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 60)
    log.info("Publication-ready synthetic survival evaluator")
    log.info(f"Args: {vars(args)}")
    log.info("=" * 60)

    log.info(f"Loading real-train: {args.real_train}")
    real_train = pd.read_csv(args.real_train)
    log.info(f"  Shape: {real_train.shape}")
    log.info(f"Loading real-test: {args.real_test}")
    real_test = pd.read_csv(args.real_test)
    log.info(f"  Shape: {real_test.shape}")

    for df in (real_train, real_test):
        if args.status_column in df.columns:
            df[args.status_column] = df[args.status_column].round().clip(0, 1).astype(int)
        if args.time_column in df.columns:
            df[args.time_column] = df[args.time_column].clip(lower=1e-3)

    if args.synthetic:
        syn_paths = [Path(args.synthetic)]
    else:
        syn_dir = Path(args.synthetic_dir)
        syn_paths = sorted(syn_dir.glob("*.csv"))
        if not syn_paths:
            log.error(f"No CSVs found in {syn_dir}")
            return 2
        log.info(f"Batch mode: found {len(syn_paths)} synthetic files")
        for p in syn_paths:
            log.info(f"  • {p.name}")

    rows: List[Dict] = []
    eval_vs_train = not args.vs_test_only
    eval_vs_test = not args.vs_train_only
    if not eval_vs_train and not eval_vs_test:
        log.error("Both --vs-train-only and --vs-test-only set; nothing to evaluate.")
        return 2

    for sp in syn_paths:
        try:
            row = evaluate_one_method(
                sp, real_train, real_test, outdir,
                args.time_column, args.status_column, args.seed,
                run_cheap=args.run_cheap,
                run_heavy=args.run_heavy,
                run_privacy=args.run_privacy,
                cheap_max_n=args.cheap_max_n,
                heavy_n_sub=args.heavy_n_sub,
                heavy_n_bootstrap=args.heavy_n_bootstrap,
                eval_vs_train=eval_vs_train,
                eval_vs_test=eval_vs_test,
                strict_preflight=args.strict_preflight,
                preflight_only=args.preflight_only,
                skip_plots=args.skip_plots,
                skip_tsne=args.skip_tsne,
                skip_corr=args.skip_corr,
            )
            rows.append(row)
        except Exception:
            log.error(f"Evaluation failed for {sp.name}; continuing with others.")
            log.error(traceback.format_exc())

    # ---- Comparison.csv (unchanged behavior) ----
    if rows:
        comp = pd.DataFrame(rows)
        priority = ["method", "n_synthetic"]
        for k, _ in DASHBOARD_KEYS:
            for prefix in ("train__", "test__"):
                col = prefix + k
                if col in comp.columns:
                    priority.append(col)
        rest = [c for c in comp.columns if c not in priority]
        comp = comp[priority + rest]
        comp_path = outdir / "comparison.csv"
        comp.to_csv(comp_path, index=False)
        log.info(f"\nWrote cross-method comparison → {comp_path}")

        # On-screen summary
        print("\n" + "=" * 78)
        print("  CROSS-METHOD COMPARISON (key metrics, vs test set)")
        print("=" * 78)
        key_cols = ["method"] + [f"test__{k}" for k, _ in DASHBOARD_KEYS if f"test__{k}" in comp.columns]
        display = comp[key_cols].copy()
        display.columns = ["method"] + [label for k, label in DASHBOARD_KEYS if f"test__{k}" in comp.columns]
        with pd.option_context("display.max_columns", None,
                               "display.width", 220,
                               "display.float_format", lambda x: f"{x:.4f}"):
            print(display.to_string(index=False))
        print("=" * 78 + "\n")

        # ---- Cross-method report ----
        if args.build_report and not args.preflight_only:
            try:
                build_report(comp, syn_paths, real_test,
                              args.time_column, args.status_column,
                              outdir, focus_method=args.focus_method)
                log.info(f"Report ready → {outdir / 'report'}")
            except Exception:
                log.error("Report build failed:"); log.error(traceback.format_exc())

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
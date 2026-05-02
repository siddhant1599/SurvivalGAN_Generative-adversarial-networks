import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import random
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset, Sampler
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
#  TTE (time-to-event) model
# --------------------------------------------------------------------------- #
from synthcity.plugins.core.models.survival_analysis import (
    get_model_template as get_surv_model_template,
)
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =========================================================================== #
#  LOCAL TTE MODEL — replaces synthcity's buggy SurvivalFunctionTimeToEvent   #
# =========================================================================== #

class LocalSurvivalFunctionTTE:
    """
    Two-stage time-to-event model:
      Stage 1: DeepHit learns S(t|X) survival function
      Stage 2: XGBRegressor predicts log(T) from [X, S(t|X), E]
    """

    def __init__(self, device: Any = "cpu", time_points: int = 100,
                 n_estimators: int = 500, max_depth: int = 6,
                 add_residual_noise: bool = False,
                 noise_scale: float = 0.5,
                 clamp_margin: float = 0.05):

        self.device = device
        self.time_points = time_points
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.add_residual_noise = add_residual_noise
        self.noise_scale = noise_scale
        self.clamp_margin = clamp_margin

        # DeepHit from synthcity's survival_analysis (not time_to_event)
        self.surv_model = get_surv_model_template("deephit")(device=device)
        self.tte_regressor = None
        self.time_horizons = None
        self._residual_std = 0.0
        # Training log-time bounds. Used to clamp predictions so OOD synthetic
        # covariates can't produce times in the millions or tiny fractions.
        self._Tlog_min = None
        self._Tlog_max = None
        self._T_min = None
        self._T_max = None

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "LocalSurvivalFunctionTTE":
        # Stage 1: fit DeepHit
        self.surv_model.fit(X, T, Y)

        self.time_horizons = np.linspace(
            T.min(), T.max(), self.time_points
        ).tolist()

        surv_fn = self.surv_model.predict(X, time_horizons=self.time_horizons)
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        data = X.copy()
        data[surv_fn.columns] = surv_fn
        data["_event_indicator"] = Y

        # Guard against T <= 0 in training data. log(0 + 1e-8) = -18.42 which
        # poisons the regression target.
        T_pos = T.clip(lower=1e-3)
        Tlog = np.log(T_pos)

        xgb_params = {
            "n_jobs": 2,
            "n_estimators": self.n_estimators,
            "verbosity": 0,
            "max_depth": self.max_depth,
            "random_state": 0,
        }

        self.tte_regressor = XGBRegressor(**xgb_params).fit(data, Tlog)

        # Store residual std for optional noise injection at predict time
        pred_log = self.tte_regressor.predict(data)
        self._residual_std = float(np.std(Tlog.values - pred_log))

        # Store training log-time bounds for clamping at predict time.
        self._Tlog_min = float(Tlog.min())
        self._Tlog_max = float(Tlog.max())
        self._T_min = float(T_pos.min())
        self._T_max = float(T_pos.max())

        log.info(
            f"TTE fit: T=[{self._T_min:.3f}, {self._T_max:.3f}]  "
            f"logT=[{self._Tlog_min:.3f}, {self._Tlog_max:.3f}]  "
            f"residual_std(log)={self._residual_std:.4f}  "
            f"noise={'on' if self.add_residual_noise else 'off'}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.predict_any(X, pd.Series([1] * len(X), index=X.index))

    # ------------------------------------------------------------------ #
    #  Backward-compat attribute access: older pickles don't have the new  #
    #  fields. We fall back to sane defaults so unpickled models keep      #
    #  working.                                                            #
    # ------------------------------------------------------------------ #
    def _get(self, name: str, default: Any) -> Any:
        return getattr(self, name, default) if name in self.__dict__ else default

    def predict_any(self, X: pd.DataFrame, E: pd.Series) -> pd.Series:
        surv_fn = self.surv_model.predict(X, time_horizons=self.time_horizons)
        surv_fn = surv_fn.loc[:, ~surv_fn.columns.duplicated()]

        data = X.copy()
        data[surv_fn.columns] = surv_fn
        data["_event_indicator"] = E

        pred_log = np.asarray(self.tte_regressor.predict(data), dtype=float)

        # Resolve backward-compat state for old pickles.
        add_noise = self._get("add_residual_noise", False)
        noise_scale = self._get("noise_scale", 0.5)
        clamp_margin = self._get("clamp_margin", 0.05)
        Tlog_min = self._get("_Tlog_min", None)
        Tlog_max = self._get("_Tlog_max", None)

        if add_noise and self._residual_std > 0:
            pred_log = pred_log + np.random.normal(
                0, self._residual_std * noise_scale, size=len(pred_log)
            )

        if Tlog_min is not None and Tlog_max is not None:
            span = max(Tlog_max - Tlog_min, 1e-6)
            lo = Tlog_min - clamp_margin * span
            hi = Tlog_max + clamp_margin * span
            pre_clip_low = int((pred_log < lo).sum())
            pre_clip_high = int((pred_log > hi).sum())
            pred_log = np.clip(pred_log, lo, hi)
            if pre_clip_low + pre_clip_high > 0:
                log.info(
                    f"TTE predict: clamped {pre_clip_low} below / "
                    f"{pre_clip_high} above log-T range "
                    f"[{lo:.3f}, {hi:.3f}] (of {len(pred_log)} total)"
                )

        out = np.exp(pred_log)
        # Final check: exp of a clamped value should already be in range,
        # but keep it as an explicit guard in case of NaN/inf or extreme edge.
        if self._T_min is not None and self._T_max is not None:
            out = np.clip(out, self._T_min * 0.5, self._T_max * 2.0)

        return pd.Series(out, index=X.index)


def retrofit_tte_bounds(tte: "LocalSurvivalFunctionTTE",
                        T_train: pd.Series,
                        add_residual_noise: bool = False,
                        noise_scale: float = 0.5,
                        clamp_margin: float = 0.05) -> "LocalSurvivalFunctionTTE":
    ""
    T_pos = T_train.clip(lower=1e-3)
    Tlog = np.log(T_pos)
    tte._Tlog_min = float(Tlog.min())
    tte._Tlog_max = float(Tlog.max())
    tte._T_min = float(T_pos.min())
    tte._T_max = float(T_pos.max())
    tte.add_residual_noise = add_residual_noise
    tte.noise_scale = noise_scale
    tte.clamp_margin = clamp_margin
    log.info(
        f"retrofit_tte_bounds: T=[{tte._T_min:.3f}, {tte._T_max:.3f}]  "
        f"logT=[{tte._Tlog_min:.3f}, {tte._Tlog_max:.3f}]  "
        f"noise={'on' if tte.add_residual_noise else 'off'}"
    )
    return tte


# =========================================================================== #
#  SECTION 1 — CONFIGURATION                                                  #
#  Adjust hyperparameters here.                                                #
# =========================================================================== #

@dataclass
class Config:
    # ---- Data ----
    input_csv: str = "rotterdam_2232_survival.csv"
    target_column: str = "status"          # event indicator (0/1)
    time_column: str = "time"              # time-to-event

    # ---- GAN architecture  >>> ARCHITECTURE — tweak these freely ----
    n_iter: int = 3000
    batch_size: int = 256
    generator_n_layers_hidden: int = 2  #3
    generator_n_units_hidden: int = 128 #256
    generator_nonlin: str = "gelu"       # try: relu, elu, selu, tanh, gelu, silu
    generator_dropout: float = 0.0
    generator_residual: bool = True            # flip to False for plain MLP
    generator_batch_norm: bool = False
    generator_lr: float = 1e-3
    generator_weight_decay: float = 1e-4
    generator_opt_betas: tuple = (0.5, 0.999)

    discriminator_n_layers_hidden: int = 3
    discriminator_n_units_hidden: int = 256 #256
    discriminator_nonlin: str = "leaky_relu"
    discriminator_n_iter: int = 1  #5
    discriminator_dropout: float = 0.1
    discriminator_batch_norm: bool = False
    discriminator_lr: float = 1e-3
    discriminator_weight_decay: float = 1e-5
    discriminator_opt_betas: tuple = (0.5, 0.999)

    # ---- GAN training ----
    clipping_value: int = 1
    lambda_gradient_penalty: float = 10.0
    lambda_identifiability_penalty: float = 0.1

    # ---- Tabular encoding ----
    encoder_max_clusters: int = 5              # BayesianGMM components per continuous col

    # ---- Survival pipeline ----
    tte_strategy: str =  "survival_function"    # "survival_function" or "uncensoring"
    uncensoring_model: str = "survival_function_regression"
    censoring_strategy: str = "covariate_dependent"         # "random" or "covariate_dependent"
    use_survival_conditional: bool = True
    sampling_strategy: str = "imbalanced_time_censoring"  # none, imbalanced_censoring, imbalanced_time_censoring

    # ---- Generation ----
    synthetic_count: int = 2000
    output_csv: str = "synthetic_survival_data_rotterdam_v2.csv"

    # ---- System ----
    seed: int = 42
    device: str = "auto"


# =========================================================================== #
#  SECTION 2 — ACTIVATION FACTORY                                              #
#  Add custom activations here.                                                #
# =========================================================================== #

class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax for differentiable discrete sampling."""
    def __init__(self, tau: float = 0.2, hard: bool = False, dim: int = -1):
        super().__init__()
        self.tau, self.hard, self.dim = tau, hard, dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.gumbel_softmax(
            logits, tau=self.tau, hard=self.hard, dim=self.dim,
        )


# >>> ARCHITECTURE — add your own activation functions to this dict
ACTIVATIONS = {
    "none":        lambda: nn.Identity(),
    "elu":         lambda: nn.ELU(),
    "relu":        lambda: nn.ReLU(),
    "leaky_relu":  lambda: nn.LeakyReLU(),
    "selu":        lambda: nn.SELU(),
    "tanh":        lambda: nn.Tanh(),
    "sigmoid":     lambda: nn.Sigmoid(),
    "softmax":     lambda: GumbelSoftmax(),
    "gelu":        lambda: nn.GELU(),
    "silu":        lambda: nn.SiLU(),       # a.k.a. swish
    "prelu":       lambda: nn.PReLU(),
    "relu6":       lambda: nn.ReLU6(),
    "softplus":    lambda: nn.Softplus(),
    "hardtanh":    lambda: nn.Hardtanh(),
}


def get_nonlin(name: str) -> nn.Module:
    key = name.lower().replace("_", "")
    # normalise common aliases
    aliases = {"leakyrelu": "leaky_relu", "swish": "silu"}
    key = aliases.get(key, key)
    # try both raw and normalised
    for k in (name, key, name.lower().replace("_", "")):
        if k in ACTIVATIONS:
            return ACTIVATIONS[k]()
    raise ValueError(f"Unknown activation: {name}")


# =========================================================================== #
#  SECTION 3 — MULTI-ACTIVATION HEAD                                           #
# =========================================================================== #

class MultiActivationHead(nn.Module):
    """Final layer that applies different activations to different output slices.
    Useful for tabular data where some outputs are continuous (identity/tanh)
    and others are discrete (softmax)."""

    def __init__(self, activations: List[Tuple[nn.Module, int]], device: Any = "cpu"):
        super().__init__()
        self.activations = nn.ModuleList()
        self.lengths: List[int] = []
        self.device = device
        for act, length in activations:
            self.activations.append(act)
            self.lengths.append(length)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(X)
        idx = 0
        for act, length in zip(self.activations, self.lengths):
            out[..., idx:idx + length] = act(X[..., idx:idx + length])
            idx += length
        return out


# =========================================================================== #
#  SECTION 4 — MLP (Generator / Discriminator backbone)                        #
#
# =========================================================================== #

class LinearLayer(nn.Module):
    """Single hidden layer: [Dropout?] → Linear → [BatchNorm?] → Activation."""

    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        nonlin: Optional[str] = "relu",
        device: Any = "cpu",
    ):
        super().__init__()
        self.device = device
        layers: list = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(n_units_in, n_units_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_units_out))
        if nonlin is not None:
            layers.append(get_nonlin(nonlin))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.float())


class ResidualLinearLayer(nn.Module):
    """LinearLayer wrapped with a skip (residual) connection.
    Output dim = n_units_out + n_units_in  (concatenation-style skip)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        device = kwargs.get("device", "cpu")
        self.inner = LinearLayer(*args, **kwargs)
        self.device = device

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] == 0:
            out_feats = None
            for m in self.inner.model:
                if isinstance(m, nn.Linear):
                    out_feats = m.out_features
                    break
            return torch.zeros((*X.shape[:-1], out_feats or 0), device=self.device)
        out = self.inner(X)
        return torch.cat([out, X], dim=-1)


class MLP(nn.Module):
    """
    ARCHITECTURE — the core network used for both Generator and Discriminator.

    Modify this class to experiment with:
      • Depth & width (n_layers_hidden, n_units_hidden)
      • Activation functions (nonlin)
      • Residual connections (residual=True → concatenation-style skip)
      • Batch normalization
      • Dropout
      • Or replace entirely with your own architecture!

    Parameters
    ----------
    n_units_in   : input dimension
    n_units_out  : output dimension
    n_layers_hidden : number of hidden layers
    n_units_hidden  : width of each hidden layer
    nonlin       : activation name (see ACTIVATIONS dict)
    nonlin_out   : per-slice output activations [(act_name, width), ...]
    residual     : use concatenation skip connections
    batch_norm   : use batch normalization
    dropout      : dropout probability (0 = off)
    """

    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        *,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 256,
        nonlin: str = "relu",
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        opt_betas: tuple = (0.9, 0.999),
        device: Any = "cpu",
        random_state: int = 0,
        # kept for API compat; unused here
        task_type: str = "regression",
    ):
        super().__init__()
        self.device = device

        Block = ResidualLinearLayer if residual else LinearLayer

        layers: List[nn.Module] = []
        cur_in = n_units_in

        if n_layers_hidden > 0:
            # First hidden layer
            layers.append(Block(
                cur_in, n_units_hidden,
                batch_norm=batch_norm, nonlin=nonlin, device=device,
            ))
            cur_in = n_units_hidden + (cur_in if residual else 0)

            # Intermediate hidden layers
            for _ in range(n_layers_hidden - 1):
                layers.append(Block(
                    cur_in, n_units_hidden,
                    batch_norm=batch_norm, nonlin=nonlin,
                    dropout=dropout, device=device,
                ))
                cur_in = n_units_hidden + (cur_in if residual else 0)

            # Output projection
            layers.append(nn.Linear(cur_in, n_units_out, device=device))
        else:
            layers.append(nn.Linear(n_units_in, n_units_out, device=device))

        # Output activations (per-slice, for tabular data)
        if nonlin_out is not None:
            acts = [(get_nonlin(name), length) for name, length in nonlin_out]
            layers.append(MultiActivationHead(acts, device=device))

        self.model = nn.Sequential(*layers).to(device)

        # Optimizer (attached to the network for convenience, as in synthcity)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay, betas=opt_betas,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.float())


# =========================================================================== #
#  SECTION 5 — GAN TRAINING LOOP                                              #
#  >>> ARCHITECTURE — modify the loss functions or training procedure here.    #
# =========================================================================== #

class GAN(nn.Module):
    """Wasserstein GAN with gradient penalty + optional identifiability penalty.

    The generator and discriminator are both MLP instances (Section 4).
    """

    def __init__(
        self,
        n_features: int,
        n_units_latent: int,
        n_units_conditional: int = 0,
        # generator
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 250,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        generator_n_iter: int = 2000,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.0,
        generator_lr: float = 2e-4,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_extra_penalties: Optional[list] = None,
        generator_extra_penalty_cbks: Optional[List[Callable]] = None,
        # discriminator
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_lr: float = 2e-4,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        # training
        batch_size: int = 64,
        random_state: int = 0,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10.0,
        lambda_identifiability_penalty: float = 0.1,
        dataloader_sampler: Optional[Sampler] = None,
        device: Any = "cpu",
        # early stopping (not used when patience_metric is None)
        n_iter_min: int = 100,
        n_iter_print: int = 50,
        patience: int = 20,
        patience_metric: Any = None,  # set to None to disable
        **kwargs: Any,
    ):
        super().__init__()
        if generator_extra_penalties is None:
            generator_extra_penalties = []
        if generator_extra_penalty_cbks is None:
            generator_extra_penalty_cbks = []

        self.device = device
        self.n_features = n_features
        self.n_units_latent = n_units_latent
        self.n_units_conditional = n_units_conditional
        self.generator_extra_penalties = generator_extra_penalties
        self.generator_extra_penalty_cbks = generator_extra_penalty_cbks

        # ---- Build Generator ---- >>> ARCHITECTURE — swap this MLP for your own network
        self.generator = MLP(
            n_units_in=n_units_latent + n_units_conditional,
            n_units_out=n_features,
            n_layers_hidden=generator_n_layers_hidden,
            n_units_hidden=generator_n_units_hidden,
            nonlin=generator_nonlin,
            nonlin_out=generator_nonlin_out,
            batch_norm=generator_batch_norm,
            dropout=generator_dropout,
            residual=generator_residual,
            lr=generator_lr,
            weight_decay=generator_weight_decay,
            opt_betas=generator_opt_betas,
            device=device,
        ).to(device)

        # ---- Build Discriminator ---- >>> ARCHITECTURE — swap this MLP for your own network
        self.discriminator = MLP(
            n_units_in=n_features + n_units_conditional,
            n_units_out=1,
            n_layers_hidden=discriminator_n_layers_hidden,
            n_units_hidden=discriminator_n_units_hidden,
            nonlin=discriminator_nonlin,
            nonlin_out=[("none", 1)],
            batch_norm=discriminator_batch_norm,
            dropout=discriminator_dropout,
            residual=False,   # discriminator typically has no skip connections
            lr=discriminator_lr,
            weight_decay=discriminator_weight_decay,
            opt_betas=discriminator_opt_betas,
            device=device,
        ).to(device)

        # Training config
        self.generator_n_iter = generator_n_iter
        self.discriminator_n_iter = discriminator_n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.patience_metric = patience_metric  # None → no metric-based early stopping
        self.batch_size = batch_size
        self.clipping_value = clipping_value
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.lambda_identifiability_penalty = lambda_identifiability_penalty
        self.random_state = random_state
        self.dataloader_sampler = dataloader_sampler

        self._original_cond: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        cond: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "GAN":
        Xt = self._to_tensor(X).float()
        condt = self._to_tensor(cond).float() if cond is not None else None

        if self.n_units_conditional > 0:
            if condt is None:
                raise ValueError("Expecting conditional for training")
            if condt.shape[1] != self.n_units_conditional:
                raise ValueError(
                    f"Conditional dim mismatch: expected {self.n_units_conditional}, got {condt.shape[1]}"
                )

        self._train(Xt, condt)
        return self

    def generate(self, count: int, cond: Optional[np.ndarray] = None) -> np.ndarray:
        self.generator.eval()
        condt = self._to_tensor(cond).float() if cond is not None else None
        with torch.no_grad():
            return self._forward(count, condt).detach().cpu().numpy()

    def _forward(
        self, count: int, cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is None and self.n_units_conditional > 0:
            if self._original_cond is None:
                raise ValueError("No original conditional stored.")
            idxs = torch.randint(len(self._original_cond), (count,))
            cond = self._original_cond[idxs]
        if cond is not None and len(cond.shape) == 1:
            cond = cond.reshape(-1, 1)

        noise = torch.randn(count, self.n_units_latent, device=self.device)
        noise = self._cat_cond(noise, cond)
        return self.generator(noise)

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #

    def _train(self, X: torch.Tensor, cond: Optional[torch.Tensor] = None) -> None:
        self._original_cond = cond

        # Optional train/test split for early stopping
        X_train, X_val, cond_train, cond_val = self._train_test_split(X, cond)

        # DataLoader
        if cond_train is None:
            dataset = TensorDataset(X_train)
        else:
            dataset = TensorDataset(X_train, cond_train)

        loader = TorchDataLoader(
            dataset, batch_size=self.batch_size,
            sampler=self.dataloader_sampler, pin_memory=False,
        )

        for i in tqdm(range(self.generator_n_iter), desc="GAN training"):
            g_loss, d_loss = self._train_epoch(loader)

            if (i + 1) % self.n_iter_print == 0:
                log.debug(
                    f"[{i}/{self.generator_n_iter}] D_loss={d_loss:.4f}  G_loss={g_loss:.4f}"
                )

    def _train_epoch(self, loader: TorchDataLoader) -> Tuple[float, float]:
        G_losses, D_losses = [], []
        for data in loader:
            cond = data[1] if self.n_units_conditional > 0 else None
            X = data[0]

            D_losses.append(self._step_discriminator(X, cond))
            G_losses.append(self._step_generator(X, cond))

        return float(np.mean(G_losses)), float(np.mean(D_losses))

    # ---- Generator step ---- >>> ARCHITECTURE — modify generator loss here
    def _step_generator(self, X: torch.Tensor, cond: Optional[torch.Tensor]) -> float:
        self.generator.train()
        self.generator.optimizer.zero_grad()

        real_X_raw = X.to(self.device)
        real_X = self._cat_cond(real_X_raw, cond)
        bs = len(real_X)

        noise = torch.randn(bs, self.n_units_latent, device=self.device)
        noise = self._cat_cond(noise, cond)
        fake_raw = self.generator(noise)
        fake = self._cat_cond(fake_raw, cond)

        output = self.discriminator(fake).squeeze().float()
        errG = -torch.mean(output)

        # Extra callback losses (e.g. conditional loss from TabularGAN)
        for cb in self.generator_extra_penalty_cbks:
            errG += cb(real_X_raw, fake_raw, cond=cond)

        # Identifiability penalty (AdsGAN)
        if "identifiability_penalty" in self.generator_extra_penalties:
            errG += self._loss_identifiability(real_X, fake)

        errG.backward()
        if self.clipping_value > 0:
            nn.utils.clip_grad_norm_(self.generator.parameters(), self.clipping_value)
        self.generator.optimizer.step()

        if torch.isnan(errG):
            raise RuntimeError("NaN in generator loss")
        return errG.item()

    # ---- Discriminator step ---- >>> ARCHITECTURE — modify discriminator loss here
    def _step_discriminator(self, X: torch.Tensor, cond: Optional[torch.Tensor]) -> float:
        self.discriminator.train()
        bs = min(self.batch_size, len(X))
        errors = []

        for _ in range(self.discriminator_n_iter):
            real_X = self._cat_cond(X.to(self.device), cond)
            real_output = self.discriminator(real_X).squeeze().float()

            noise = torch.randn(bs, self.n_units_latent, device=self.device)
            noise = self._cat_cond(noise, cond)
            fake_raw = self.generator(noise)
            fake = self._cat_cond(fake_raw, cond)
            fake_output = self.discriminator(fake.detach()).squeeze()

            # Wasserstein loss
            errD = -torch.mean(real_output) + torch.mean(fake_output)

            # Gradient penalty
            gp = self._loss_gradient_penalty(real_X, fake, bs)

            self.discriminator.optimizer.zero_grad()
            gp.backward(retain_graph=True)
            errD.backward()
            if self.clipping_value > 0:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clipping_value)
            self.discriminator.optimizer.step()

            errors.append(errD.item())

        if np.isnan(np.mean(errors)):
            raise RuntimeError("NaN in discriminator loss")
        return float(np.mean(errors))

    # ---- Losses ----

    def _loss_gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor, bs: int,
    ) -> torch.Tensor:
        alpha = torch.rand(bs, 1, device=self.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.discriminator(interp).squeeze(-1)
        labels = torch.ones(len(interp), device=self.device)
        grads = torch.autograd.grad(
            d_interp, interp, grad_outputs=labels,
            create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True,
        )[0]
        grads = grads.view(grads.size(0), -1)
        penalty = ((grads.norm(2, dim=-1) - 1) ** 2).mean()
        return self.lambda_gradient_penalty * penalty

    def _loss_identifiability(
        self, real: torch.Tensor, fake: torch.Tensor,
    ) -> torch.Tensor:
        return (
            -self.lambda_identifiability_penalty
            * (real - fake).square().sum(dim=-1).sqrt().mean()
        )

    # ---- Helpers ----

    def _cat_cond(self, X: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        return torch.cat([X, cond], dim=1) if cond is not None else X

    def _to_tensor(self, X: Any) -> torch.Tensor:
        if X is None:
            return None
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        return torch.from_numpy(np.asarray(X)).to(self.device)

    def _train_test_split(self, X, cond):
        """80/20 split only when patience_metric is set; otherwise use all data."""
        if self.patience_metric is None:
            return X, None, cond, None

        n = len(X)
        idx = np.arange(n)
        np.random.shuffle(idx)
        split = int(n * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]

        X_train = X[train_idx]
        X_val = X[val_idx]
        cond_train = cond[train_idx] if cond is not None else None
        cond_val = cond[val_idx] if cond is not None else None
        return X_train, X_val, cond_train, cond_val


# =========================================================================== #
#  SECTION 6 — TABULAR ENCODER                                                #
#  Encodes mixed continuous/discrete columns for the GAN.                      #
#                                                                              #
#  Continuous → BayesianGMM: [normalized_value, component_prob_0, ..., _n-1]   #
#  Discrete   → OneHot: [0, 0, 1, 0, ...]                                     #
# =========================================================================== #

def _find_discrete_columns(df: pd.DataFrame, limit: int = 10) -> List[str]:
    """Columns with ≤ `limit` unique values are treated as discrete."""
    return [c for c in df.columns if df[c].nunique() <= limit]


class BayesianGMMFeatureEncoder:
    """Encodes a single continuous feature using BayesianGMM (CTGAN-style).

    Output per sample:  [normalized_value, p(comp_0), p(comp_1), ...]
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, feature: pd.Series) -> "BayesianGMMFeatureEncoder":
        self.feature_name_in = feature.name
        vals = feature.values.astype(float).reshape(-1, 1)

        self.bgm = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior=0.001,
            max_iter=300,
            n_init=1,
            random_state=0,
        )
        self.bgm.fit(vals)

        self.means = self.bgm.means_.flatten()
        self.stds = np.sqrt(self.bgm.covariances_.flatten())
        self.stds = np.clip(self.stds, 1e-6, None)

        self.n_features_out = 1 + self.n_components
        self.feature_names_out = [f"{self.feature_name_in}.norm"] + [
            f"{self.feature_name_in}.comp_{i}" for i in range(self.n_components)
        ]
        self.feature_types_out = ["continuous"] + ["discrete"] * self.n_components
        return self

    def transform(self, feature: pd.Series) -> pd.DataFrame:
        vals = feature.values.astype(float).reshape(-1, 1)
        probs = self.bgm.predict_proba(vals)   # (n, n_components)
        comps = probs.argmax(axis=1)

        # Normalize using selected component
        mu = self.means[comps]
        sigma = self.stds[comps]
        normed = (vals.flatten() - mu) / (4 * sigma)
        normed = np.clip(normed, -0.99, 0.99)
        normed = (normed + 1) / 2  # shift to [0, 1]

        # One-hot component
        onehot = np.zeros((len(vals), self.n_components))
        onehot[np.arange(len(vals)), comps] = 1.0

        out = np.column_stack([normed, onehot])
        return pd.DataFrame(out, columns=self.feature_names_out, index=feature.index)

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        normed = data.iloc[:, 0].values
        comp_probs = data.iloc[:, 1:].values
        comps = comp_probs.argmax(axis=1)

        v = normed * 2 - 1  # back to [-1, 1]
        mu = self.means[comps]
        sigma = self.stds[comps]
        vals = v * (4 * sigma) + mu
        return pd.Series(vals, name=self.feature_name_in, index=data.index)


class OneHotFeatureEncoder:
    """Encodes a single discrete feature using OneHot."""

    def fit(self, feature: pd.Series) -> "OneHotFeatureEncoder":
        self.feature_name_in = feature.name
        self.enc = SklearnOneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.enc.fit(feature.values.reshape(-1, 1))

        cats = self.enc.categories_[0]
        self.n_features_out = len(cats)
        self.feature_names_out = [f"{self.feature_name_in}.{c}" for c in cats]
        self.feature_types_out = ["discrete"] * self.n_features_out
        return self

    def transform(self, feature: pd.Series) -> pd.DataFrame:
        out = self.enc.transform(feature.values.reshape(-1, 1))
        return pd.DataFrame(out, columns=self.feature_names_out, index=feature.index)

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        inv = self.enc.inverse_transform(data.values)
        return pd.Series(inv.flatten(), name=self.feature_name_in, index=data.index)


class _FeatureInfo:
    """Metadata for a single encoded feature."""
    def __init__(self, name, feature_type, transform, output_dimensions,
                 transformed_features, trans_feature_types):
        self.name = name
        self.feature_type = feature_type
        self.transform = transform
        self.output_dimensions = output_dimensions
        self.transformed_features = transformed_features
        self.trans_feature_types = trans_feature_types


class TabularEncoder:
    """Encodes a DataFrame with mixed continuous/discrete columns."""

    def __init__(self, max_clusters: int = 10, categorical_limit: int = 10):
        self.max_clusters = max_clusters
        self.categorical_limit = categorical_limit

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        disc_cols = _find_discrete_columns(df, self.categorical_limit)
        self._column_raw_dtypes = df.dtypes
        self._info_list: List[_FeatureInfo] = []
        self.output_dimensions = 0

        for col in df.columns:
            ftype = "discrete" if col in disc_cols else "continuous"
            if ftype == "discrete":
                enc = OneHotFeatureEncoder().fit(df[col])
            else:
                enc = BayesianGMMFeatureEncoder(n_components=self.max_clusters).fit(df[col])

            info = _FeatureInfo(
                name=col, feature_type=ftype, transform=enc,
                output_dimensions=enc.n_features_out,
                transformed_features=enc.feature_names_out,
                trans_feature_types=enc.feature_types_out,
            )
            self._info_list.append(info)
            self.output_dimensions += info.output_dimensions

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for info in self._info_list:
            parts.append(info.transform.transform(df[info.name]))
        result = pd.concat(parts, axis=1)
        result.index = df.index
        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        recovered = []
        names = []
        st = 0
        for info in self._info_list:
            dim = info.output_dimensions
            chunk = data.iloc[:, st:st + dim]
            recovered.append(info.transform.inverse_transform(chunk).values)
            names.append(info.name)
            st += dim

        out = pd.DataFrame(np.column_stack(recovered), columns=names, index=data.index)
        # Restore original dtypes where possible.
        # For integer columns, ROUND before casting — pandas/numpy `astype(int)`
        # truncates toward zero, so without rounding a GMM-generated 2.6 becomes
        # 2 (should be 3) and 17.7 becomes 17 (should be 18). Rounding preserves
        # the intended value and composes cleanly with the clip-to-bounds step
        # applied in train_survGAN_aws.py after generation.
        for col in out.columns:
            if col in self._column_raw_dtypes.index:
                tgt = self._column_raw_dtypes[col]
                try:
                    if np.issubdtype(tgt, np.integer):
                        out[col] = out[col].round().astype(tgt)
                    else:
                        out[col] = out[col].astype(tgt)
                except (ValueError, TypeError):
                    pass
        return out

    def layout(self) -> List[_FeatureInfo]:
        return self._info_list

    def n_features(self) -> int:
        return sum(info.output_dimensions for info in self._info_list)

    def activation_layout(
        self, discrete_activation: str, continuous_activation: str,
    ) -> List[Tuple[str, int]]:
        """Returns [(activation_name, width), ...] for the generator output."""
        out = []
        acts = {"discrete": discrete_activation, "continuous": continuous_activation}
        for info in self._info_list:
            ct = info.trans_feature_types[0]
            d = 0
            for t in info.trans_feature_types:
                if t != ct:
                    out.append((acts[ct], d))
                    ct = t
                    d = 0
                d += 1
            out.append((acts[ct], d))
        return out


# =========================================================================== #
#  SECTION 6b — BIN ENCODER (for survival conditional)                         #
# =========================================================================== #

class BinEncoder:
    """Simplified encoder: continuous → GMM bin index, discrete → passthrough.
    Used to create the conditioning vector for SurvivalGAN."""

    def __init__(self, n_components: int = 2, categorical_limit: int = 10):
        self.n_components = n_components
        self.categorical_limit = categorical_limit

    def fit(self, df: pd.DataFrame) -> "BinEncoder":
        disc_cols = _find_discrete_columns(df, self.categorical_limit)
        self._encoders: List[Tuple[str, str, Any]] = []  # (col, type, encoder)
        for col in df.columns:
            ftype = "discrete" if col in disc_cols else "continuous"
            if ftype == "continuous":
                enc = BayesianGMMFeatureEncoder(n_components=self.n_components).fit(df[col])
            else:
                enc = None  # passthrough
            self._encoders.append((col, ftype, enc))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for col, ftype, enc in self._encoders:
            if ftype == "discrete":
                parts.append(df[[col]].reset_index(drop=True))
            else:
                full = enc.transform(df[col])
                # Take argmax of component columns (drop the normalized value)
                comp_cols = full.iloc[:, 1:]
                bins = comp_cols.values.argmax(axis=1)
                parts.append(pd.DataFrame({col: bins}))
        return pd.concat(parts, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# =========================================================================== #
#  SECTION 7 — TABULAR GAN (wraps TabularEncoder + GAN)                        #
# =========================================================================== #

class TabularGAN(nn.Module):
    """GAN for tabular data: encodes → trains GAN → decodes.

    This is the bridge between raw DataFrames and the raw tensor-level GAN.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        n_units_latent: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        encoder_max_clusters: int = 5,
        dataloader_sampler: Optional[Sampler] = None,
        device: Any = "cpu",
        # Generator params
        generator_n_layers_hidden: int = 2,
        generator_n_units_hidden: int = 150,
        generator_nonlin: str = "leaky_relu",
        generator_nonlin_out_discrete: str = "softmax",
        generator_nonlin_out_continuous: str = "none",
        generator_n_iter: int = 2000,
        generator_batch_norm: bool = False,
        generator_dropout: float = 0.0,
        generator_lr: float = 1e-3,
        generator_weight_decay: float = 1e-3,
        generator_residual: bool = True,
        generator_opt_betas: tuple = (0.9, 0.999),
        generator_extra_penalties: Optional[list] = None,
        # Discriminator params
        discriminator_n_layers_hidden: int = 3,
        discriminator_n_units_hidden: int = 300,
        discriminator_nonlin: str = "leaky_relu",
        discriminator_n_iter: int = 1,
        discriminator_batch_norm: bool = False,
        discriminator_dropout: float = 0.1,
        discriminator_lr: float = 1e-3,
        discriminator_weight_decay: float = 1e-3,
        discriminator_opt_betas: tuple = (0.9, 0.999),
        # Training
        batch_size: int = 64,
        random_state: int = 0,
        clipping_value: int = 0,
        lambda_gradient_penalty: float = 10.0,
        lambda_identifiability_penalty: float = 0.1,
        n_iter_print: int = 50,
        n_iter_min: int = 100,
        patience: int = 10,
        **kwargs: Any,
    ):
        super().__init__()
        if generator_extra_penalties is None:
            generator_extra_penalties = []

        self.columns = X.columns
        self.device = device

        # ---- Tabular Encoder ----
        self.encoder = TabularEncoder(
            max_clusters=encoder_max_clusters,
            categorical_limit=10,
        ).fit(X)

        # ---- Conditional encoding ----
        self.cond_encoder: Optional[SklearnOneHotEncoder] = None
        n_units_conditional = 0
        self.predefined_conditional = cond is not None

        if cond is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)
            self.cond_encoder = SklearnOneHotEncoder(
                handle_unknown="ignore", sparse_output=False,
            ).fit(cond)
            n_units_conditional = self.cond_encoder.transform(cond).shape[-1]

        # ---- Conditional loss callback ----
        def _cond_loss(
            real_samples: torch.Tensor, fake_samples: torch.Tensor,
            cond: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if cond is None or self.predefined_conditional:
                return 0
            # When using ConditionalDatasetSampler (not in survival case)
            losses = []
            idx = 0
            cond_idx = 0
            for item in self.encoder.layout():
                length = item.output_dimensions
                if item.feature_type != "discrete":
                    idx += length
                    continue
                mask = cond[:, cond_idx:cond_idx + length].sum(dim=1).bool()
                if mask.sum() == 0:
                    idx += length
                    continue
                item_loss = nn.NLLLoss()(
                    torch.log(fake_samples[mask, idx:idx + length] + 1e-8),
                    torch.argmax(real_samples[mask, idx:idx + length], dim=1),
                )
                losses.append(item_loss)
                cond_idx += length
                idx += length
            if not losses:
                return 0
            return torch.stack(losses).sum() / len(real_samples)

        # ---- Build GAN ----
        self.model = GAN(
            n_features=self.encoder.n_features(),
            n_units_latent=n_units_latent,
            n_units_conditional=n_units_conditional,
            batch_size=batch_size,
            generator_n_layers_hidden=generator_n_layers_hidden,
            generator_n_units_hidden=generator_n_units_hidden,
            generator_nonlin=generator_nonlin,
            generator_nonlin_out=self.encoder.activation_layout(
                discrete_activation=generator_nonlin_out_discrete,
                continuous_activation=generator_nonlin_out_continuous,
            ),
            generator_n_iter=generator_n_iter,
            generator_batch_norm=generator_batch_norm,
            generator_dropout=generator_dropout,
            generator_lr=generator_lr,
            generator_residual=generator_residual,
            generator_weight_decay=generator_weight_decay,
            generator_opt_betas=generator_opt_betas,
            generator_extra_penalties=generator_extra_penalties,
            generator_extra_penalty_cbks=[_cond_loss],
            discriminator_n_layers_hidden=discriminator_n_layers_hidden,
            discriminator_n_units_hidden=discriminator_n_units_hidden,
            discriminator_n_iter=discriminator_n_iter,
            discriminator_nonlin=discriminator_nonlin,
            discriminator_batch_norm=discriminator_batch_norm,
            discriminator_dropout=discriminator_dropout,
            discriminator_lr=discriminator_lr,
            discriminator_weight_decay=discriminator_weight_decay,
            discriminator_opt_betas=discriminator_opt_betas,
            lambda_gradient_penalty=lambda_gradient_penalty,
            lambda_identifiability_penalty=lambda_identifiability_penalty,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            n_iter_min=n_iter_min,
            random_state=random_state,
            dataloader_sampler=dataloader_sampler,
            device=device,
            patience=patience,
            patience_metric=None,  # no metric-based early stopping
        )

    def fit(
        self,
        X: pd.DataFrame,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> "TabularGAN":
        X_enc = self.encoder.transform(X)

        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)
            cond = self.cond_encoder.transform(cond)

        self.model.fit(np.asarray(X_enc), np.asarray(cond) if cond is not None else None)
        return self

    def generate(
        self, count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)
            cond = self.cond_encoder.transform(cond)

        raw = self.model.generate(count, cond=cond)
        return self.encoder.inverse_transform(pd.DataFrame(raw))


# =========================================================================== #
#  SECTION 8 — IMBALANCED DATASET SAMPLER                                      #
# =========================================================================== #

class ImbalancedSampler(Sampler):
    """Oversamples underrepresented label groups.
    Compatible with PyTorch DataLoader's `sampler` argument."""

    def __init__(self, labels: list):
        self.labels = labels
        unique, counts = np.unique(labels, return_counts=True, axis=0)
        label_to_weight = {tuple(np.atleast_1d(u)): 1.0 / c for u, c in zip(unique, counts)}
        self.weights = np.array([
            label_to_weight[tuple(np.atleast_1d(l))] for l in labels
        ])
        self.weights /= self.weights.sum()
        self.n = len(labels)

    def __iter__(self):
        idxs = np.random.choice(self.n, size=self.n, replace=True, p=self.weights)
        return iter(idxs.tolist())

    def __len__(self):
        return self.n


# =========================================================================== #
#  SECTION 9 — SURVIVAL PIPELINE                                               #
#  Orchestrates: TTE model (synthcity) + GAN (decoupled) + censoring model     #
# =========================================================================== #

class SurvivalPipeline:
    """
    Full SurvivalGAN pipeline:
        1. Train a TTE model (synthcity) on real data
        2. Train a DeepHIT on covariates + T + E (with survival conditional)
        3. Generate synthetic data:
           a. GAN produces synthetic covariates + E (+ placeholder T)
           b. TTE model predicts realistic T from synthetic covariates + E
        4. XGBoost optionally predicts censoring from covariates
    """

    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg
        self.device = device

        # TTE model — now fully local (no synthcity time_to_event dependency)
        self.tte_model = None
        if cfg.uncensoring_model != "none":
            self.tte_model = LocalSurvivalFunctionTTE(device=device)

    def fit(self, df: pd.DataFrame) -> "SurvivalPipeline":
        cfg = self.cfg
        E = df[cfg.target_column]
        T = df[cfg.time_column]
        Xcov = df.drop(columns=[cfg.target_column, cfg.time_column])

        self.target_column = cfg.target_column
        self.time_column = cfg.time_column
        self.censoring_ratio = (E == 0).sum() / len(E)
        self.columns = df.columns.tolist()

        # ---- 1. Train TTE uncensoring model (synthcity) ----
        if self.tte_model is not None:
            log.info("Training TTE model...")
            self.tte_model.fit(Xcov, T, E)

        # ---- 2. Build sampling labels for imbalanced sampler ----
        sampler = None
        if cfg.sampling_strategy == "imbalanced_censoring":
            sampler = ImbalancedSampler(E.values.tolist())
        elif cfg.sampling_strategy == "imbalanced_time_censoring":
            # Bin T into 2 bins, combine with E
            t_bins = BinEncoder(n_components=2).fit_transform(T.to_frame()).values.squeeze().tolist()
            labels = list(zip(E.tolist(), t_bins))
            sampler = ImbalancedSampler(labels)

        # ---- 3. Build survival conditional ----
        train_conditional = None
        if cfg.use_survival_conditional:
            # Use all covariates + T + E for the conditional (binned)
            precond = pd.concat([T.to_frame(), E.to_frame(), Xcov], axis=1)
            train_conditional = BinEncoder(n_components=2).fit_transform(precond)
            log.info(f"Survival conditional shape: {train_conditional.shape}")

        # ---- 4. Prepare training data ----
        if cfg.tte_strategy == "uncensoring" and self.tte_model is not None:
            T_uncensored = pd.Series(self.tte_model.predict(Xcov), index=Xcov.index)
            T_uncensored[E == 1] = T[E == 1]
            df_train = Xcov.copy()
            df_train[cfg.time_column] = T_uncensored
            # No event column in training data for uncensoring strategy
        elif cfg.tte_strategy == "survival_function":
            df_train = df.copy()  # full dataframe with T and E
        else:
            raise ValueError(f"Unknown strategy: {cfg.tte_strategy}")

        # ---- 5. Train TabularGAN ----
        log.info(f"Training TabularGAN on {df_train.shape} ...")
        self.gan = TabularGAN(
            X=df_train,
            n_units_latent=cfg.generator_n_units_hidden,
            cond=train_conditional,
            encoder_max_clusters=cfg.encoder_max_clusters,
            dataloader_sampler=sampler,
            device=self.device,
            # generator
            generator_n_layers_hidden=cfg.generator_n_layers_hidden,
            generator_n_units_hidden=cfg.generator_n_units_hidden,
            generator_nonlin=cfg.generator_nonlin,
            generator_nonlin_out_discrete="softmax",
            generator_nonlin_out_continuous="none",
            generator_n_iter=cfg.n_iter,
            generator_batch_norm=cfg.generator_batch_norm,
            generator_dropout=cfg.generator_dropout,
            generator_lr=cfg.generator_lr,
            generator_weight_decay=cfg.generator_weight_decay,
            generator_residual=cfg.generator_residual,
            generator_opt_betas=cfg.generator_opt_betas,
            generator_extra_penalties=["identifiability_penalty"],
            # discriminator
            discriminator_n_layers_hidden=cfg.discriminator_n_layers_hidden,
            discriminator_n_units_hidden=cfg.discriminator_n_units_hidden,
            discriminator_nonlin=cfg.discriminator_nonlin,
            discriminator_n_iter=cfg.discriminator_n_iter,
            discriminator_batch_norm=cfg.discriminator_batch_norm,
            discriminator_dropout=cfg.discriminator_dropout,
            discriminator_lr=cfg.discriminator_lr,
            discriminator_weight_decay=cfg.discriminator_weight_decay,
            discriminator_opt_betas=cfg.discriminator_opt_betas,
            # training
            batch_size=cfg.batch_size,
            random_state=cfg.seed,
            clipping_value=cfg.clipping_value,
            lambda_gradient_penalty=cfg.lambda_gradient_penalty,
            lambda_identifiability_penalty=cfg.lambda_identifiability_penalty,
        )
        self.gan.fit(df_train, cond=train_conditional)
        self.train_conditional = train_conditional

        # ---- 6. Train censoring predictor (XGBoost) ----
        xgb_params = {"tree_method": "approx", "n_jobs": 2, "verbosity": 0,
                       "max_depth": 3, "random_state": 0}
        self.censoring_predictor = XGBClassifier(**xgb_params).fit(Xcov, E)

        return self

    def generate(self, count: int) -> pd.DataFrame:
        cfg = self.cfg

        # Resample the training conditional to match `count`
        gen_cond = None
        if self.train_conditional is not None:
            gen_cond = self.train_conditional.copy()
            while len(gen_cond) < count:
                gen_cond = pd.concat([gen_cond, gen_cond], ignore_index=True)
            gen_cond = gen_cond.head(count)

        generated = self.gan.generate(count, cond=gen_cond)

        # ---- Apply censoring strategy ----
        if cfg.censoring_strategy == "covariate_dependent":
            cov_cols = [c for c in generated.columns
                        if c not in (cfg.target_column, cfg.time_column)]
            generated[cfg.target_column] = self.censoring_predictor.predict(
                generated[cov_cols]
            )

        # ---- Apply TTE strategy ----
        if cfg.tte_strategy == "uncensoring":
            generated[cfg.target_column] = 1
        elif cfg.tte_strategy == "survival_function":
            if self.tte_model is not None:
                # Drop GAN-generated T; predict realistic T from TTE model
                generated = generated.drop(columns=[cfg.time_column])
                cov_cols = [c for c in generated.columns if c != cfg.target_column]
                generated[cfg.time_column] = self.tte_model.predict_any(
                    generated[cov_cols],
                    generated[cfg.target_column],
                )

        # Reorder columns to match original
        out_cols = [c for c in self.columns if c in generated.columns]
        return generated[out_cols]


# =========================================================================== #
#  SECTION 10 — MAIN ENTRY POINT                                               #
# =========================================================================== #

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device(preference: str) -> str:
    if preference == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = (preference == "cuda") and torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {name} ({mem:.1f} GB)")
        return "cuda"
    log.info("Using CPU")
    return "cpu"


def main():
    cfg = Config()
    if len(sys.argv) > 1:
        cfg.input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        cfg.output_csv = sys.argv[2]

    seed_everything(cfg.seed)
    device = setup_device(cfg.device)
    cfg.device = device

    # ---- Load data ----
    log.info(f"Loading {cfg.input_csv}")
    df = pd.read_csv(cfg.input_csv)
    log.info(f"Shape: {df.shape}  Columns: {df.columns.tolist()}")
    log.info(f"Event rate: {df[cfg.target_column].mean():.3f}")

    # ---- Train ----
    pipeline = SurvivalPipeline(cfg, device=device)
    t0 = time.time()
    pipeline.fit(df)
    log.info(f"Training done in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        log.info(f"Peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ---- Generate ----
    log.info(f"Generating {cfg.synthetic_count} synthetic samples...")
    synthetic_df = pipeline.generate(cfg.synthetic_count)
    log.info(f"Synthetic shape: {synthetic_df.shape}")

    # ---- Save ----
    synthetic_df.to_csv(cfg.output_csv, index=False)
    log.info(f"Saved to {cfg.output_csv}")

    # ---- Quick comparison ----
    log.info("=== Real vs Synthetic Summary ===")
    for col in df.columns:
        r = df[col].mean()
        s = synthetic_df[col].mean() if col in synthetic_df.columns else float("nan")
        log.info(f"  {col:20s}  real={r:8.3f}  synth={s:8.3f}")


if __name__ == "__main__":
    main()
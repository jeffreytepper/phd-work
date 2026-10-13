"""Braun et al. FORCE variant (Braun Methods eq. 7-10, 16-19; spec 6.7).

At update times z (every m integration steps), with R = r(z), the a-priori error e = z_input(z) - target(z) and

    P <- P - P R R^T P / (1 + R^T P R),     c = 1 / (1 + R^T P R)  [with the updated P],     J <- J - c e (P R)^T.

Switches (spec 6.7, all [VERIFY]): ``error_target`` ('methods': x*; 'supplement': y* = x* + tau x*_dot),
``c_uses_updated_P`` (True: as printed; False: standard Sussillo-Abbott J <- J - e (P_new R)^T),
``state_reset`` (reset h to h_target(snippet start) each iteration), ``target_time`` ('analytic': target evaluated at the
exact update time t0 + k dt; 'bin': at t0 + (k/m) * bin_size), ``weight_decay`` (J <- (1 - lambda_J) J after each update),
``mask`` (sparsity, Methods lines 1031-1074).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from scipy.special import expit

from ..models.rate_rnn import RNNParams, noise_scale
from ..targets import SequenceSpec, logit_target

__all__ = ["ForceConfig", "train_realization", "realization_rng", "pearson"]


@dataclass
class ForceConfig:
    """One training condition (Braun Methods 'FORCE-training of the RNNs'; defaults = synthetic setting)."""

    # target (Braun eq. 4-6)
    N: int = 501
    T: float = 8.5
    amp: float = 0.5
    width: float = 1.7
    offset: float = 0.0
    eps: float = 0.01
    # network (eq. 1-3)
    tau: float = 0.3224
    dt: float = 0.0081
    noise_gain: float = 0.5
    # training (eq. 7-9)
    g_init: float = 1.5
    p0: float = 0.08
    m: int = 16
    steps_per_iter: int = 2086
    bin_size: float = 0.129
    snippet_start_max: float = 119.2
    n_train: int = 300
    n_test: int = 10
    # [VERIFY] switches (spec 6.7)
    error_target: Literal["methods", "supplement"] = "methods"
    c_uses_updated_P: bool = True
    state_reset: bool = True
    target_time: Literal["analytic", "bin"] = "analytic"
    weight_decay: float = 0.0
    mask: dict | None = None  # {"kind": "presyn_random"|"presyn_block", "fraction": f}
    data: dict | None = None  # experimental target, see seqnet.data.registry (overrides N and the synthetic target)
    seed: int = 0

    def spec(self) -> SequenceSpec:
        return SequenceSpec(self.N, self.T, self.amp, self.width, self.offset, self.eps)

    def params(self) -> RNNParams:
        return RNNParams(self.tau, self.dt, self.noise_gain)

    def as_dict(self) -> dict:
        return asdict(self)


def realization_rng(seed: int, realization: int) -> np.random.Generator:
    """One Generator per realization, seeded from (base seed, realization index) (spec 5.3)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(realization)]))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of flattened arrays (spec 6.17)."""
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _mask_columns(cfg: ForceConfig, rng: np.random.Generator, N: int) -> np.ndarray | None:
    """Presynaptic set L whose outgoing weights are trained (Braun Methods lines 1033-1052)."""
    if not cfg.mask:
        return None
    kind, frac = cfg.mask["kind"], float(cfg.mask["fraction"])
    n_zero = int(round(frac * N))
    if kind == "presyn_random":
        zero = rng.choice(N, n_zero, replace=False)
    elif kind == "presyn_block":
        zero = np.arange(N - n_zero, N)  # the last fraction of neurons in the sequence ordering (Supp. 2E)
    else:
        raise NotImplementedError(f"mask kind {kind!r} (per-synapse masks with per-row P_i) not implemented yet")
    keep = np.ones(N, bool)
    keep[zero] = False
    return keep


def train_realization(cfg: ForceConfig, realization: int, keep_W0: bool = False) -> dict:
    """Train one network (one random J0, one random snippet start) for n_train + n_test iterations.

    Returns a dict with the final J, per-iteration Pearson correlation between the recorded rates at update times and
    the target rates, per-iteration mean absolute weight change (Braun eq. 10), the last test iteration's recorded
    rates and targets, and the snippet start.
    """
    t_start = time.perf_counter()
    rng = realization_rng(cfg.seed, realization)
    m, K = cfg.m, cfg.steps_per_iter
    prm = cfg.params()
    n_up = (K - 1) // m + 1
    if cfg.data:
        # experimental target (Braun Methods lines 913-925): rates at bin_size; update u uses target bin start + u
        from ..data.registry import load_target

        F, bin_size = load_target(cfg.data)
        N = F.shape[0]
        J = cfg.g_init / np.sqrt(N) * rng.standard_normal((N, N))
        sb = cfg.data.get("snippet_bins")
        start = int(rng.integers(0, F.shape[1] - sb + 1)) if sb else 0
        idx = np.clip(start + np.arange(n_up), 0, F.shape[1] - 1)
        t0 = start * bin_size
        f_z = F[:, idx]
        x_z = logit_target(f_z, cfg.eps)
        target = x_z if cfg.error_target == "methods" else x_z + cfg.tau * np.gradient(x_z, m * cfg.dt, axis=1)
        h_init = x_z[:, 0].copy()
    else:
        N = cfg.N
        spec = cfg.spec()
        J = cfg.g_init / np.sqrt(N) * rng.standard_normal((N, N))
        t0 = float(rng.uniform(0.0, cfg.snippet_start_max))
        ks = np.arange(n_up) * m
        tz = t0 + (ks * cfg.dt if cfg.target_time == "analytic" else np.arange(n_up) * cfg.bin_size)
        f_z = spec.rates(tz)  # (N, n_up) target rates
        target = spec.h_target(tz) if cfg.error_target == "methods" else spec.y_target(tz, cfg.tau)
        h_init = spec.h_target(np.array([t0]))[:, 0]
    keep = _mask_columns(cfg, rng, N)
    if keep is not None:
        J[:, ~keep] = 0.0
    J0 = J.copy() if keep_W0 else None
    nL = N if keep is None else int(keep.sum())
    P = cfg.p0 * np.eye(nL)
    ns = noise_scale(prm)
    a = cfg.dt / cfg.tau
    n_it = cfg.n_train + cfg.n_test
    corr = np.zeros(n_it)
    dW = np.zeros(n_it)
    h = h_init.copy()
    R_rec = np.zeros((N, n_up))
    for q in range(n_it):
        train = q < cfg.n_train
        if cfg.state_reset or q == 0:
            h = h_init.copy()
        J_prev = J.copy()
        u = 0
        for k in range(K):
            r = expit(h)
            z = J @ r
            z += ns * rng.standard_normal(N)
            if k % m == 0:
                R_rec[:, u] = r
                if train:
                    e = z - target[:, u]
                    rL = r if keep is None else r[keep]
                    Pr = P @ rL
                    denom = 1.0 + rL @ Pr
                    P -= np.outer(Pr, Pr) / denom
                    Pr_new = Pr / denom  # = P_new r
                    c = 1.0 / (1.0 + rL @ Pr_new) if cfg.c_uses_updated_P else 1.0
                    if keep is None:
                        J -= c * np.outer(e, Pr_new)
                    else:
                        J[:, keep] -= c * np.outer(e, Pr_new)
                    if cfg.weight_decay:
                        J *= 1.0 - cfg.weight_decay
                u += 1
            h += a * (-h + z)
        corr[q] = pearson(R_rec, f_z)
        dW[q] = float(np.mean(np.abs(J - J_prev)))
    out = {
        "J": J, "corr": corr, "dW": dW, "R_test": R_rec.copy(), "f_target": f_z, "t0": t0,
        "wall_time": time.perf_counter() - t_start,
    }
    if keep is not None:
        out["keep"] = keep
    if J0 is not None:
        out["J0"] = J0
    return out

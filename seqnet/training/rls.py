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

import numpy as np
from scipy.special import expit

from ..models.rate_rnn import noise_scale
from ..sequences import logit, training_target
from .config import ForceConfig

__all__ = ["ForceConfig", "train_realization", "realization_rng", "pearson"]


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
    elif kind == "synapse_random":
        M = rng.random((N, N)) >= frac  # True = synapse kept and trained (Braun lines 1053-1073)
        return M
    else:
        raise ValueError(f"unknown mask kind {kind!r}")
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
    seq = cfg.build_sequence()
    N = seq.N
    J = cfg.g_init / np.sqrt(N) * rng.standard_normal((N, N))
    offsets = np.arange(n_up) * m * cfg.dt if cfg.target_time == "analytic" else np.arange(n_up) * cfg.bin_size
    snip = seq.draw_snippet(rng, offsets, cfg.snippet_start_max)
    t0, f_z = snip.t0, snip.rates  # (N, n_up) target rates
    target = training_target(snip, cfg.error_target, cfg.tau, cfg.eps)
    h_init = logit(f_z[:, :1], cfg.eps)[:, 0]  # x*(t0)
    keep = _mask_columns(cfg, rng, N)
    rowmask = None
    if keep is not None and keep.ndim == 2:
        # per-synapse mask with one P_i per postsynaptic row (Braun eq. 18-19); rows padded to a common length with a
        # sentinel presynaptic index N whose rate is 0, which leaves the padded P entries untouched (exact).
        rowmask, keep = keep, None
        J[~rowmask] = 0.0
        L = int(rowmask.sum(1).max())
        mem = N * L * L * 8
        if mem > float(cfg.mask.get("max_bytes", 6e8)):
            raise MemoryError(f"per-row P_i needs {mem / 1e9:.1f} GB at fraction {cfg.mask['fraction']}; raise mask.max_bytes")
        idx = np.full((N, L), N)
        for i in range(N):
            nz = np.nonzero(rowmask[i])[0]
            idx[i, : nz.size] = nz
        P_rows = np.broadcast_to(cfg.p0 * np.eye(L), (N, L, L)).copy()
        J_ext = np.concatenate([J, np.zeros((N, 1))], axis=1)
    elif keep is not None:
        J[:, ~keep] = 0.0
    J0 = J.copy() if keep_W0 else None
    nL = N if keep is None else int(keep.sum())
    P = cfg.p0 * np.eye(nL) if rowmask is None else None
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
                if train and rowmask is not None:
                    e = z - target[:, u]
                    rp = np.append(r, 0.0)[idx]  # (N, L) presynaptic rates per row
                    Pr = np.einsum("nij,nj->ni", P_rows, rp)
                    denom = 1.0 + np.einsum("ni,ni->n", rp, Pr)
                    P_rows -= np.einsum("ni,nj->nij", Pr, Pr) / denom[:, None, None]
                    Prn = Pr / denom[:, None]
                    c = 1.0 / (1.0 + np.einsum("ni,ni->n", rp, Prn)) if cfg.c_uses_updated_P else 1.0
                    J_ext[np.arange(N)[:, None], idx] -= (c * e)[:, None] * Prn
                    J_ext[:, N] = 0.0
                    J = J_ext[:, :N]
                    if cfg.weight_decay:
                        J_ext *= 1.0 - cfg.weight_decay
                elif train:
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
    if rowmask is not None:
        out["keep"] = rowmask
    if J0 is not None:
        out["J0"] = J0
    return out

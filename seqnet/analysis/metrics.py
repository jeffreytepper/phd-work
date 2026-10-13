"""Metrics (spec 6.17)."""
from __future__ import annotations

import numpy as np

from ..conventions import circulant_projection, eigs_from_profile
from ..training.rls import pearson

__all__ = ["pearson", "residual_fraction", "eig_errors", "subspace_residual", "active_projector"]


def residual_fraction(W: np.ndarray) -> float:
    """|W - W_cp|_F / |W|_F."""
    return float(np.linalg.norm(W - circulant_projection(W)) / np.linalg.norm(W))


def eig_errors(cp: np.ndarray, eig_pred: dict[int, complex], k_max: int = 10) -> dict:
    """Relative magnitude and phase errors of learned eigenvalues lambda~_k (from the profile) against predictions."""
    lam = eigs_from_profile(cp)
    N = cp.size
    ks = [k for k in range(-k_max, k_max + 1) if k in eig_pred]
    learned = np.array([lam[k % N] for k in ks])
    pred = np.array([eig_pred[k] for k in ks])
    return {"k": np.array(ks), "learned": learned, "predicted": pred,
            "rel_mag_err": np.abs(np.abs(learned) - np.abs(pred)) / np.abs(pred),
            "rel_err": np.abs(learned - pred) / np.abs(pred),
            "phase_err": np.angle(learned / pred)}


def active_projector(N: int, D: int) -> np.ndarray:
    """P_D = (1/N) sum_{|k|<=D} v_k v_k^* (real), the projector onto the active subspace (Braun Suppl. Prop. 4.1)."""
    n = np.arange(N)
    P = np.zeros((N, N))
    for k in range(-D, D + 1):
        v = np.exp(-2j * np.pi * k * n / N)
        P += np.real(np.outer(v, v.conj()))
    return P / N


def subspace_residual(W: np.ndarray, D: int) -> float:
    """|(W - W_cp) P_D|_F: the non-circulant part acting on the active subspace (~0 for a supporting matrix)."""
    N = W.shape[0]
    return float(np.linalg.norm((W - circulant_projection(W)) @ active_projector(N, D)))


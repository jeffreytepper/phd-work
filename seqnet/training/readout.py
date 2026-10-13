"""Readout learning in the main package (spec 6.8).

``braun_downstream`` is Braun's downstream rule (Methods lines 1264-1275).  The delta rule and the ridge closed form
are thin wrappers over :mod:`seqnet.thesis` (spec 6A.2 rule 6).
"""
from __future__ import annotations

import numpy as np

from ..thesis import DeltaRule, RLSReadout, ridge_readout

__all__ = ["braun_downstream", "ridge_readout_closed_form", "DeltaRule", "RLSReadout"]


def braun_downstream(inputs: np.ndarray, targets: np.ndarray, alpha: float = 0.015, n_iter: int = 1000,
                     A0: np.ndarray | None = None, record_every: int = 0) -> dict:
    """A initialized to all ones; each iteration A <- A + alpha * mean_t(error_i(t) input_j(t)), error = target - A input.

    inputs: (N_u, n_t) upstream rates; targets: (N_d, n_t).  Returns the final A, outputs and the correlation history.
    (Braun eq. 23 and the downstream update; one update per iteration.)
    """
    N_d, n_t = targets.shape
    A = np.ones((N_d, inputs.shape[0])) if A0 is None else np.array(A0, dtype=float)
    hist = []
    for q in range(n_iter):
        out = A @ inputs
        err = targets - out
        A += alpha * (err @ inputs.T) / n_t
        if record_every and q % record_every == 0:
            hist.append(float(np.corrcoef(out.ravel(), targets.ravel())[0, 1]))
    out = A @ inputs
    return {"A": A, "output": out, "corr": float(np.corrcoef(out.ravel(), targets.ravel())[0, 1]), "history": np.array(hist)}


def ridge_readout_closed_form(X: np.ndarray, y: np.ndarray, decay: float) -> np.ndarray:
    """(X^T X + decay I)^{-1} X^T y (Thesis eq. 3.13), via seqnet.thesis."""
    return ridge_readout(X, y, decay)

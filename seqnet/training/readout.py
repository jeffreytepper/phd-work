"""Readout learning in the main package (spec 6.8).

``braun_downstream`` is Braun's downstream rule (Methods lines 1264-1275).  The delta rule, RLS and the ridge
closed form are copied from ``msc_thesis.readout`` (the standalone thesis package, formerly ``seqnet.thesis``;
spec 6A.2 rule 6), so seqnet has no dependency on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
    """(X^T X + decay I)^{-1} X^T y (Thesis eq. 3.13)."""
    return ridge_readout(X, y, decay)


# --- copied from msc_thesis.readout (Thesis §3) -------------------------------------------------------------------
# Regularizer bookkeeping: the delta rule with decay (eq. 3.18), applied to positions drawn uniformly over T
# observations, has fixed point ridge with lambda = T * lambda_d.  DeltaRule therefore takes the *ridge* lambda and
# uses lambda / T internally unless ``decay_is_ridge=False``.

def ridge_readout(X: np.ndarray, y: np.ndarray, decay: float) -> np.ndarray:
    """Ridge solution w*_lambda = (X^T X + lambda I)^{-1} X^T y (Thesis eq. 3.13)."""
    N = X.shape[1]
    return np.linalg.solve(X.T @ X + decay * np.eye(N), X.T @ y)


@dataclass
class DeltaRule:
    """Delta rule with weight decay, w <- w + eta (e x - lambda_d w), e = y - x^T w (Thesis eq. 3.17-3.18).

    Parameters
    ----------
    eta : learning rate.
    decay : regularizer.  If ``decay_is_ridge`` (default) this is the ridge lambda of eq. 3.13 and the
        update uses lambda_d = lambda / T_obs so that the fixed point is the eq. 3.13 ridge solution.
        Otherwise it is lambda_d itself (eq. 3.18 literally).
    """

    N: int
    eta: float
    decay: float
    T_obs: int
    decay_is_ridge: bool = True
    w: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.zeros(self.N)

    @property
    def decay_d(self) -> float:
        return self.decay / self.T_obs if self.decay_is_ridge else self.decay

    @property
    def equivalent_ridge(self) -> float:
        """Ridge regularizer of the rule's fixed point: T_obs * lambda_d."""
        return self.decay_d * self.T_obs

    def step(self, x: np.ndarray, y: float) -> np.ndarray:
        e = y - x @ self.w
        self.w = self.w + self.eta * (e * x - self.decay_d * self.w)
        return self.w

    def epoch(self, X: np.ndarray, y: np.ndarray, order: np.ndarray | None = None) -> np.ndarray:
        """One pass over the T observations (in ``order`` if given)."""
        idx = range(X.shape[0]) if order is None else order
        for t in idx:
            self.step(X[t], y[t])
        return self.w

    def batch_epoch(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Expected (averaged) update over one epoch: w <- w + eta (X^T (y - Xw)/T - lambda_d w).

        This is the mean-field version of T stochastic steps divided by T; it has the same fixed point
        and per-mode rates (eq. 3.19) and is used when the inner loop must be cheap.
        """
        T = X.shape[0]
        e = y - X @ self.w
        self.w = self.w + self.eta * (X.T @ e / T - self.decay_d * self.w)
        return self.w


@dataclass
class RLSReadout:
    """Recursive least squares on a linear readout (spec 6.7 / 6.8 with a scalar error).

    P(0) = p0 I starting from w = 0 makes RLS minimize sum_n e_n^2 + (1/p0) |w|^2 over all
    presented samples (no forgetting), i.e. ridge with lambda = 1/p0 on the pooled data.
    """

    N: int
    p0: float
    w: np.ndarray = field(default=None)  # type: ignore[assignment]
    P: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.zeros(self.N)
        if self.P is None:
            self.P = self.p0 * np.eye(self.N)

    def step(self, x: np.ndarray, y: float) -> np.ndarray:
        Px = self.P @ x
        k = Px / (1.0 + x @ Px)
        e = y - x @ self.w
        self.w = self.w + k * e
        self.P = self.P - np.outer(k, Px)
        return self.w

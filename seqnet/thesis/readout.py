"""Readout learning: pseudoinverse, ridge, per-mode ridge, spectral filter, delta rule, RLS (Thesis §3-4).

Regularizer bookkeeping (a discrepancy recorded in results/thesis/REPORT.md)
--------------------------------------------------------------------------
The ridge loss (eq. 3.12) is a *sum* over T observations, |Xw - y|^2 + lambda |w|^2, with minimizer
(X^T X + lambda I)^{-1} X^T y (eq. 3.13).  The delta rule with decay (eq. 3.18),
w <- w + eta (e x - lambda_d w), applied to positions drawn uniformly over the T observations, has
expected fixed point (X^T X / T + lambda_d I) w = X^T y / T, i.e. ridge with lambda = T * lambda_d.
The thesis's own per-mode rate (eq. 3.19, eq. 4.14), eta (sigma^2 / T + lambda_d), is consistent with
this.  :class:`DeltaRule` therefore takes the *ridge* regularizer and uses decay lambda / T
internally unless ``decay_is_ridge=False`` is passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "pinv_readout",
    "ridge_readout",
    "per_mode_ridge",
    "spectral_filter",
    "delta_rule_rates",
    "DeltaRule",
    "RLSReadout",
]


def pinv_readout(X: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Minimum-norm least squares w* = X^+ y (Thesis eq. 3.11)."""
    return np.linalg.pinv(X, rcond=rcond) @ y


def ridge_readout(X: np.ndarray, y: np.ndarray, decay: float) -> np.ndarray:
    """Ridge solution w*_lambda = (X^T X + lambda I)^{-1} X^T y (Thesis eq. 3.13)."""
    N = X.shape[1]
    return np.linalg.solve(X.T @ X + decay * np.eye(N), X.T @ y)


def per_mode_ridge(K: np.ndarray, y_hat: np.ndarray, decay: float) -> np.ndarray:
    """Per-mode ridge w_hat*_m = K_m y_hat_m / (|K_m|^2 + lambda) (Thesis eq. 4.5, D10).

    ``K`` is the unnormalized kernel DFT (circulant eigenvalue), ``y_hat`` the unitary DFT of the
    target sampled at T_obs = N positions; returns the unitary DFT of the ridge weights.
    With X_{t,i} = k(r_i - r_t), X w is the circular cross-correlation of k and w, so
    (Xw)^_m = conj(K_m) w_hat_m and the ridge solution carries K_m (not its conjugate) in the
    numerator.  For a symmetric kernel K_m is real and the distinction is moot (Thesis A6).
    """
    return K * y_hat / (np.abs(K) ** 2 + decay)


def spectral_filter(a: np.ndarray, decay: float) -> np.ndarray:
    """g_m = a_m / (a_m + lambda) (Thesis eq. 4.6 / 3.15)."""
    a = np.asarray(a, dtype=float)
    return a / (a + decay)


def delta_rule_rates(sigma2: np.ndarray, eta: float, decay_d: float, T_obs: int) -> np.ndarray:
    """Expected per-mode convergence rate tau^{-1} = eta (sigma^2 / T + lambda_d) (Thesis eq. 3.19, 4.14).

    ``decay_d`` is the delta-rule decay (not the ridge lambda; see module docstring).  For updates
    drawn uniformly over the T observations, E[w_{n+1} - w*] = (I - eta (X^T X/T + lambda_d)) E[w_n - w*],
    so each mode contracts by the factor (1 - tau^{-1}) per step.
    """
    return eta * (np.asarray(sigma2) / T_obs + decay_d)


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

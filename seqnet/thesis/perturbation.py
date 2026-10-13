"""Position-dependent kernels and the coupling error (Thesis App. E; aggregate energy A_m, eq. E.11 / M.1-M.2)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .kernels import Kernel
from .ring import Ring, Schedule

__all__ = [
    "position_dependent_design",
    "amplitude_modulated",
    "position_averaged_kernel",
    "coupling_matrix",
    "bridge_energy",
    "expected_ridge_error",
    "coupling_error",
]

KernelFn = Callable[[np.ndarray, float], np.ndarray]  # k(u; theta) for displacements u at bump position theta


def position_dependent_design(kfn: KernelFn, N: int, schedule: Schedule) -> np.ndarray:
    """X_{t,i} = k(r_i - c_t; c_t) (Thesis App. E.1, eq. E.8)."""
    r = Ring(N).positions
    return np.stack([kfn(r - c, c) for c in schedule.positions])


def amplitude_modulated(kernel: Kernel, h: Callable[[np.ndarray], np.ndarray], eps: float) -> KernelFn:
    """k(u; theta) = k(u) (1 + eps h(theta)) (spec WP5.3 perturbation family)."""

    def kfn(u: np.ndarray, theta: float) -> np.ndarray:
        return kernel(u) * (1.0 + eps * float(h(np.asarray(theta))))

    return kfn


def position_averaged_kernel(kfn: KernelFn, N: int, schedule: Schedule) -> Kernel:
    """k_bar(r) = (1/T) sum_t k(r; theta_t) evaluated relative to each bump centre (Thesis eq. E.1)."""
    r = Ring(N).positions
    vals = np.mean([kfn(r, c) for c in schedule.positions], axis=0)
    return Kernel(values=vals, name="position_averaged")


def coupling_matrix(X_bar: np.ndarray, dX: np.ndarray) -> np.ndarray:
    """C = F [X_bar^T dX + dX^T X_bar] F* with F the unitary DFT matrix (Thesis eq. E.6)."""
    N = X_bar.shape[1]
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    M = X_bar.T @ dX + dX.T @ X_bar
    return F @ M @ F.conj().T


def bridge_energy(X: np.ndarray) -> np.ndarray:
    """A_m = (F X^T X F*)_{mm}, per-mode observation energy (Thesis eq. E.11; aggregate energy of App. M)."""
    N = X.shape[1]
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    return np.real(np.diag(F @ (X.T @ X) @ F.conj().T))


def expected_ridge_error(X: np.ndarray, S_full: np.ndarray, decay: float) -> float:
    """E_y |X w*(y) - y|^2 for ridge weights and a stationary target with circulant covariance.

    ``S_full`` is S on all T FFT bins of the observation grid (unitary normalization), so that
    Cov(y) = F* diag(S) F.  Exact expectation, no sampling: tr((I - H)^T (I - H) Cov(y)).
    """
    T, N = X.shape
    H = X @ np.linalg.solve(X.T @ X + decay * np.eye(N), X.T)
    F = np.fft.fft(np.eye(T)) / np.sqrt(T)
    C = np.real(F.conj().T @ np.diag(S_full) @ F)
    R = np.eye(T) - H
    return float(np.trace(R.T @ R @ C))


def coupling_error(
    kernel: Kernel,
    S_half: np.ndarray,
    decay: float,
    eps_values: np.ndarray,
    h: Callable[[np.ndarray], np.ndarray] | None = None,
    schedule: Schedule | None = None,
) -> dict:
    """Excess expected error of the position-dependent kernel over the translation-invariant baseline.

    Thesis App. E.4 predicts E_coupling = O(eps^2) (first order cancels under stationarity, eq. E.12-E.13).
    Returns eps, excess error, and the fitted log-log exponent.  Default h(theta) = cos(theta) (zero mean,
    smooth, so the position-averaged kernel equals the baseline).
    """
    N = kernel.N
    schedule = Schedule(N, 1) if schedule is None else schedule
    h = (lambda th: np.cos(th)) if h is None else h
    T = schedule.T_obs
    m = np.abs(np.rint(np.fft.fftfreq(T, 1.0 / T)).astype(int))
    S_full = np.asarray(S_half)[np.clip(m, 0, len(S_half) - 1)]
    base = expected_ridge_error(position_dependent_design(lambda u, c: kernel(u), N, schedule), S_full, decay)
    excess = []
    for eps in eps_values:
        X = position_dependent_design(amplitude_modulated(kernel, h, float(eps)), N, schedule)
        excess.append(expected_ridge_error(X, S_full, decay) - base)
    excess = np.asarray(excess)
    eps_values = np.asarray(eps_values, dtype=float)
    pos = (eps_values > 0) & (np.abs(excess) > 0)
    slope = float(np.polyfit(np.log(eps_values[pos]), np.log(np.abs(excess[pos])), 1)[0]) if pos.sum() >= 2 else np.nan
    return {"eps": eps_values, "excess": excess, "base": base, "exponent": slope}

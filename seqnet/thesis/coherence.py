"""Coherence, independence distance and the uncertainty principle (Thesis §5, App. C, App. D)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .design import activity
from .fourier import signed_modes
from .kernels import Kernel
from .ring import Ring

__all__ = [
    "autocorrelation",
    "coherence",
    "pairwise_coherence",
    "spectral_energy_distribution",
    "spectral_spread",
    "spatial_width",
    "uncertainty_product",
    "independence_distance",
]


def _signed_power(kernel: Kernel) -> tuple[np.ndarray, np.ndarray]:
    """(signed modes, |K_m|^2) with the even-N Nyquist bin split evenly between +-N/2."""
    N = kernel.N
    m = signed_modes(N).astype(float)
    P = np.abs(kernel.spectrum()) ** 2
    if N % 2 == 0:
        nyq = N // 2
        m = np.append(m, float(nyq))
        P = np.append(P, P[nyq] / 2)
        P[nyq] /= 2
    return m, P


def autocorrelation(kernel: Kernel, delta: np.ndarray) -> np.ndarray:
    """R_k(Delta) = sum_m |k_hat_m|^2 e^{i m Delta} (Thesis eq. 5.5), at arbitrary displacements.

    Equals sum_i k(r_i) k(r_i - Delta) (eq. 5.1) on the grid, and the inner product of the kernel's
    trigonometric interpolants off the grid.
    """
    m, P = _signed_power(kernel)
    delta = np.asarray(delta, dtype=float)
    return np.real(np.exp(1j * np.multiply.outer(delta, m)) @ P) / kernel.N


def coherence(kernel: Kernel, delta: np.ndarray) -> np.ndarray:
    """mu(Delta) = sum_m p_m e^{i m Delta} (Thesis eq. 5.6), the spectral form."""
    return autocorrelation(kernel, delta) / autocorrelation(kernel, np.array(0.0))


def pairwise_coherence(kernel: Kernel, delta: np.ndarray, c: float = 0.0) -> np.ndarray:
    """mu(Delta) = x(c)^T x(c + Delta) / |x(c)|^2 from activity vectors (Thesis eq. 5.1-5.2)."""
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    x0 = activity(kernel, c)
    xs = activity(kernel, c + delta)
    return xs @ x0 / (x0 @ x0)


def spectral_energy_distribution(kernel: Kernel) -> tuple[np.ndarray, np.ndarray]:
    """(m, p_m) with p_m = |k_hat_m|^2 / sum |k_hat|^2 over signed modes (Thesis eq. 5.6)."""
    m, P = _signed_power(kernel)
    return m, P / P.sum()


def spectral_spread(kernel: Kernel) -> float:
    """sigma_xi = sqrt(sum m^2 p_m - (sum m p_m)^2) (Thesis eq. 5.8)."""
    m, p = spectral_energy_distribution(kernel)
    return float(np.sqrt(np.sum(m**2 * p) - np.sum(m * p) ** 2))


def spatial_width(kernel: Kernel) -> float:
    """sigma_x = sqrt(sum r_i^2 k(r_i)^2 / sum k(r_i)^2) with r_i in [-pi, pi) (Thesis eq. 5.10).

    Note: under this definition a Gaussian k ~ exp(-r^2 / 2 sigma^2) has sigma_x = sigma / sqrt(2)
    (the thesis's text takes sigma_x = sigma; see results/thesis/REPORT.md).
    """
    r = Ring(kernel.N).centred_positions
    k2 = kernel.values**2
    return float(np.sqrt(np.sum(r**2 * k2) / np.sum(k2)))


def uncertainty_product(kernel: Kernel) -> float:
    """sigma_x * sigma_xi (Theorem 5.1 claims >= 1/2 with equality iff Gaussian, eq. 5.11)."""
    return spatial_width(kernel) * spectral_spread(kernel)


def independence_distance(kernel: Kernel, eps: float = 0.1, n_grid: int = 20000) -> float:
    """d*(eps) = min{Delta > 0 : |mu(Delta)| <= eps} (Thesis eq. 5.7), refined by root finding.

    Returns NaN when |mu| never drops to eps on (0, pi].
    """
    d = np.linspace(0, np.pi, n_grid + 1)[1:]
    mu = np.abs(coherence(kernel, d))
    idx = np.nonzero(mu <= eps)[0]
    if idx.size == 0:
        return float("nan")
    j = idx[0]
    if j == 0:
        return float(d[0])
    f = lambda x: abs(float(coherence(kernel, np.array([x]))[0])) - eps  # noqa: E731
    try:
        return float(brentq(f, d[j - 1], d[j]))
    except ValueError:
        return float(d[j])

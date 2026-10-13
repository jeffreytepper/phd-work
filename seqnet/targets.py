"""Target sequences (Braun Methods eq. 4-6; Braun Suppl. eq. 1-3)."""
from __future__ import annotations

import numpy as np

__all__ = [
    "wrapped_gaussian",
    "wrapped_gaussian_dt",
    "sequence_rates",
    "sequence_rates_dt",
    "logit_target",
    "logit_target_dt",
    "fourier_coeffs_gaussian",
    "SequenceSpec",
]


def wrapped_gaussian(t: np.ndarray, width: float, T: float, n_images: int = 6) -> np.ndarray:
    """g_{sigma,T}(t) = sum_{|m| <= n_images} exp(-(t - m T)^2 / (2 sigma^2)) (Braun Methods eq. 5).

    The argument is first reduced to [-T/2, T/2) so that the image sum is centred (exact up to truncation; T5).
    """
    t = np.asarray(t, dtype=float)
    u = (t + T / 2) % T - T / 2
    m = np.arange(-n_images, n_images + 1)
    return np.exp(-((u[..., None] - m * T) ** 2) / (2 * width**2)).sum(-1)


def wrapped_gaussian_dt(t: np.ndarray, width: float, T: float, n_images: int = 6) -> np.ndarray:
    """Time derivative of :func:`wrapped_gaussian`."""
    t = np.asarray(t, dtype=float)
    u = (t + T / 2) % T - T / 2
    m = np.arange(-n_images, n_images + 1)
    x = u[..., None] - m * T
    return (-(x / width**2) * np.exp(-(x**2) / (2 * width**2))).sum(-1)


def sequence_rates(N: int, T: float, amp: float, width: float, offset: float, t: np.ndarray,
                   n_images: int = 6) -> np.ndarray:
    """f_n(t) = a g(t - mu_n) + b with mu_n = n T / N, n = 0..N-1; shape (N, n_t) (Braun Methods eq. 4).

    Braun indexes neurons i = 1..N with mu_i = i T/N; starting at n = 0 only relabels the time origin.
    """
    mu = np.arange(N) * T / N
    return amp * wrapped_gaussian(np.asarray(t)[None, :] - mu[:, None], width, T, n_images) + offset


def sequence_rates_dt(N: int, T: float, amp: float, width: float, t: np.ndarray, n_images: int = 6) -> np.ndarray:
    """d f_n / dt (analytic)."""
    mu = np.arange(N) * T / N
    return amp * wrapped_gaussian_dt(np.asarray(t)[None, :] - mu[:, None], width, T, n_images)


def logit_target(f: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """x* = phi^{-1}(f) = ln((f + eps) / (1 - f + eps)) (Braun Methods eq. 6)."""
    f = np.asarray(f, dtype=float)
    return np.log((f + eps) / (1.0 - f + eps))


def logit_target_dt(f: np.ndarray, fdot: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """d x*/dt = f_dot [1/(f + eps) + 1/(1 - f + eps)] (spec 6.2), used for y* = x* + tau x*_dot."""
    f = np.asarray(f, dtype=float)
    return np.asarray(fdot) * (1.0 / (f + eps) + 1.0 / (1.0 - f + eps))


def fourier_coeffs_gaussian(k: np.ndarray, amp: float, width: float, offset: float, T: float) -> np.ndarray:
    """Temporal Fourier coefficients of a g(t) + b: a sigma sqrt(2 pi)/T exp(-sigma^2 (k omega)^2 / 2) (+ b at k = 0).

    Braun Suppl. eq. 2-3 (infinite image sum; the wrapped Gaussian's coefficients are exactly these).
    """
    k = np.asarray(k, dtype=float)
    w = 2 * np.pi / T
    c = amp * width * np.sqrt(2 * np.pi) / T * np.exp(-(width**2) * (k * w) ** 2 / 2)
    return c + np.where(k == 0, offset, 0.0)


class SequenceSpec:
    """Synthetic Braun target: N neurons, period T, amplitude a, width sigma (s), offset b, logit eps."""

    def __init__(self, N: int = 501, T: float = 8.5, amp: float = 0.5, width: float = 1.7, offset: float = 0.0,
                 eps: float = 0.01, n_images: int = 6):
        self.N, self.T, self.amp, self.width, self.offset, self.eps, self.n_images = N, T, amp, width, offset, eps, n_images

    def rates(self, t: np.ndarray) -> np.ndarray:
        return sequence_rates(self.N, self.T, self.amp, self.width, self.offset, t, self.n_images)

    def rates_dt(self, t: np.ndarray) -> np.ndarray:
        return sequence_rates_dt(self.N, self.T, self.amp, self.width, t, self.n_images)

    def h_target(self, t: np.ndarray) -> np.ndarray:
        """x*(t) (Braun eq. 6)."""
        return logit_target(self.rates(t), self.eps)

    def y_target(self, t: np.ndarray, tau: float) -> np.ndarray:
        """y* = x* + tau dx*/dt, the input required by the support condition (Braun Suppl. Def. 1.1)."""
        f = self.rates(t)
        return logit_target(f, self.eps) + tau * logit_target_dt(f, self.rates_dt(t), self.eps)

    def as_dict(self) -> dict:
        return {"N": self.N, "T": self.T, "amp": self.amp, "width": self.width, "offset": self.offset, "eps": self.eps}

"""Preprocessing shared by the bat and mouse pipelines (Braun Methods lines 841-905, 1075-1127; spec 6.18).

MATLAB semantics are emulated where the Methods name a MATLAB function:
* ``movmean(x, k)``: centred moving mean over k samples; for even k the window is [i - k/2, i + k/2 - 1]; at the
  edges the window shrinks to the available samples (MATLAB default 'shrink').
* ``fit`` of a Gaussian: least squares on a * exp(-(t - mu)^2 / (2 sigma^2)) after subtracting the offset b, which is
  taken as the minimum of the bump (Braun eq. 20 and lines 1114-1117).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["movmean", "resample_bins", "order_by_peak", "normalize_max", "centre_sequence", "rescale_duration",
           "BumpFit", "fit_bump", "activity_bump"]


def movmean(x: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """MATLAB ``movmean(x, k)`` along ``axis`` (centred window, shrinking at the ends)."""
    x = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    n = x.shape[-1]
    lo = k // 2 if k % 2 else k // 2
    hi = k // 2 if k % 2 else k // 2 - 1
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    i = np.arange(n)
    a = np.clip(i - lo, 0, n)
    b = np.clip(i + hi + 1, 0, n)
    out = (c[..., b] - c[..., a]) / (b - a)
    return np.moveaxis(out, -1, axis)


def resample_bins(x: np.ndarray, dt_in: float, dt_out: float, duration: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Resample (..., n_t) at bin size dt_in to bin size dt_out by linear interpolation of bin centres.

    Returns (resampled, new bin centres).  ``duration`` defaults to n_t * dt_in.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    duration = n * dt_in if duration is None else duration
    t_in = (np.arange(n) + 0.5) * dt_in
    n_out = int(np.floor(duration / dt_out + 1e-9))
    t_out = (np.arange(n_out) + 0.5) * dt_out
    flat = x.reshape(-1, n)
    out = np.stack([np.interp(t_out, t_in, row) for row in flat]).reshape(x.shape[:-1] + (n_out,))
    return out, t_out


def order_by_peak(x: np.ndarray) -> np.ndarray:
    """Indices sorting neurons by the time of their maximum (ascending) (Braun lines 856-859, 895-902)."""
    return np.argsort(np.argmax(x, axis=-1), kind="stable")


def normalize_max(x: np.ndarray) -> np.ndarray:
    """Divide each neuron's activity by its maximum (Braun lines 863, 882)."""
    m = x.max(axis=-1, keepdims=True)
    return x / np.where(m > 0, m, 1.0)


def centre_sequence(x: np.ndarray, centre_bin: int) -> np.ndarray:
    """Circularly shift each neuron so that its maximum sits at ``centre_bin`` (Braun lines 1077-1088)."""
    out = np.empty_like(x)
    for i, row in enumerate(x):
        out[i] = np.roll(row, centre_bin - int(np.argmax(row)))
    return out


def rescale_duration(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly interpolate each row to ``n_out`` samples spanning the same interval (Braun lines 1096-1097, 1108)."""
    n = x.shape[-1]
    s_in = np.linspace(0, 1, n)
    s_out = np.linspace(0, 1, n_out)
    return np.stack([np.interp(s_out, s_in, row) for row in x])


@dataclass(frozen=True)
class BumpFit:
    """Gaussian bump g(t) = a exp(-(t - mu)^2 / (2 sigma^2)) + b (Braun eq. 20)."""

    amp: float
    width: float
    offset: float
    mu: float
    r2: float
    t: np.ndarray
    bump: np.ndarray

    def model(self, t: np.ndarray) -> np.ndarray:
        return self.amp * np.exp(-((t - self.mu) ** 2) / (2 * self.width**2)) + self.offset


def fit_bump(t: np.ndarray, bump: np.ndarray) -> BumpFit:
    """b = min(bump); fit a, mu, sigma to bump - b by least squares (Braun lines 1114-1117)."""
    t = np.asarray(t, float)
    bump = np.asarray(bump, float)
    b = float(bump.min())
    y = bump - b
    p0 = (float(y.max()), float(t[np.argmax(y)]), float((t[-1] - t[0]) / 6))
    (a, mu, s), _ = curve_fit(lambda tt, a, mu, s: a * np.exp(-((tt - mu) ** 2) / (2 * s**2)), t, y, p0=p0, maxfev=20000)
    pred = a * np.exp(-((t - mu) ** 2) / (2 * s**2)) + b
    r2 = 1 - np.sum((bump - pred) ** 2) / np.sum((bump - bump.mean()) ** 2)
    return BumpFit(amp=float(a), width=float(abs(s)), offset=b, mu=float(mu), r2=float(r2), t=t, bump=bump)


def activity_bump(seq: np.ndarray, centre_bin: int, T_target: float = 8.5, n_out: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred sequence -> rescaled to T_target seconds -> mean across neurons (Braun lines 1077-1113).

    Returns (centred sequence, time axis of the rescaled bump in seconds, bump).
    """
    c = centre_sequence(seq, centre_bin)
    n_out = c.shape[-1] if n_out is None else n_out
    r = rescale_duration(c, n_out)
    t = np.linspace(0, T_target, n_out)
    return c, t, r.mean(0)

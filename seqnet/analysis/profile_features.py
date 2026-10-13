"""Connectivity-profile features (Braun Methods lines 979-1012, eq. 11-13; spec 6.13)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths

from ..conventions import centred_profile

__all__ = ["smoothdata_gaussian", "preprocess_profile", "ProfileFeatures", "profile_features"]


def smoothdata_gaussian(x: np.ndarray, window: int = 50, sigma_frac: float = 0.2, periodic: bool = False) -> np.ndarray:
    """Gaussian-weighted moving average mimicking MATLAB ``smoothdata(x, 'gaussian', window)``.

    [VERIFY, A5] MATLAB's kernel standard deviation for a given window is taken as ``window * sigma_frac`` (default
    window/5).  Non-periodic mode shrinks the window at the ends (MATLAB's default 'shrink' end treatment, i.e. the
    weights are renormalized over the available samples); periodic mode wraps around.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    half = window // 2
    offs = np.arange(-half, window - half)  # window samples (even window: one more on the right, as MATLAB)
    w = np.exp(-0.5 * (offs / (window * sigma_frac)) ** 2)
    out = np.empty(n)
    for i in range(n):
        j = i + offs
        if periodic:
            out[i] = np.sum(w * x[j % n]) / w.sum()
        else:
            ok = (j >= 0) & (j < n)
            out[i] = np.sum(w[ok] * x[j[ok]]) / w[ok].sum()
    return out


def preprocess_profile(cp: np.ndarray, window: int = 50, sigma_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre the profile, replace d = 0 by the mean of d = +-1, smooth (Braun lines 992-998).

    Returns (d, smoothed centred profile, raw self-weight cp_0).
    """
    d, c = centred_profile(cp)
    c = c.copy()
    i0 = int(np.nonzero(d == 0)[0][0])
    self_w = float(c[i0])
    c[i0] = 0.5 * (c[i0 - 1] + c[i0 + 1])
    return d, smoothdata_gaussian(c, window, sigma_frac), self_w


@dataclass(frozen=True)
class ProfileFeatures:
    """Braun's profile features: mean peak width (oscillation), range (eq. 12), asymmetry (eq. 13), plus self-weight."""

    peak_width: float
    range: float
    asymmetry: float
    self_weight: float
    peak_positions: np.ndarray


def profile_features(cp: np.ndarray, window: int = 50, sigma_frac: float = 0.2, prominence: float = 0.001) -> ProfileFeatures:
    """Features of one connectivity profile (spec 6.13 steps 1-7; MATLAB findpeaks with MinPeakProminence 0.001)."""
    d, s, self_w = preprocess_profile(cp, window, sigma_frac)
    pk, _ = find_peaks(s, prominence=prominence)
    widths = peak_widths(s, pk, rel_height=0.5)[0] if pk.size else np.array([])
    asym = float(np.min(np.abs(d[pk]))) if pk.size else float("nan")
    return ProfileFeatures(
        peak_width=float(np.mean(widths)) if widths.size else float("nan"),
        range=float(s.max() - s.min()),
        asymmetry=asym,
        self_weight=self_w,
        peak_positions=d[pk],
    )

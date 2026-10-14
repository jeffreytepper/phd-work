"""Connectivity profiles of individual neurons, and their comparison between groups of neurons along the sequence.

The mean connectivity profile cp_d = (1/N) sum_n W[n, (n + d) mod N] (Braun eq. 11; ``conventions.profile_from_matrix``)
averages per-neuron profiles over the sequence.  Here they are kept apart:

* incoming profile of neuron n:  P_in[n, d] = W[n, (n + d) mod N]   (weights onto n from the neuron d steps later);
* outgoing profile of neuron n:  P_out[n, d] = W[(n - d) mod N, n]  (weights from n onto the neuron d steps earlier),

so that both average over n to cp_d.  For a circulant matrix every row is the same profile; a sequence that is not
translation-invariant (a pulsed parameter, silenced neurons) shows up as profiles that change along the sequence.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..conventions import eigs_from_profile

__all__ = ["local_profiles", "window", "remove_self", "smooth_neurons", "neuron_groups", "group_profiles",
           "profile_summary", "local_spectra", "implied_speed", "binned_profiles", "variance_along_sequence",
           "dilation_fit", "difference_modes", "localize"]

Direction = Literal["in", "out"]


def local_profiles(W: np.ndarray, direction: Direction = "in") -> np.ndarray:
    """(..., N, N) matrices -> (..., N, N) profiles P[n, d], d = 0..N-1 (see module docstring)."""
    W = np.asarray(W)
    N = W.shape[-1]
    n = np.arange(N)[:, None]
    d = np.arange(N)[None, :]
    if direction == "in":
        return W[..., n, (n + d) % N]
    if direction == "out":
        return W[..., (n - d) % N, n]
    raise ValueError(f"unknown direction {direction!r}")


def window(P: np.ndarray, d_max: int) -> tuple[np.ndarray, np.ndarray]:
    """(d, P[..., d]) for d = -d_max..d_max."""
    d = np.arange(-d_max, d_max + 1)
    return d, np.take(P, d % P.shape[-1], axis=-1)


def remove_self(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Replace the self-weight (d = 0) by the mean of d = +-1 (Braun line 992), for display and shape features."""
    p = np.array(p, dtype=float, copy=True)
    i0 = int(np.nonzero(d == 0)[0][0])
    p[..., i0] = 0.5 * (p[..., i0 - 1] + p[..., i0 + 1])
    return p


def smooth_neurons(P: np.ndarray, width: int) -> np.ndarray:
    """Circular moving mean over ``width`` neighbouring neurons (axis -2 of (..., N, D))."""
    if width <= 1:
        return np.asarray(P, dtype=float)
    k = np.arange(width) - width // 2
    return np.mean([np.roll(P, s, axis=-2) for s in k], axis=0)


def neuron_groups(centre: float, N: int, windows: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    """Neurons whose sequence position lies in ``centre + [lo, hi]`` (inclusive, wrapped), per named window."""
    c = int(round(centre))
    return {name: (c + np.arange(int(lo), int(hi) + 1)) % N for name, (lo, hi) in windows.items()}


def group_profiles(P: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean profile over each group's neurons: (..., N, D) -> {name: (..., D)}."""
    return {name: np.asarray(P)[..., idx, :].mean(axis=-2) for name, idx in groups.items()}


def profile_summary(d: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Shape features of profiles p[..., d] (self-weight excluded from the shape features).

    self: W at d = 0; peak, peak_d: height and offset of the largest weight; trough: most negative weight;
    asymmetry: (sum_{d>0} p - sum_{d<0} p) / sum |p| over d != 0 (positive: input mainly from later neurons).
    """
    p = np.asarray(p, dtype=float)
    i0 = int(np.nonzero(d == 0)[0][0])
    shape = remove_self(d, p)
    pos, neg = d > 0, d < 0
    absum = np.abs(shape[..., d != 0]).sum(-1)
    return {
        "self": p[..., i0],
        "peak": shape.max(-1),
        "peak_d": d[np.argmax(shape, axis=-1)],
        "trough": shape.min(-1),
        "asymmetry": (shape[..., pos].sum(-1) - shape[..., neg].sum(-1)) / np.where(absum > 0, absum, 1.0),
    }


# ---------------------------------------------------------------------------------------------- views of the change
def local_spectra(P: np.ndarray, drop_self: bool = True, d_max: int | None = None) -> np.ndarray:
    """Per-neuron eigenvalue spectra lambda_n(k) = sum_d P[n, d] e^{-2 pi i k d / N} (full profiles, d = 0..N-1).

    The circulant eigenvalue convention of ``conventions.eigs_from_profile`` applied row by row; without ``d_max``
    their mean over n is the spectrum of the mean profile.  ``drop_self`` replaces d = 0 by the mean of d = +-1 first
    (the self-weight adds the same constant to every lambda_k).

    ``d_max`` keeps only the local profile, |d| <= d_max (circularly), before the transform.  Use it when the
    connectivity is not translation-invariant: a far-away change (e.g. the altered outgoing column of a few pulsed
    neurons) enters every other row at d = (pulsed - n) and adds a phase that rotates linearly with n, which a full-row
    spectrum would attribute to the row's own neuron.
    """
    P = np.array(P, dtype=float, copy=True)
    if drop_self:
        P[..., 0] = 0.5 * (P[..., 1] + P[..., -1])
    if d_max is not None:
        N = P.shape[-1]
        dist = np.minimum(np.arange(N), N - np.arange(N))
        P[..., dist > d_max] = 0.0
    return eigs_from_profile(P)


def implied_speed(lam: np.ndarray, k, tau: float, N: int) -> np.ndarray:
    """Bump speed (neurons / s) implied by the harmonics: Im lambda_k = c k Re lambda_k, v = N c / (2 pi tau).

    A travelling bump supported by W needs Im lambda_k = k omega tau Re lambda_k with omega = 2 pi v / N (Braun Suppl.
    Cor. 3.2, supplement convention; test T8).  ``k`` is one harmonic (c from that ratio alone) or several, fitted by
    least squares weighted by Re lambda_k (robust where a single Re lambda_k is near zero).  FORCE here trains the
    methods convention, under which the ratio also depends on the bump's shape: compare with a control, and only for
    conditions that keep the shape.
    """
    ks = np.atleast_1d(np.asarray(k))
    re, im = lam[..., ks].real, lam[..., ks].imag
    c = (ks * re * im).sum(-1) / (ks**2 * re**2).sum(-1)
    return N * c / (2 * np.pi * tau)


def binned_profiles(P: np.ndarray, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean profile over consecutive bins of ``bin_size`` neurons (axis -2): (..., n_bins, D) and the bin centres."""
    N = P.shape[-2]
    n_bins = N // bin_size
    B = P[..., : n_bins * bin_size, :].reshape(P.shape[:-2] + (n_bins, bin_size, P.shape[-1])).mean(-2)
    return B, (np.arange(n_bins) + 0.5) * bin_size


def variance_along_sequence(P: np.ndarray, neurons: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """How much the profile varies from neuron to neuron, per offset d, against the noise floor.

    P: (realizations, N, D).  ``neurons`` restricts the statistics (e.g. away from the snippet ends).
    across: variance over neurons of the realization-mean profile; noise: mean over neurons of the variance of that
    mean across realizations (what ``across`` would be for a translation-invariant matrix); excess = across - noise.
    """
    P = np.asarray(P, dtype=float)
    if neurons is not None:
        P = P[:, neurons]
    R = P.shape[0]
    across = P.mean(0).var(0)
    noise = (P.var(0, ddof=1) / R).mean(0) if R > 1 else np.zeros_like(across)
    return {"across": across, "noise": noise, "excess": across - noise}


def dilation_fit(d: np.ndarray, P: np.ndarray, ref: np.ndarray, scales: np.ndarray = np.geomspace(0.5, 2.0, 61),
                 min_r2: float = 0.0) -> dict[str, np.ndarray]:
    """Fit each profile as a stretched copy of a reference, p_n(d) ~ a_n ref(d / s_n), over a grid of scales s.

    s < 1: the profile is compressed toward d = 0 (lobes closer); s > 1: stretched.  Returns scale, amplitude a and
    r2 per profile; scale and amplitude are NaN where r2 < ``min_r2`` (the profile is no stretched copy of the
    reference, and the best scale would only sit at the edge of the grid).  Self-weights should be removed from both
    (``remove_self``).
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    R = np.stack([np.interp(d / s, d, ref, left=0.0, right=0.0) for s in scales])  # (S, D)
    dots = P @ R.T  # (n, S)
    rr = (R**2).sum(1)
    pp = (P**2).sum(1, keepdims=True)
    r2 = dots**2 / (rr * np.where(pp > 0, pp, 1.0))
    best = np.argmax(r2, axis=1)
    i = np.arange(P.shape[0])
    r2b = r2[i, best]
    ok = r2b >= min_r2
    return {"scale": np.where(ok, scales[best], np.nan), "amplitude": np.where(ok, dots[i, best] / rr[best], np.nan),
            "r2": r2b}


def difference_modes(dP: np.ndarray, n_modes: int = 3) -> dict[str, np.ndarray]:
    """Principal shapes of a profile change: SVD of (N, D) differences (pulsed - control) across neurons.

    modes: (n_modes, D) unit shapes in d; scores: (N, n_modes) how much of each shape every neuron carries;
    explained: fraction of the total squared change per mode.  Mode signs are fixed so the largest |score| is positive.
    """
    U, s, Vt = np.linalg.svd(np.asarray(dP, dtype=float), full_matrices=False)
    scores = U[:, :n_modes] * s[:n_modes]
    sign = np.sign(scores[np.argmax(np.abs(scores), axis=0), np.arange(n_modes)])
    return {"modes": Vt[:n_modes] * sign[:, None], "scores": scores * sign, "explained": s[:n_modes] ** 2 / (s**2).sum()}


def localize(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Peak position, height and full width at half maximum of a non-negative curve y(x) (half-maximum crossings
    interpolated linearly on each side of the peak; the curve's ends if it never drops to half)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    half = y[i] / 2

    def crossing(step):
        j = i
        while 0 <= j + step < y.size and y[j + step] >= half:
            j += step
        k = j + step
        if not 0 <= k < y.size:
            return x[j]
        return x[j] + (x[k] - x[j]) * (y[j] - half) / (y[j] - y[k])

    left, right = crossing(-1), crossing(+1)
    return {"peak": float(x[i]), "height": float(y[i]), "fwhm": float(right - left), "left": float(left),
            "right": float(right)}

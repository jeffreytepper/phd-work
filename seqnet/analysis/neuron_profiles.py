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

__all__ = ["local_profiles", "window", "remove_self", "smooth_neurons", "neuron_groups", "group_profiles",
           "profile_summary"]

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

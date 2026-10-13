"""Structured and unstructured perturbation protocols (Braun Methods lines 1130-1222; spec 6.15).

[VERIFY] The Methods do not say whether perturbed and unperturbed runs share the noise realization.  Default
``common_noise=True`` (same seed for both), so that the correlation measures the effect of the perturbation rather
than of independent noise; ``False`` uses independent seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.rate_rnn import Perturbation, RNNParams, simulate
from ..training.rls import pearson

__all__ = ["ensembles", "active_window", "structured_grid", "align_rows", "unstructured", "PerturbResult"]


def ensembles(N: int, f_E: float) -> list[np.ndarray]:
    """floor(1/f_E) contiguous ensembles of floor(N f_E) neurons in the sequence ordering (Braun line 1146)."""
    size = int(np.floor(N * f_E))
    e = int(np.floor(1 / f_E))
    return [np.arange(i * size, (i + 1) * size) for i in range(e)]


def active_window(r: np.ndarray, ens: np.ndarray, window_steps: int, n_windows: int) -> int:
    """Index of the first-half time window in which the ensemble's mean rate peaks (unperturbed run)."""
    m = r[: window_steps * n_windows, ens].mean(1)
    return int(np.argmax(m) // window_steps)


def align_rows(CC: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Shift each row so that the ensemble's own active window sits in the middle column (Braun lines 1215-1222)."""
    n_w = CC.shape[1]
    mid = n_w // 2
    return np.stack([np.roll(row, mid - a) for row, a in zip(CC, active)])


@dataclass
class PerturbResult:
    CC: np.ndarray  # (n_ensembles, n_windows) correlations
    active: np.ndarray  # active window of each ensemble
    aligned: np.ndarray  # rows aligned so the active window is the middle column
    aligned_mean: np.ndarray
    r_unperturbed: np.ndarray
    examples: dict


def _run(J, h0, n_steps, params, seed, perts, noise=True):
    return simulate(J, h0, n_steps, params, np.random.default_rng(seed), noise=noise, perturbations=perts)["r"]


def structured_grid(
    J: np.ndarray,
    h0: np.ndarray,
    kind: str,
    f_E: float = 0.2,
    amplitude: float = 0.0,
    params: RNNParams = RNNParams(),
    n_steps: int = 2099,
    window_steps: int = 209,
    n_windows: int = 5,
    seed: int = 0,
    common_noise: bool = True,
    keep_examples: list[tuple[int, int]] | None = None,
) -> PerturbResult:
    """Perturb each (ensemble, window) of the first half and correlate perturbed vs unperturbed activity from the end of
    the window to the end of the run (Braun Methods 'Structured perturbations' and 'Quantification')."""
    N = J.shape[0]
    r0 = _run(J, h0, n_steps, params, seed, [])
    ens = ensembles(N, f_E)
    CC = np.zeros((len(ens), n_windows))
    act = np.array([active_window(r0, e, window_steps, n_windows) for e in ens])
    examples = {}
    for i, e in enumerate(ens):
        for w in range(n_windows):
            start, stop = w * window_steps, (w + 1) * window_steps
            p = Perturbation(kind, e, start, stop, amplitude)
            r = _run(J, h0, n_steps, params, seed if common_noise else seed + 1 + i * n_windows + w, [p])
            CC[i, w] = pearson(r[stop:], r0[stop:])
            if keep_examples and (i, w) in keep_examples:
                examples[(i, w)] = r
    al = align_rows(CC, act)
    return PerturbResult(CC=CC, active=act, aligned=al, aligned_mean=al.mean(0), r_unperturbed=r0, examples=examples)


def unstructured(
    J: np.ndarray,
    h0: np.ndarray,
    fraction: float,
    gain_p: float,
    params: RNNParams = RNNParams(),
    n_steps: int = 2099,
    seed: int = 0,
    common_noise: bool = True,
    select_seed: int = 0,
) -> dict:
    """Noise injection into a random fraction Q of neurons during the first half; correlation over the second half
    (Braun eq. 22 and lines 1199-1202)."""
    N = J.shape[0]
    Q = np.random.default_rng(select_seed).choice(N, int(round(fraction * N)), replace=False)
    half = n_steps // 2
    r0 = _run(J, h0, n_steps, params, seed, [])
    r = _run(J, h0, n_steps, params, seed if common_noise else seed + 1, [Perturbation("noise", Q, 0, half, gain_p)])
    return {"corr": pearson(r[half:], r0[half:]), "r": r, "r0": r0, "Q": Q}

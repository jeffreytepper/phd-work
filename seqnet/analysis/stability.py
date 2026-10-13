"""Floquet analysis of the periodic orbit of a frozen, noise-free rate network (spec 6.16; Section 3.4 fact 3).

Support is not stability: the support condition fixes W on the active subspace but says nothing about whether the
sequence is an attracting orbit.  For the Euler-discretized dynamics h <- h + a(-h + J phi(h)), a = dt/tau, the
tangent map of one step is I + a(-I + J diag(phi'(h))).  The monodromy matrix is the product over one period;
its eigenvalues are the Floquet multipliers.  The phase direction (the flow vector) has multiplier 1 when the period
is an integer number of steps; for a non-integer period the flow-direction multiplier deviates by O(step / period),
which is reported as ``closure_error``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from ..conventions import fft
from ..models.rate_rnn import RNNParams

__all__ = ["FloquetResult", "settle", "estimate_period", "monodromy", "floquet"]


def _step(J, h, a):
    return h + a * (-h + J @ expit(h))


def settle(J: np.ndarray, h0: np.ndarray, n_steps: int, params: RNNParams = RNNParams(), record_every: int = 1):
    """Noise-free integration; returns (h trajectory every ``record_every`` steps, final h)."""
    a = params.dt / params.tau
    h = np.array(h0, float)
    rec = []
    for k in range(n_steps):
        if k % record_every == 0:
            rec.append(h.copy())
        h = _step(J, h, a)
    return np.asarray(rec), h


def estimate_period(J: np.ndarray, h_start: np.ndarray, guess_steps: int, params: RNNParams = RNNParams(),
                    search: float = 0.3) -> tuple[int, float]:
    """Integer number of steps n minimizing |h(n) - h(0)| for n within (1 +- search) * guess, and that distance."""
    lo, hi = int(guess_steps * (1 - search)), int(guess_steps * (1 + search))
    traj, _ = settle(J, h_start, hi + 1, params)
    d = np.linalg.norm(traj[lo:hi + 1] - h_start, axis=1)
    i = int(np.argmin(d))
    return lo + i, float(d[i] / np.linalg.norm(h_start))


def monodromy(J: np.ndarray, h_start: np.ndarray, n_steps: int, params: RNNParams = RNNParams()) -> tuple[np.ndarray, np.ndarray]:
    """Product of one-step tangent maps I + a(-I + J diag(phi'(h))) along the trajectory (exact for the Euler map)."""
    a = params.dt / params.tau
    N = J.shape[0]
    M = np.eye(N)
    h = np.array(h_start, float)
    for _ in range(n_steps):
        r = expit(h)
        M = M + a * (-M + J @ ((r * (1 - r))[:, None] * M))
        h = _step(J, h, a)
    return M, h


@dataclass
class FloquetResult:
    multipliers: np.ndarray  # sorted by decreasing modulus
    period_steps: int
    closure_error: float  # |h(T) - h(0)| / |h(0)| on the settled orbit
    flow_multiplier: complex  # Rayleigh quotient of M along the flow direction
    sequence_kept: bool  # the noise-free network still produces a periodic travelling sequence
    leading_transverse: np.ndarray  # largest |multipliers| excluding the one closest to the flow multiplier
    fourier_content: np.ndarray  # power of the leading transverse eigenvector per spatial mode (|k| <= 20)


def floquet(J: np.ndarray, h0: np.ndarray, guess_steps: int, params: RNNParams = RNNParams(), burn_periods: int = 6,
            closure_tol: float = 0.05, n_transverse: int = 5) -> FloquetResult:
    """Settle onto the orbit, estimate the period, and return the Floquet multipliers (spec 6.16)."""
    _, h = settle(J, h0, burn_periods * guess_steps, params)
    n, err = estimate_period(J, h, guess_steps, params)
    M, h_end = monodromy(J, h, n, params)
    ev, vecs = np.linalg.eig(M)
    order = np.argsort(-np.abs(ev))
    ev, vecs = ev[order], vecs[:, order]
    a = params.dt / params.tau
    f = -h + J @ expit(h)  # flow direction
    flow_mult = complex(f @ (M @ f) / (f @ f))
    i_flow = int(np.argmin(np.abs(ev - flow_mult)))
    trans = np.delete(ev, i_flow)
    v = np.delete(vecs, i_flow, axis=1)[:, 0]
    pw = np.abs(fft(v)) ** 2
    N = J.shape[0]
    fc = np.array([pw[k % N] + (pw[-k % N] if k else 0) for k in range(21)]) / pw.sum()
    kept = bool(err < closure_tol and np.std(expit(h)) > 1e-3)
    return FloquetResult(multipliers=ev, period_steps=n, closure_error=err, flow_multiplier=flow_mult, sequence_kept=kept,
                         leading_transverse=trans[:n_transverse], fourier_content=fc)

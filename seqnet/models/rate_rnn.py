"""Braun et al. rate RNN (Braun Methods eq. 1-3) with perturbation hooks (Methods eq. 21-22, lines 1130-1192).

    h(t) = h(t - dt) + dt/tau (-h(t - dt) + z(t - dt)),   z = J r + zeta,   r = phi(h) = 1/(1 + e^{-h}),
    zeta_i = gain sqrt(tau/dt) nu_i,  nu_i ~ N(0, 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.special import expit

__all__ = ["RNNParams", "Perturbation", "phi", "noise_scale", "simulate"]


def phi(h: np.ndarray) -> np.ndarray:
    """Logistic sigmoid (Braun Methods, line 778)."""
    return expit(h)


@dataclass(frozen=True)
class RNNParams:
    """tau = 0.3224 s, dt = 0.0081 s, noise gain 0.5 (Braun Methods eq. 1-2)."""

    tau: float = 0.3224
    dt: float = 0.0081
    noise_gain: float = 0.5


def noise_scale(p: RNNParams) -> float:
    """Standard deviation of zeta: gain sqrt(tau/dt) (Braun eq. 2)."""
    return p.noise_gain * np.sqrt(p.tau / p.dt)


@dataclass
class Perturbation:
    """One perturbation applied to a set of neurons over integration steps [start, stop).

    kind:
      * ``'silence'``: r_i = 0 (h keeps evolving) (Braun eq. 21);
      * ``'current'``: z_i += amplitude (step current I) (Braun lines 1170-1178);
      * ``'noise'``: z_i += amplitude * sqrt(tau/dt) nu_i (noise gain gain_p) (Braun eq. 22).
    """

    kind: Literal["silence", "current", "noise"]
    neurons: np.ndarray
    start: int
    stop: int
    amplitude: float = 0.0
    mask: np.ndarray = field(init=False, repr=False)

    def build(self, N: int) -> "Perturbation":
        m = np.zeros(N, bool)
        m[np.asarray(self.neurons)] = True
        self.mask = m
        return self


def simulate(
    J: np.ndarray,
    h0: np.ndarray,
    n_steps: int,
    params: RNNParams = RNNParams(),
    rng: np.random.Generator | None = None,
    noise: bool = True,
    perturbations: Sequence[Perturbation] = (),
    record_every: int = 1,
    record_inputs: bool = False,
) -> dict:
    """Integrate the network for ``n_steps`` Euler steps (Braun eq. 3) from h(0) = h0.

    Returns {"r": (n_rec, N) rates recorded every ``record_every`` steps (after silencing, i.e. the rates the
    network actually transmits), "t": step indices, "h_final": h, and optionally "z": inputs incl. perturbations}.
    """
    J = np.asarray(J, dtype=float)
    N = J.shape[0]
    h = np.array(h0, dtype=float, copy=True)
    ns = noise_scale(params)
    a = params.dt / params.tau
    rng = np.random.default_rng() if rng is None else rng
    perts = [p.build(N) for p in perturbations]
    rec_r, rec_z, rec_t = [], [], []
    for k in range(n_steps):
        r = expit(h)
        for p in perts:
            if p.kind == "silence" and p.start <= k < p.stop:
                r[p.mask] = 0.0
        z = J @ r
        if noise:
            z += ns * rng.standard_normal(N)
        for p in perts:
            if p.start <= k < p.stop:
                if p.kind == "current":
                    z[p.mask] += p.amplitude
                elif p.kind == "noise":
                    z[p.mask] += p.amplitude * np.sqrt(params.tau / params.dt) * rng.standard_normal(p.mask.sum())
        if k % record_every == 0:
            rec_r.append(r.copy())
            rec_t.append(k)
            if record_inputs:
                rec_z.append(z.copy())
        h += a * (-h + z)
    out = {"r": np.asarray(rec_r), "t": np.asarray(rec_t), "h_final": h}
    if record_inputs:
        out["z"] = np.asarray(rec_z)
    return out

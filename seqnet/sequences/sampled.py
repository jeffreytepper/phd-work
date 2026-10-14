"""Sequences known on a uniform time grid (simulated networks, recordings), linearly interpolated in between."""
from __future__ import annotations

from functools import cached_property

import numpy as np

from .base import Sequence

__all__ = ["SampledSequence"]


class SampledSequence(Sequence):
    """Rates ``samples`` (N, n) at times t_j = j dt, j = 0..n-1.

    ``periodic``: the grid holds whole periods of length n dt and time wraps around; otherwise the sequence is
    defined on [0, (n - 1) dt] and evaluating outside it is an error.  ``rates_dt`` interpolates the second-order
    finite difference of the samples.
    """

    def __init__(self, samples: np.ndarray, dt: float, periodic: bool = False):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] < 2:
            raise ValueError("samples must be (N, n) with n >= 2")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.samples = samples
        self.dt = float(dt)
        self.periodic = bool(periodic)

    @property
    def N(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.dt

    @property
    def duration(self) -> float | None:
        return None if self.periodic else (self.n_samples - 1) * self.dt

    def _interp(self, values: np.ndarray, t) -> np.ndarray:
        s = np.asarray(t, dtype=float) / self.dt
        n = self.n_samples
        if self.periodic:
            s = s % n
            ext = np.concatenate([values, values[:, :1]], axis=1)
        else:
            if np.any(s < -1e-9) or np.any(s > n - 1 + 1e-9):
                raise ValueError(f"time outside the sampled interval [0, {self.duration:g}] s")
            s = np.clip(s, 0, n - 1)
            ext = values
        i = np.minimum(np.floor(s).astype(int), ext.shape[1] - 2)
        f = s - i
        return ext[:, i] * (1 - f) + ext[:, i + 1] * f

    def rates(self, t):
        return self._interp(self.samples, t)

    @cached_property
    def _samples_dt(self) -> np.ndarray:
        if self.periodic:
            return (np.roll(self.samples, -1, axis=1) - np.roll(self.samples, 1, axis=1)) / (2 * self.dt)
        return np.gradient(self.samples, self.dt, axis=1)

    def rates_dt(self, t):
        return self._interp(self._samples_dt, t)

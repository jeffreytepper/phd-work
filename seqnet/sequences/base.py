"""The sequence interface every generator and dataset implements.

A :class:`Sequence` is the target rate matrix f_n(t) of N neurons as a function of continuous time (seconds).
Consumers (FORCE training, simulations, figures) depend only on this interface:

* ``N``, ``rates(t)`` -> (N, n_t), ``rates_dt(t)`` -> (N, n_t);
* ``duration``: None when defined for all t >= 0, else the last valid time;
* ``draw_snippet``: the target over one training iteration (Braun Methods lines 913-925), which recorded datasets
  override with Braun's bin-per-update rule.

Sequences whose bump has a fixed shape and travels at constant speed with every neuron taking part (circulant
statistics; Braun Suppl. Sec. 2-5) additionally provide a :class:`StationaryWave` view for the spectral analysis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .registry import Registry

__all__ = ["Sequence", "SEQUENCES", "Snippet", "StationaryWave"]


@dataclass(frozen=True)
class Snippet:
    """Target over one training iteration: column u belongs to update u.

    ``rates_dt`` is None when the source has no exact derivative (recorded data); the y* target then uses a finite
    difference of x* with step ``spacing``.
    """

    t0: float
    times: np.ndarray
    rates: np.ndarray
    rates_dt: np.ndarray | None
    spacing: float


class Sequence(ABC):
    """Target rates f_n(t), n = 0..N-1, in sequence order."""

    N: int  # number of neurons (a field or a property of each implementation)

    @abstractmethod
    def rates(self, t: np.ndarray) -> np.ndarray:
        """(N, n_t) rates at times t (seconds)."""

    @abstractmethod
    def rates_dt(self, t: np.ndarray) -> np.ndarray:
        """(N, n_t) time derivative of :meth:`rates`."""

    @property
    def duration(self) -> float | None:
        """Last valid time; None if the sequence is defined for all t >= 0."""
        return None

    def draw_snippet(self, rng: np.random.Generator, offsets: np.ndarray, start_max: float) -> Snippet:
        """Target at t0 + offsets with t0 ~ U[0, start_max], kept inside the sequence (Braun lines 918-921).

        Exactly one draw from ``rng``.
        """
        offsets = np.asarray(offsets, dtype=float)
        hi = start_max
        if self.duration is not None:
            hi = min(start_max, self.duration - float(offsets[-1]))
            if hi < 0:
                raise ValueError(f"sequence of {self.duration:g} s is shorter than a snippet of {offsets[-1]:g} s")
        t0 = float(rng.uniform(0.0, hi))
        times = t0 + offsets
        spacing = float(offsets[1] - offsets[0]) if offsets.size > 1 else 0.0
        return Snippet(t0=t0, times=times, rates=self.rates(times), rates_dt=self.rates_dt(times), spacing=spacing)


SEQUENCES: Registry[Sequence] = Registry("sequence", Sequence)


@dataclass(frozen=True)
class StationaryWave:
    """f_n(t) = f0(t - n period / N): one scalar bump f0 shared by all neurons (Braun Suppl. Sec. 2).

    ``profile`` is f0 (the rates of neuron 0); ``coeffs`` its temporal Fourier coefficients
    f_hat_k = (1/period) int_0^period f0(t) e^{-i k omega t} dt, when known in closed form.
    """

    N: int
    period: float
    profile: Callable[[np.ndarray], np.ndarray]
    coeffs: Callable[[np.ndarray], np.ndarray] | None = None

    @property
    def omega(self) -> float:
        return 2 * np.pi / self.period

    def fourier_coeffs(self, k: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            raise NotImplementedError("this wave has no closed-form Fourier series; use a temporal DFT of profile()")
        return self.coeffs(np.asarray(k))

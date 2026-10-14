"""Putting sequences on a common clock.

Sequences come with their own time units: Braun's targets cycle every T = 8.5 s, the Lehr ring network at about one
cycle per second of model time (tau = 100 ms), datasets at their recording's pace.  The FORCE network has fixed time
constants (tau = 0.3224 s), so what it learns depends on the target's period relative to tau.  The convention here:
Braun's 8.5 s cycle is the reference training timescale (``BRAUN_PERIOD``); parametric Lehr bumps use it by default,
and any other sequence is brought to it, or to any period, by :class:`TimeScaled`.
"""
from __future__ import annotations

import numpy as np

from .base import SEQUENCES, Sequence
from .measures import mean_period

__all__ = ["TimeScaled", "time_scaled", "natural_period"]


class TimeScaled(Sequence):
    """f'(t) = f(t / factor): the same sequence running ``factor`` times slower (factor > 1) or faster.

    Training snippets are drawn on the scaled clock (the default continuous rule), also for recorded data.
    """

    def __init__(self, base: Sequence, factor: float):
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.base = base
        self.factor = float(factor)

    @property
    def N(self) -> int:
        return self.base.N

    @property
    def duration(self) -> float | None:
        d = self.base.duration
        return None if d is None else d * self.factor

    def rates(self, t):
        return self.base.rates(np.asarray(t, dtype=float) / self.factor)

    def rates_dt(self, t):
        return self.base.rates_dt(np.asarray(t, dtype=float) / self.factor) / self.factor


def natural_period(sequence: Sequence, t: np.ndarray | None = None) -> float:
    """Cycle duration of a sequence: exact for stationary waves, else N over the mean bump speed.

    The speed is measured at times ``t``; by default the sample grid of a sampled sequence or 4000 points over the
    duration of a bounded one.  Unbounded modulated sequences need ``t``.
    """
    if getattr(sequence, "is_stationary", False):
        return sequence.stationary().period
    if t is None:
        if getattr(sequence, "times", None) is not None:
            t = sequence.times
        elif sequence.duration is not None:
            t = np.linspace(0.0, sequence.duration, 4000)
        else:
            raise ValueError("unbounded modulated sequence: give the times t to measure its period over")
    t = np.asarray(t, dtype=float)
    return mean_period(sequence.rates(t), float(t[1] - t[0]))


@SEQUENCES.register("time_scaled")
def time_scaled(sequence: dict | Sequence, factor: float | None = None, period: float | None = None,
                measure_until: float | None = None) -> TimeScaled:
    """Rescale ``sequence`` by ``factor``, or so that its natural period becomes ``period`` (seconds).

    ``measure_until`` bounds the window [0, measure_until] over which an unbounded modulated sequence's period is
    measured.
    """
    from . import build_sequence

    base = build_sequence(sequence)
    if (factor is None) == (period is None):
        raise ValueError("give exactly one of factor and period")
    if period is not None:
        t = None if measure_until is None else np.linspace(0.0, measure_until, 4000)
        factor = period / natural_period(base, t)
    return TimeScaled(base, factor)

"""Bump kinematics measured from a rate matrix on a ring (population-vector estimates)."""
from __future__ import annotations

import numpy as np

__all__ = ["bump_position", "bump_speed"]


def bump_position(rates: np.ndarray) -> np.ndarray:
    """Unwrapped bump centre (neurons) per column of ``rates`` (N, n_t): the circular mean of the activity."""
    rates = np.asarray(rates, dtype=float)
    N = rates.shape[0]
    z = np.exp(2j * np.pi * np.arange(N) / N) @ rates
    return np.unwrap(np.angle(z)) * N / (2 * np.pi)


def bump_speed(rates: np.ndarray, dt: float) -> np.ndarray:
    """Instantaneous bump speed (neurons / s) between successive columns."""
    return np.diff(bump_position(rates)) / dt

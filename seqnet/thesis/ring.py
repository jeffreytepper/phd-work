"""The ring and the observation schedule (Thesis §2.1, §2.3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Ring", "Schedule", "wrap"]


def wrap(u: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi) (periodic domain, Thesis A1)."""
    return (np.asarray(u, dtype=float) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Ring:
    """N neurons at r_i = 2 pi i / N with spacing dr = 2 pi / N (Thesis §2.1)."""

    N: int

    @property
    def positions(self) -> np.ndarray:
        return 2 * np.pi * np.arange(self.N) / self.N

    @property
    def dr(self) -> float:
        return 2 * np.pi / self.N

    @property
    def centred_positions(self) -> np.ndarray:
        """Positions wrapped to [-pi, pi) (used for spatial widths, Thesis eq. 5.10)."""
        return wrap(self.positions)


@dataclass(frozen=True)
class Schedule:
    """T_obs = N/d uniformly spaced bump positions c_t = t * d * 2 pi / N (Thesis §2.3).

    ``d`` may be non-integer; then the positions fall between neurons and the design matrix uses
    the kernel's trigonometric interpolant.  Construct with ``Schedule(N, d)`` or
    ``Schedule.from_T(N, T_obs)``.
    """

    N: int
    d: float = 1.0
    offset: float = 0.0  # phase of the first observation position (radians)
    T_obs: int = field(init=False)

    def __post_init__(self) -> None:
        T = self.N / self.d
        if abs(T - round(T)) > 1e-9:
            raise ValueError(f"N/d must be an integer number of observations (got {T})")
        object.__setattr__(self, "T_obs", int(round(T)))

    @classmethod
    def from_T(cls, N: int, T_obs: int, offset: float = 0.0) -> "Schedule":
        return cls(N=N, d=N / T_obs, offset=offset)

    @property
    def positions(self) -> np.ndarray:
        return self.offset + np.arange(self.T_obs) * self.d * 2 * np.pi / self.N

    @property
    def on_grid(self) -> bool:
        """True when every c_t coincides with a neuron position."""
        g = self.positions / (2 * np.pi / self.N)
        return bool(np.allclose(g, np.round(g), atol=1e-9))

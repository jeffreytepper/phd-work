"""Which neurons take part in the sequence: per-neuron gains g_n(t) in [0, 1].

Silencing a subset by clustered inhibition multiplies the activity by a diagonal projection P (Lehr et al. 2024
eq. 7; Lehr et al. 2025, eq. S12).  Switching between inhibitory ensembles switches P over time and rotates the
manifold (Lehr et al. 2024 Fig. 5).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .registry import Registry

__all__ = ["Participation", "PARTICIPATION", "AllActive", "RandomSilencing", "Switching"]


class Participation(ABC):
    """Per-neuron gains, (N, n_t) or broadcastable to it (e.g. (N, 1) when constant in time)."""

    @abstractmethod
    def gains(self, N: int, t: np.ndarray) -> np.ndarray: ...

    @property
    def is_full(self) -> bool:
        """True when every neuron participates with gain 1 at all times."""
        return False

    @property
    def is_static(self) -> bool:
        return True


PARTICIPATION: Registry[Participation] = Registry("participation", Participation)


@PARTICIPATION.register("all_active")
@dataclass(frozen=True)
class AllActive(Participation):
    def gains(self, N, t):
        return np.ones((N, 1))

    @property
    def is_full(self) -> bool:
        return True


@PARTICIPATION.register("random_silencing")
@dataclass(frozen=True)
class RandomSilencing(Participation):
    """round(p_inh N) neurons, drawn uniformly without replacement from ``seed``, are silenced (gain 0).

    Lehr et al. (2025) Suppl. 'Projection matrices': numpy.random.choice without replacement.  The draw depends only
    on (N, p_inh, seed), so one seed is one inhibitory ensemble.
    """

    p_inh: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if not 0.0 <= self.p_inh <= 1.0:
            raise ValueError("p_inh must lie in [0, 1]")

    def mask(self, N: int) -> np.ndarray:
        """Diagonal of P: 1 for active, 0 for silenced neurons."""
        p = np.ones(N)
        n_off = int(round(self.p_inh * N))
        p[np.random.default_rng(self.seed).choice(N, n_off, replace=False)] = 0.0
        return p

    def gains(self, N, t):
        return self.mask(N)[:, None]

    @property
    def is_full(self) -> bool:
        return self.p_inh == 0.0


@PARTICIPATION.register("switching")
@dataclass(frozen=True)
class Switching(Participation):
    """Stage i is active on [switch_times[i-1], switch_times[i]); stage 0 before the first switch (Lehr 2024 Fig. 5l)."""

    stages: tuple
    switch_times: tuple

    def __post_init__(self):
        stages = tuple(PARTICIPATION.coerce(s) for s in self.stages)
        times = tuple(float(x) for x in self.switch_times)
        if len(stages) != len(times) + 1:
            raise ValueError("need one more stage than switch times")
        if list(times) != sorted(times):
            raise ValueError("switch times must be increasing")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "switch_times", times)

    def gains(self, N, t):
        t = np.asarray(t, dtype=float)
        which = np.searchsorted(self.switch_times, t, side="right")
        out = np.empty((N, t.size))
        for i, st in enumerate(self.stages):
            sel = which == i
            if sel.any():
                out[:, sel] = np.broadcast_to(st.gains(N, t[sel]), (N, int(sel.sum())))
        return out

    @property
    def is_static(self) -> bool:
        return False

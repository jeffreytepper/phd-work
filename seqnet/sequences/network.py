"""Mechanistic sequence generator: the controlled ring network of Lehr, Kumar & Tetzlaff (2024).

    tau dI/dt = -I + P(t) S(t) (W r + (E - I)(t)),     r = F(I) = min(max(I, 0), 1),

with x_i = i on a ring of N neurons and the circulant weights of Lehr et al. (2025, eq. S10)

    w_ij = w_E exp(-dx_ij^2 / (2 sigma^2)) - w_I,   dx_ij = ring distance from x_i to c_j = x_j + s.

The three controls are the paper's: gain S(t) sets the speed (Fig. 3-4), a global transient current (E - I)(t) the
bump width and so the trajectory shape (Fig. 2), and the clustered-inhibition projection P(t) the subspace (Fig. 5).
The run starts from a Gaussian bump r_i(0) = p_i exp(-(x_i - x_c)^2 / (2 sigma_0^2)) with I(0) = r(0) and
x_c = sigma_0 = sigma (Lehr et al. 2025, eq. S13 and Table 3), integrated with forward Euler.

State.  Read literally (``bounded_state=False``), the current I is unbounded and r = F(I).  Once the bump saturates,
S then scales every input and the threshold crossing at the front alike, so the speed no longer depends on S (about
0.47 rot/s for S = 1 to 1000 at the printed parameters), contrary to the paper's Fig. 3 and S3.  With the state
itself kept in [0, 1] (``bounded_state=True``, default: I <- F(I + dt/tau (...)), r = I) the network reproduces
Fig. S3: speed rises with S and saturates at one shift per Euler step, 19 / 39 / 59 rot/s for shifts 0.02 / 0.04 /
0.06 at dt = 1 ms (paper: about 19 / 38 / 58); and the drive (E - I) changes the bump width with little effect on
speed (Fig. 2).  So the default is the reading that matches the published figures, not the printed equation.

Size.  ``w_E`` and ``w_I`` are the paper's values at its ``reference_N`` = 1000 neurons; the weights used are scaled by
reference_N / N, so the summed input, and with it speed, width and the effect of every control, is the same at any N
(without it a 501-neuron ring runs at 0.61 instead of 1.0 rot/s, and a 200-neuron ring loses the bump at S = 1).

Units.  ``sigma`` and ``shift`` are fractions of the ring: Lehr et al. (2024) print sigma = s = 0.04 at N = 1000, and
their Fig. S3 labels the shift 0.02-0.06 (40 neurons for 0.04).  Time is the model's own, in seconds (tau = 100 ms;
about 1 cycle/s at S = 1).  To train on it at another timescale, wrap it: ``{"kind": "time_scaled", "period": 8.5,
"sequence": {"kind": "lehr_network", ...}}`` (:mod:`.timescale`).

Time 0 of the output is the end of a ``burn_in`` (default 0.5 s = 5 tau; the bump settles from its initial Gaussian in
about 1.5 tau).  The controls are functions of output time and are evaluated at negative times during the burn-in.
``rescale`` divides W by the fraction of active neurons (Lehr et al. 2025, eq. S15) so silencing does not change the
dominant eigenvalue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import SEQUENCES
from .participation import PARTICIPATION, AllActive, Participation
from .profiles import wrap
from .sampled import SampledSequence
from .timecourses import Constant, TimeCourse, as_timecourse

__all__ = ["LehrRingNetwork", "lehr_network"]


@dataclass(frozen=True)
class LehrRingNetwork:
    N: int = 1000
    tau: float = 0.1
    dt: float = 1e-3
    duration: float = 2.0
    w_E: float = 0.05
    w_I: float = 0.01
    reference_N: int = 1000
    sigma: float = 0.04
    shift: float = 0.04
    gain: TimeCourse = field(default_factory=lambda: Constant(1.0))
    drive: TimeCourse = field(default_factory=lambda: Constant(0.0))
    participation: Participation = field(default_factory=AllActive)
    rescale: bool = True
    bounded_state: bool = True
    record_every: int = 1
    burn_in: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "gain", as_timecourse(self.gain))
        object.__setattr__(self, "drive", as_timecourse(self.drive))
        object.__setattr__(self, "participation", PARTICIPATION.coerce(self.participation))

    @property
    def kernel(self) -> np.ndarray:
        """First column k_d = w_{d,0} of the circulant W (W_ij = k_{(i - j) mod N})."""
        d = np.arange(self.N)
        dist = np.abs(wrap(d - self.shift * self.N, self.N))
        scale = self.reference_N / self.N
        return scale * (self.w_E * np.exp(-(dist**2) / (2 * (self.sigma * self.N) ** 2)) - self.w_I)

    def weights(self) -> np.ndarray:
        """W as a dense matrix (for spectra and inspection; the simulation uses the FFT)."""
        k = self.kernel
        i = np.arange(self.N)
        return k[(i[:, None] - i[None, :]) % self.N]

    def initial_rates(self) -> np.ndarray:
        s0 = self.sigma * self.N
        x = np.arange(self.N)
        p = np.broadcast_to(self.participation.gains(self.N, np.zeros(1)), (self.N, 1))[:, 0]
        return p * np.exp(-(wrap(x - s0, self.N) ** 2) / (2 * s0**2))

    def simulate(self) -> SampledSequence:
        n_burn = int(round(self.burn_in / self.dt))
        n_steps = n_burn + int(round(self.duration / self.dt))
        t = (np.arange(n_steps + 1) - n_burn) * self.dt
        S, D = self.gain(t), self.drive(t)
        P = np.broadcast_to(self.participation.gains(self.N, t), (self.N, t.size))
        k_hat = np.fft.rfft(self.kernel)
        frac = P.mean(0) if self.rescale else np.ones(t.size)
        a = self.dt / self.tau
        r = self.initial_rates()
        I = r.copy()
        rec = [r.copy()] if n_burn == 0 else []
        for j in range(n_steps):
            Wr = np.fft.irfft(k_hat * np.fft.rfft(r), self.N) / max(frac[j], 1e-12)
            I += a * (-I + P[:, j] * S[j] * (Wr + D[j]))
            if self.bounded_state:
                np.clip(I, 0.0, 1.0, out=I)
            r = np.clip(I, 0.0, 1.0)
            if j + 1 >= n_burn and (j + 1 - n_burn) % self.record_every == 0:
                rec.append(r.copy())
        return SampledSequence(np.stack(rec, axis=1), self.dt * self.record_every)


@SEQUENCES.register("lehr_network")
def lehr_network(**params) -> SampledSequence:
    """Simulate :class:`LehrRingNetwork` with these parameters and return its rates as a sequence."""
    return LehrRingNetwork(**params).simulate()

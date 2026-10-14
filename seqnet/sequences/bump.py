"""Parametric travelling bumps: Braun et al. targets and the Lehr et al. (2024) Methods sequences as one family.

    f_n(t) = g_n(t) [ A(t) phi(c(t) - x_n; w0 s(t)) + b ],     c(t) = c0 + int_0^t v(t') dt',     x_n = n,

on a ring of N neurons (units: neurons and seconds).  The parts are independent and swappable:

* shape phi (:mod:`.profiles`): Braun's wrapped Gaussian, Lehr's raised cosine psi_k, or the cosine-rectangle morph;
* speed v(t) (:mod:`.timecourses`): constant, or a gain-modulated course (Lehr 2024 Fig. 3-4: S(t) scales speed);
* amplitude A(t) and width scale s(t): transient shape control (Lehr 2024 Fig. 2, S1-S2);
* participation g_n(t) (:mod:`.participation`): silencing by clustered inhibition and switching between ensembles
  (Lehr 2024 eq. 7, Fig. 1h-k, 5).

Braun's target (Methods eq. 4-5) is the member with a Gaussian of width sigma N/T neurons, speed N/T, amplitude a and
offset b: neuron n peaks at t = n T/N, as in f_n(t) = a g(t - n T/N) + b.  Silenced neurons are silent (the offset is
inside the gain).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import SEQUENCES, Sequence, StationaryWave
from .participation import PARTICIPATION, AllActive, Participation, RandomSilencing
from .profiles import PROFILES, BumpProfile, MorphedCosine, RaisedCosine, WrappedGaussian
from .timecourses import Constant, Scaled, TimeCourse, as_timecourse

__all__ = ["TravelingBump", "braun_sequence", "lehr_bump", "BRAUN_PERIOD"]

BRAUN_PERIOD = 8.5  # s, Braun et al. synthetic target period: the reference training timescale


@SEQUENCES.register("traveling_bump")
@dataclass(frozen=True)
class TravelingBump(Sequence):
    """General parametric bump on a ring (see module docstring).  Nested parts accept instances or dict specs."""

    N: int
    profile: BumpProfile
    width: float  # neurons, at width scale 1
    speed: TimeCourse  # neurons / s
    amplitude: TimeCourse = field(default_factory=lambda: Constant(1.0))
    width_scale: TimeCourse = field(default_factory=lambda: Constant(1.0))
    offset: float = 0.0
    participation: Participation = field(default_factory=AllActive)
    phase0: float = 0.0  # bump centre at t = 0 (neurons)

    def __post_init__(self):
        object.__setattr__(self, "profile", PROFILES.coerce(self.profile))
        for name in ("speed", "amplitude", "width_scale"):
            object.__setattr__(self, name, as_timecourse(getattr(self, name)))
        object.__setattr__(self, "participation", PARTICIPATION.coerce(self.participation))

    # ------------------------------------------------------------------ evaluation
    def _parts(self, t, neurons):
        t = np.asarray(t, dtype=float)
        x = np.arange(self.N, dtype=float)[neurons]
        delta = (self.phase0 + self.speed.integral(t))[None, :] - x[:, None]
        w = self.width * self.width_scale(t)[None, :]
        g = self.participation.gains(self.N, t)[neurons]
        return t, delta, w, g

    def _rates(self, t, neurons=slice(None)) -> np.ndarray:
        t, delta, w, g = self._parts(t, neurons)
        return g * (self.amplitude(t)[None, :] * self.profile.value(delta, w, self.N) + self.offset)

    def rates(self, t):
        return self._rates(t)

    def rates_dt(self, t):
        """Exact derivative (participation switches are jumps; their derivative is taken as zero)."""
        t, delta, w, g = self._parts(t, slice(None))
        p, N = self.profile, self.N
        A, dA = self.amplitude(t)[None, :], self.amplitude.derivative(t)[None, :]
        v = self.speed(t)[None, :]
        dw = self.width * self.width_scale.derivative(t)[None, :]
        return g * (dA * p.value(delta, w, N) + A * (p.d_delta(delta, w, N) * v + p.d_width(delta, w, N) * dw))

    # ------------------------------------------------------------------ stationary view
    @property
    def is_stationary(self) -> bool:
        return (self.speed.is_constant and self.amplitude.is_constant and self.width_scale.is_constant
                and self.participation.is_full and self.speed(np.zeros(1))[0] != 0)

    def stationary(self) -> StationaryWave:
        """The scalar-bump view f_n(t) = f0(t - n T/N) (Braun Suppl. Sec. 2); ValueError if the bump is modulated."""
        if not self.is_stationary:
            raise ValueError("only a constant-speed, unmodulated, fully active bump is a stationary wave")
        v = float(self.speed(np.zeros(1))[0])
        A = float(self.amplitude(np.zeros(1))[0])
        w = self.width * float(self.width_scale(np.zeros(1))[0])
        period = self.N / abs(v)

        def coeffs(k):
            # f0(t) = A phi(c0 + v t) + b; substituting d = c0 + v t gives the ring series at sign(v) k
            k = np.asarray(k)
            ck = A * self.profile.fourier_coeffs(np.sign(v) * k, w, self.N)
            return ck * np.exp(2j * np.pi * k * self.phase0 / (v * period)) + np.where(k == 0, self.offset, 0.0)

        return StationaryWave(N=self.N, period=period, profile=lambda t: self._rates(t, [0])[0], coeffs=coeffs)


@SEQUENCES.register("braun")
def braun_sequence(N: int = 501, T: float = BRAUN_PERIOD, amp: float = 0.5, width: float = 1.7, offset: float = 0.0,
                   n_images: int = 6) -> TravelingBump:
    """Braun et al. Methods eq. 4-5: f_n(t) = a g_{sigma,T}(t - n T/N) + b, sigma = ``width`` in seconds."""
    return TravelingBump(N=N, profile=WrappedGaussian(n_images), width=width * N / T, speed=Constant(N / T),
                         amplitude=Constant(amp), offset=offset)


@SEQUENCES.register("lehr_bump")
def lehr_bump(N: int = 1000, k: float = 1.0, eps: float | None = None, amplitude=1.0, period: float = BRAUN_PERIOD,
              gain=1.0, width_scale=1.0, p_inh: float = 0.0, seed: int = 0, participation=None, offset: float = 0.0,
              phase0: float = 0.0) -> TravelingBump:
    """Lehr, Kumar & Tetzlaff (2024) Methods eq. 1-7 in their parameters.

    k: width of the bump as the support N/k (k >= 1); eps: shape, None for the raised cosine psi_k (eq. 5), else
    Phi_{k,eps} (eq. 3; 0 = rectangle); amplitude A (number or time course); p_inh, seed: neurons silenced by P
    (eq. 7), unless an explicit ``participation`` (e.g. switching ensembles) is given.

    Time: the bump moves at dc/dt = (N / period) S(t).  ``period`` (s) is the cycle duration at gain 1, by default
    Braun's T = 8.5 s so that the family shares one training timescale; ``gain`` S(t) is the speed control of Lehr
    Fig. 3-4 (number or time course).
    """
    if k < 1:
        raise ValueError("k >= 1 (the bump support N/k cannot exceed the ring)")
    profile = RaisedCosine() if eps is None else MorphedCosine(float(eps))
    part = participation if participation is not None else RandomSilencing(p_inh, seed)
    return TravelingBump(N=N, profile=profile, width=N / k, speed=Scaled(gain, N / period), amplitude=amplitude,
                         width_scale=width_scale, offset=offset, participation=part, phase0=phase0)

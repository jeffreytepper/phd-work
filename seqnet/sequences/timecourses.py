"""Scalar control signals of time: gain S(t), bump amplitude A(t), width scale, transient drive E(t) - I(t).

Each :class:`TimeCourse` gives its value, derivative and integral from 0, all in closed form, so that a bump driven by
a speed signal has an exact position c(t) = c0 + int_0^t v(t') dt' and an exact time derivative.

The shapes are the time-dependent gain modulations of Lehr, Kumar & Tetzlaff (2024, Fig. 4): step, exponential decay
and growth, sigmoid, concave and convex parabola, sinusoid; plus the Gaussian pulse used for transient top-down and
E - I inputs (Lehr et al. 2025 eLife, Suppl. eq. S25; 2024 Fig. 2).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from .registry import Registry

__all__ = ["TimeCourse", "TIMECOURSES", "as_timecourse", "Constant", "Step", "Exponential", "Sigmoid", "Parabola",
           "Sinusoid", "GaussianPulse"]


class TimeCourse(ABC):
    """A scalar function of time with closed-form derivative and integral."""

    @abstractmethod
    def __call__(self, t: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def derivative(self, t: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def integral(self, t: np.ndarray) -> np.ndarray:
        """int_0^t s(t') dt'."""

    @property
    def is_constant(self) -> bool:
        return False


TIMECOURSES: Registry[TimeCourse] = Registry("time course", TimeCourse)


def as_timecourse(x) -> TimeCourse:
    """Numbers become :class:`Constant`; dict specs are built; time courses pass through."""
    if isinstance(x, (int, float, np.floating, np.integer)):
        return Constant(float(x))
    return TIMECOURSES.coerce(x)


def _t(t) -> np.ndarray:
    return np.asarray(t, dtype=float)


@TIMECOURSES.register("constant")
@dataclass(frozen=True)
class Constant(TimeCourse):
    value: float = 1.0

    def __call__(self, t):
        return np.full_like(_t(t), self.value)

    def derivative(self, t):
        return np.zeros_like(_t(t))

    def integral(self, t):
        return self.value * _t(t)

    @property
    def is_constant(self) -> bool:
        return True


@TIMECOURSES.register("step")
@dataclass(frozen=True)
class Step(TimeCourse):
    """``before`` for t < t_step, ``after`` from t_step on (Lehr 2024 Fig. 4a)."""

    before: float = 1.0
    after: float = 2.0
    t_step: float = 0.0

    def __call__(self, t):
        return np.where(_t(t) < self.t_step, self.before, self.after)

    def derivative(self, t):
        return np.zeros_like(_t(t))  # zero almost everywhere

    def integral(self, t):
        t = _t(t)
        # int_0^t = before * (min(t, ts) - min(0, ts)) + after * (max(t, ts) - max(0, ts))
        ts = self.t_step
        return self.before * (np.minimum(t, ts) - min(0.0, ts)) + self.after * (np.maximum(t, ts) - max(0.0, ts))


@TIMECOURSES.register("exponential")
@dataclass(frozen=True)
class Exponential(TimeCourse):
    """baseline + scale * exp(rate * t): rate < 0 decays to baseline, rate > 0 grows (Lehr 2024 Fig. 4b-d)."""

    baseline: float = 0.0
    scale: float = 1.0
    rate: float = -1.0

    def __call__(self, t):
        return self.baseline + self.scale * np.exp(self.rate * _t(t))

    def derivative(self, t):
        return self.scale * self.rate * np.exp(self.rate * _t(t))

    def integral(self, t):
        t = _t(t)
        if self.rate == 0:
            return (self.baseline + self.scale) * t
        return self.baseline * t + self.scale * np.expm1(self.rate * t) / self.rate


@TIMECOURSES.register("sigmoid")
@dataclass(frozen=True)
class Sigmoid(TimeCourse):
    """low + (high - low) / (1 + exp(-(t - t_mid) / width)) (Lehr 2024 Fig. 4e)."""

    low: float = 1.0
    high: float = 2.0
    t_mid: float = 0.0
    width: float = 1.0

    def __call__(self, t):
        return self.low + (self.high - self.low) / (1 + np.exp(-(_t(t) - self.t_mid) / self.width))

    def derivative(self, t):
        z = (_t(t) - self.t_mid) / self.width
        s = 1 / (1 + np.exp(-z))
        return (self.high - self.low) * s * (1 - s) / self.width

    def integral(self, t):
        t = _t(t)
        # int sigma(z) dz = softplus(z); softplus(z) = logaddexp(0, z)
        sp = np.logaddexp(0.0, (t - self.t_mid) / self.width) - np.logaddexp(0.0, -self.t_mid / self.width)
        return self.low * t + (self.high - self.low) * self.width * sp


@TIMECOURSES.register("parabola")
@dataclass(frozen=True)
class Parabola(TimeCourse):
    """vertex + curvature * (t - t_vertex)^2: curvature < 0 concave, > 0 convex (Lehr 2024 Fig. 4f, g)."""

    vertex: float = 1.0
    curvature: float = -1.0
    t_vertex: float = 0.0

    def __call__(self, t):
        return self.vertex + self.curvature * (_t(t) - self.t_vertex) ** 2

    def derivative(self, t):
        return 2 * self.curvature * (_t(t) - self.t_vertex)

    def integral(self, t):
        t = _t(t)
        return self.vertex * t + self.curvature / 3 * ((t - self.t_vertex) ** 3 - (-self.t_vertex) ** 3)


@TIMECOURSES.register("sinusoid")
@dataclass(frozen=True)
class Sinusoid(TimeCourse):
    """mean + amplitude * sin(2 pi t / period + phase) (Lehr 2024 Fig. 4h)."""

    mean: float = 1.0
    amplitude: float = 0.5
    period: float = 1.0
    phase: float = 0.0

    def __call__(self, t):
        return self.mean + self.amplitude * np.sin(2 * np.pi * _t(t) / self.period + self.phase)

    def derivative(self, t):
        w = 2 * np.pi / self.period
        return self.amplitude * w * np.cos(w * _t(t) + self.phase)

    def integral(self, t):
        t = _t(t)
        w = 2 * np.pi / self.period
        return self.mean * t - self.amplitude / w * (np.cos(w * t + self.phase) - np.cos(self.phase))


@TIMECOURSES.register("gaussian_pulse")
@dataclass(frozen=True)
class GaussianPulse(TimeCourse):
    """baseline + height * exp(-(t - t_peak)^2 / (2 width^2)): transient input (Lehr et al. 2025, eq. S25)."""

    baseline: float = 0.0
    height: float = 1.0
    t_peak: float = 0.0
    width: float = 1.0

    def __call__(self, t):
        return self.baseline + self.height * np.exp(-((_t(t) - self.t_peak) ** 2) / (2 * self.width**2))

    def derivative(self, t):
        u = _t(t) - self.t_peak
        return -self.height * u / self.width**2 * np.exp(-(u**2) / (2 * self.width**2))

    def integral(self, t):
        t = _t(t)
        c = self.width * np.sqrt(np.pi / 2)
        r = np.sqrt(2) * self.width
        return self.baseline * t + self.height * c * (erf((t - self.t_peak) / r) - erf(-self.t_peak / r))

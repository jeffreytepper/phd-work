"""Bump shapes on the ring: the activity of a neuron at signed ring displacement delta from the bump centre.

Units are neurons: a ring of ``period`` = N neurons, displacement ``delta`` (any real; reduced to [-N/2, N/2)), and a
shape-specific ``width``.  Every profile gives its value and its partial derivatives in delta and width, which the
travelling bump combines into exact time derivatives (the y* = x* + tau dx*/dt training target needs them).

* :class:`WrappedGaussian`: Braun et al. Methods eq. 5 (width = standard deviation).
* :class:`RaisedCosine`: Lehr, Kumar & Tetzlaff (2024) eq. 1, psi_k (width = support N/k).
* :class:`MorphedCosine`: Lehr et al. (2024) eq. 2-5, Phi_{k,eps}, morphing psi_k (eps -> inf) into a rectangle of
  the same support (eps -> 0).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .registry import Registry

__all__ = ["BumpProfile", "PROFILES", "wrap", "WrappedGaussian", "RaisedCosine", "MorphedCosine"]


def wrap(delta: np.ndarray, period: float) -> np.ndarray:
    """Reduce displacements to [-period/2, period/2)."""
    return (np.asarray(delta, dtype=float) + period / 2) % period - period / 2


class BumpProfile(ABC):
    """phi(delta; width) on a ring of circumference ``period``, peak value 1 at delta = 0."""

    @abstractmethod
    def value(self, delta: np.ndarray, width: np.ndarray | float, period: float) -> np.ndarray: ...

    @abstractmethod
    def d_delta(self, delta: np.ndarray, width: np.ndarray | float, period: float) -> np.ndarray: ...

    @abstractmethod
    def d_width(self, delta: np.ndarray, width: np.ndarray | float, period: float) -> np.ndarray: ...

    def fourier_coeffs(self, k: np.ndarray, width: float, period: float) -> np.ndarray:
        """c_k = (1/period) int phi(delta) e^{-2 pi i k delta / period} d delta; analytic where a subclass knows it."""
        raise NotImplementedError(f"{type(self).__name__} has no closed-form Fourier series")


PROFILES: Registry[BumpProfile] = Registry("bump profile", BumpProfile)


@PROFILES.register("wrapped_gaussian")
@dataclass(frozen=True)
class WrappedGaussian(BumpProfile):
    """sum_{|m| <= n_images} exp(-(delta - m period)^2 / (2 width^2)) (Braun Methods eq. 5)."""

    n_images: int = 6

    def _images(self, delta, period):
        m = np.arange(-self.n_images, self.n_images + 1)
        return wrap(delta, period)[..., None] - m * period

    def value(self, delta, width, period):
        x = self._images(delta, period)
        w = np.asarray(width, dtype=float)[..., None]
        return np.exp(-(x**2) / (2 * w**2)).sum(-1)

    def d_delta(self, delta, width, period):
        x = self._images(delta, period)
        w = np.asarray(width, dtype=float)[..., None]
        return (-(x / w**2) * np.exp(-(x**2) / (2 * w**2))).sum(-1)

    def d_width(self, delta, width, period):
        x = self._images(delta, period)
        w = np.asarray(width, dtype=float)[..., None]
        return ((x**2 / w**3) * np.exp(-(x**2) / (2 * w**2))).sum(-1)

    def fourier_coeffs(self, k, width, period):
        """width sqrt(2 pi) / period exp(-(2 pi k width / period)^2 / 2) (Braun Suppl. eq. 2-3, infinite image sum)."""
        k = np.asarray(k, dtype=float)
        return width * np.sqrt(2 * np.pi) / period * np.exp(-((2 * np.pi * k * width / period) ** 2) / 2)


@PROFILES.register("raised_cosine")
@dataclass(frozen=True)
class RaisedCosine(BumpProfile):
    """psi(delta) = (1 + cos(2 pi delta / width)) / 2 for |delta| < width / 2, else 0 (Lehr et al. 2024 eq. 1).

    ``width`` is the support, N/k in the paper's notation; width = N gives a pure sinusoid around the ring.
    """

    def value(self, delta, width, period):
        d = wrap(delta, period)
        w = np.asarray(width, dtype=float)
        inside = np.abs(d) < w / 2
        return np.where(inside, 0.5 * (1 + np.cos(2 * np.pi * d / w)), 0.0)

    def d_delta(self, delta, width, period):
        d = wrap(delta, period)
        w = np.asarray(width, dtype=float)
        inside = np.abs(d) < w / 2
        return np.where(inside, -0.5 * (2 * np.pi / w) * np.sin(2 * np.pi * d / w), 0.0)

    def d_width(self, delta, width, period):
        d = wrap(delta, period)
        w = np.asarray(width, dtype=float)
        inside = np.abs(d) < w / 2
        return np.where(inside, 0.5 * np.sin(2 * np.pi * d / w) * 2 * np.pi * d / w**2, 0.0)

    def fourier_coeffs(self, k, width, period):
        """Closed form of the raised-cosine lobe of support ``width`` on a ring of ``period``."""
        k = np.asarray(k, dtype=float)
        w, P = float(width), float(period)
        x = k * w / P  # frequency in units of the lobe's own period
        # (1/P) int_{-w/2}^{w/2} (1 + cos(2 pi d / w))/2 e^{-2 pi i k d / P} dd = (w / 2P) [sinc(x) + (sinc(x-1) + sinc(x+1))/2]
        return w / (2 * P) * (np.sinc(x) + 0.5 * (np.sinc(x - 1) + np.sinc(x + 1)))


@PROFILES.register("morphed_cosine")
@dataclass(frozen=True)
class MorphedCosine(BumpProfile):
    """Phi_eps = psi mu(eps) / ||psi mu(eps)||_inf with mu = (1 + eps) / (psi + eps) (Lehr et al. 2024 eq. 2-3).

    The supremum is reached at psi = 1, so Phi_eps = psi (1 + eps) / (psi + eps) on the support: eps -> inf gives the
    raised cosine (eq. 5), eps -> 0 the rectangle of the same support (eq. 4).  At eps = 0 the delta and width
    derivatives are zero almost everywhere (the edges are jumps), so a y* target is not defined there.
    """

    eps: float = 1.0

    def __post_init__(self):
        if self.eps < 0:
            raise ValueError("eps must be >= 0")

    _psi = RaisedCosine()

    def _dPhi_dpsi(self, psi):
        e = self.eps
        if np.isinf(e):
            return np.ones_like(psi)
        if e == 0:
            return np.zeros_like(psi)
        return (1 + e) * e / (psi + e) ** 2

    def value(self, delta, width, period):
        psi = self._psi.value(delta, width, period)
        e = self.eps
        if np.isinf(e):
            return psi
        if e == 0:
            return (psi > 0).astype(float)
        return psi * (1 + e) / (psi + e)

    def d_delta(self, delta, width, period):
        psi = self._psi.value(delta, width, period)
        return self._dPhi_dpsi(psi) * self._psi.d_delta(delta, width, period)

    def d_width(self, delta, width, period):
        psi = self._psi.value(delta, width, period)
        return self._dPhi_dpsi(psi) * self._psi.d_width(delta, width, period)

"""Kernel families (Thesis §4.5, App. D) and kernels built from a spectral allocation (Thesis §7).

A :class:`Kernel` stores its values on the ring grid, k(r_i), i = 0..N-1, with the bump centre at
r = 0.  All spectral quantities use the D10 operational normalization of :mod:`.fourier`
(a_m = |K_m|^2 = N |k_hat_m|^2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from .fourier import (
    alloc_from_kernel,
    circulant_eigs,
    kernel_from_half_spectrum,
    mode_multiplicity,
    trig_interpolate,
)
from .ring import Ring

__all__ = ["Kernel", "analytic_spectrum_ratio", "FAMILIES"]

FAMILIES = ("point", "gaussian", "gabor", "box", "bspline")

SignPolicy = Literal["positive", "alternating", "explicit"]
OffsetPolicy = Literal["none", "positive"]


def _normalize_integral(k: np.ndarray, dr: float) -> np.ndarray:
    """Normalize so that sum_i k(r_i) dr = 1 (Thesis §2.1)."""
    s = k.sum() * dr
    if abs(s) < 1e-300:
        raise ValueError("kernel integrates to zero; cannot normalize")
    return k / s


def _gaussian_periodic(u: np.ndarray, sigma: float, n_images: int = 6) -> np.ndarray:
    n = np.arange(-n_images, n_images + 1)
    return np.exp(-((u[..., None] + 2 * np.pi * n) ** 2) / (2 * sigma**2)).sum(-1)


def _cardinal_bspline(u: np.ndarray, p: int, h: float) -> np.ndarray:
    """Centred cardinal B-spline of degree p with knot spacing h (support (p+1) h), App. D.3."""
    knots = h * (np.arange(p + 2) - (p + 1) / 2)
    b = BSpline.basis_element(knots, extrapolate=False)
    out = np.nan_to_num(b(u), nan=0.0)
    return out


@dataclass(frozen=True)
class Kernel:
    """A real kernel on the ring, k(r_i) for i = 0..N-1 (Thesis eq. 2.1).

    Attributes
    ----------
    values : array (N,)
        Kernel samples k(r_i), bump centre at r_0 = 0.
    name : str
        Family name or ``"alloc"`` for kernels built from an allocation.
    params : dict
        Construction parameters (recorded for reproducibility).
    """

    values: np.ndarray
    name: str = "custom"
    params: dict | None = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_family(cls, name: str, N: int, normalize: bool = True, **params) -> "Kernel":
        """Construct one of the App. D families on an N-neuron ring.

        Families and parameters (Thesis §4.5, App. D):
          * ``point``: discrete delta, k(r_0) = 1/dr (flat spectrum, eq. 4.9).
          * ``gaussian``: ``sigma`` (rad); periodic Gaussian, spectrum ~ exp(-sigma^2 m^2 / 2) (eq. 4.10).
          * ``gabor``: ``sigma``, ``xi0``; Gaussian envelope times cos(xi0 r) (eq. 4.11).  Normalized so
            the *envelope* integrates to one (a Gabor's own integral can vanish).
          * ``box``: ``R`` (rad); 1/R on |r| < R/2 (eq. 4.12).
          * ``bspline``: ``p`` (degree), ``h`` (knot spacing, rad); support (p+1) h (eq. 4.13).
        """
        ring = Ring(N)
        u = ring.centred_positions
        dr = ring.dr
        if name == "point":
            k = np.zeros(N)
            k[0] = 1.0 / dr
        elif name == "gaussian":
            k = _gaussian_periodic(u, params["sigma"])
            if normalize:
                k = _normalize_integral(k, dr)
        elif name == "gabor":
            env = _gaussian_periodic(u, params["sigma"])
            if normalize:
                env = env / (env.sum() * dr)
            k = env * np.cos(params["xi0"] * u)
        elif name == "box":
            R = params["R"]
            k = (np.abs(u) < R / 2 - 1e-12).astype(float)
            # include boundary points with weight 1/2 when they fall exactly on the edge
            k += 0.5 * (np.abs(np.abs(u) - R / 2) <= 1e-12)
            k = k / R
            if normalize:
                k = _normalize_integral(k, dr)
        elif name == "bspline":
            p, h = int(params["p"]), float(params["h"])
            k = _cardinal_bspline(u, p, h)
            if normalize:
                k = _normalize_integral(k, dr)
        else:
            raise ValueError(f"unknown kernel family {name!r}; choose from {FAMILIES}")
        return cls(values=np.asarray(k, dtype=float), name=name, params=dict(params))

    @classmethod
    def from_alloc(
        cls,
        a: np.ndarray,
        N: int | None = None,
        sign_policy: SignPolicy = "positive",
        signs: np.ndarray | None = None,
        offset_policy: OffsetPolicy = "none",
        r_min: float = 0.0,
    ) -> "Kernel":
        """Real symmetric kernel with operational power a_m on m = 0..floor(N/2) (Thesis eq. 7.1, 7.13).

        The thesis fixes magnitudes only (A6: real coefficients, sign free).  ``sign_policy``:
          * ``positive``: all K_m >= 0 (maximal central peak; default),
          * ``alternating``: K_m = (-1)^m sqrt(a_m),
          * ``explicit``: use ``signs`` (array of +-1 on m >= 0).
        ``offset_policy='positive'`` adds a constant so that min k = r_min (this changes a_0 only).
        """
        a = np.asarray(a, dtype=float)
        if N is None:
            N = 2 * (len(a) - 1)  # assume even N
        if len(a) != N // 2 + 1:
            raise ValueError("allocation must have floor(N/2)+1 entries")
        if np.any(a < -1e-12):
            raise ValueError("allocation must be non-negative")
        mag = np.sqrt(np.clip(a, 0, None))
        m = np.arange(len(a))
        if sign_policy == "positive":
            s = np.ones_like(mag)
        elif sign_policy == "alternating":
            s = (-1.0) ** m
        elif sign_policy == "explicit":
            if signs is None:
                raise ValueError("explicit sign policy requires `signs`")
            s = np.sign(np.asarray(signs, dtype=float))
        else:
            raise ValueError(sign_policy)
        k = kernel_from_half_spectrum(s * mag, N)
        params = {"sign_policy": sign_policy, "offset_policy": offset_policy}
        if offset_policy == "positive":
            shift = r_min - k.min()
            if shift > 0:
                k = k + shift
            params["offset"] = float(max(shift, 0.0))
        elif offset_policy != "none":
            raise ValueError(offset_policy)
        return cls(values=k, name="alloc", params=params)

    # ------------------------------------------------------------------ properties
    @property
    def N(self) -> int:
        return int(self.values.size)

    def spectrum(self) -> np.ndarray:
        """Unnormalized DFT K_m on all N bins (Thesis eq. 4.1 times sqrt(N); D10)."""
        return circulant_eigs(self.values)

    def alloc(self) -> np.ndarray:
        """Operational power a_m = |K_m|^2 on m = 0..floor(N/2) (Thesis eq. 7.1; D10)."""
        return alloc_from_kernel(self.values, half=True)

    def energy(self) -> float:
        """Spectral energy K0 = sum over signed modes a_m = N sum_i k(r_i)^2 (Thesis eq. 7.3; D10)."""
        a = self.alloc()
        return float(np.sum(mode_multiplicity(self.N) * a))

    def effective_rank(self, eps: float = 1e-2, reference: Literal["dc", "max"] = "dc") -> int:
        """Effective rank r_k(eps) = #{m : |k_hat_m| / |k_hat_0| > eps} over all N signed modes (Thesis eq. 4.8).

        ``reference='max'`` uses max_m |k_hat_m| instead of the DC coefficient; needed for kernels
        (e.g. Gabor) whose DC coefficient is near zero.
        """
        K = np.abs(self.spectrum())
        ref = K[0] if reference == "dc" else K.max()
        if ref <= 0:
            raise ValueError("reference coefficient is zero; use reference='max'")
        return int(np.sum(K / ref > eps))

    def autocorrelation(self) -> np.ndarray:
        """Discrete autocorrelation R_k(j dr) = sum_i k(r_i) k(r_i - j dr), j = 0..N-1 (Thesis eq. 5.1)."""
        K = self.spectrum()
        return np.real(np.fft.ifft(np.abs(K) ** 2))

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Evaluate k(u) at arbitrary displacements via the trigonometric interpolant."""
        return trig_interpolate(self.values, u)

    def is_symmetric(self, tol: float = 1e-10) -> bool:
        k = self.values
        return bool(np.allclose(k, np.roll(k[::-1], 1), atol=tol * max(1.0, np.abs(k).max())))


def analytic_spectrum_ratio(name: str, m: np.ndarray, **params) -> np.ndarray:
    """|k_hat_m| / |k_hat_0| from the continuous-domain spectra of Thesis eq. 4.9-4.13 (used by TT5).

    These ignore periodization/aliasing on the discrete ring, so they are accurate for kernels much
    narrower than the ring and much wider than one neuron spacing.
    """
    m = np.asarray(m, dtype=float)
    if name == "point":
        return np.ones_like(m)
    if name == "gaussian":
        return np.exp(-params["sigma"] ** 2 * m**2 / 2)
    if name == "gabor":
        s, xi = params["sigma"], params["xi0"]
        g = lambda x: np.exp(-(s**2) * x**2 / 2)  # noqa: E731
        return np.abs(g(m - xi) + g(m + xi)) / np.abs(2 * g(xi))
    if name == "box":
        R = params["R"]
        return np.abs(np.sinc(m * R / 2 / np.pi))  # np.sinc(x) = sin(pi x)/(pi x)
    if name == "bspline":
        p, h = int(params["p"]), params["h"]
        return np.abs(np.sinc(m * h / 2 / np.pi)) ** (p + 1)
    raise ValueError(name)


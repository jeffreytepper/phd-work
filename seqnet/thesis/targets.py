"""Target classes specified by their power spectral density S(m) (Thesis §6.2, A4, App. I; Fig. 2).

Normalization
-------------
A target y(c) on an N-neuron ring is written y(c) = N^{-1/2} sum_{m in Z} y_hat_m e^{i m c}, so that
for a target band-limited below N/2 the unitary DFT of its N grid samples (Thesis eq. 4.1) returns
y_hat_m, and S(m) = E|y_hat_m|^2 (Thesis eq. 6.3).  This is the normalization in which the per-mode
error E(m) = lambda^2 S(m) / (a_m + lambda)^2 (eq. 6.6) holds with the D10 allocation a_m and the
sum-of-squares loss of eq. 3.10.

S is stored on m = 0..M_max (M_max may exceed N/2: content beyond the network Nyquist enters the
network-floor term, eq. 6.9) and is symmetric, S(-m) = S(m).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

__all__ = ["TargetClass", "fixed_target"]

PhaseMode = Literal["phase", "gaussian"]


@dataclass(frozen=True)
class TargetClass:
    """A stationary target class with PSD S(m) on m = 0..M_max (Thesis eq. 6.3, A4).

    ``psd`` holds S(m).  ``sample`` draws a target with random phases; ``amplitude='phase'`` (default)
    keeps |y_hat_m|^2 = S(m) exactly (random phases only), ``'gaussian'`` draws complex Gaussian
    coefficients with E|y_hat_m|^2 = S(m).
    """

    psd: np.ndarray
    name: str = "custom"
    params: dict = field(default_factory=dict)
    fixed_coeffs: np.ndarray | None = None  # set for deterministic targets (``fixed``)

    @property
    def M_max(self) -> int:
        return int(len(self.psd) - 1)

    def coeffs(self, rng: np.random.Generator | None = None, amplitude: PhaseMode = "phase") -> np.ndarray:
        """Fourier coefficients y_hat_m on m = 0..M_max (m >= 0; y_hat_{-m} = conj(y_hat_m))."""
        if self.fixed_coeffs is not None:
            return np.asarray(self.fixed_coeffs, dtype=complex)
        if rng is None:
            raise ValueError("random target class needs an rng")
        S = np.asarray(self.psd, dtype=float)
        if amplitude == "phase":
            mag = np.sqrt(S)
        elif amplitude == "gaussian":
            mag = np.sqrt(S) * np.abs(rng.normal(size=S.shape) + 1j * rng.normal(size=S.shape)) / np.sqrt(2)
        else:
            raise ValueError(amplitude)
        ph = np.exp(2j * np.pi * rng.random(S.shape))
        y = mag * ph
        # the DC coefficient of a real signal is real; keep its power with a random sign
        y[0] = mag[0] * (1.0 if rng.random() < 0.5 else -1.0)
        return y

    def evaluate(self, coeffs: np.ndarray, N: int, c: np.ndarray) -> np.ndarray:
        """y(c) = N^{-1/2} sum_m y_hat_m e^{i m c} for real targets, at positions c."""
        c = np.asarray(c, dtype=float)
        m = np.arange(len(coeffs))
        ph = np.exp(1j * np.multiply.outer(c, m))
        terms = ph * coeffs
        # real signal: y = N^{-1/2} [y_0 + 2 Re sum_{m>0} y_m e^{imc}]
        val = np.real(terms[..., 0]) + 2 * np.real(terms[..., 1:].sum(-1))
        return val / np.sqrt(N)

    def sample(
        self, rng: np.random.Generator, N: int, c: np.ndarray, amplitude: PhaseMode = "phase"
    ) -> np.ndarray:
        """Draw one target and evaluate it at positions c (Thesis A4: random phases)."""
        return self.evaluate(self.coeffs(rng, amplitude), N, c)

    # ------------------------------------------------------------------ App. I classes
    @classmethod
    def from_function(cls, S: Callable[[np.ndarray], np.ndarray], M_max: int, name: str = "custom", **params):
        m = np.arange(M_max + 1)
        return cls(psd=np.asarray(S(m), dtype=float), name=name, params=params)

    @classmethod
    def constant(cls, M_max: int, S0: float = 1.0) -> "TargetClass":
        """S(m) = S0 delta_{m,0} (Thesis eq. I.1)."""
        S = np.zeros(M_max + 1)
        S[0] = S0
        return cls(psd=S, name="constant", params={"S0": S0})

    @classmethod
    def single_mode(cls, M_max: int, m0: int, S0: float = 1.0) -> "TargetClass":
        """S(m) = S0 (delta_{m,m0} + delta_{m,-m0}) (Thesis eq. I.3)."""
        S = np.zeros(M_max + 1)
        S[m0] = S0
        return cls(psd=S, name="single_mode", params={"m0": m0, "S0": S0})

    @classmethod
    def flat(cls, M_max: int, B: int, S0: float = 1.0) -> "TargetClass":
        """S(m) = S0 for |m| <= B, else 0 (Thesis eq. I.6)."""
        m = np.arange(M_max + 1)
        return cls(psd=np.where(m <= B, S0, 0.0), name="flat", params={"B": B, "S0": S0})

    @classmethod
    def gaussian_psd(cls, M_max: int, beta: float, S0: float = 1.0) -> "TargetClass":
        """S(m) = S0 exp(-beta m^2) (Thesis eq. I.11)."""
        m = np.arange(M_max + 1)
        return cls(psd=S0 * np.exp(-beta * m**2), name="gaussian_psd", params={"beta": beta, "S0": S0})

    @classmethod
    def polynomial(cls, M_max: int, s: float, S0: float = 1.0, dc: float | None = None) -> "TargetClass":
        """S(m) = S0 |m|^{-2s} (Thesis eq. I.14).  S(0) is undefined in the thesis; default S(0) = S0."""
        m = np.arange(M_max + 1).astype(float)
        S = S0 * np.maximum(m, 1.0) ** (-2 * s)
        if dc is not None:
            S[0] = dc
        return cls(psd=S, name="polynomial", params={"s": s, "S0": S0, "dc": dc})

    @classmethod
    def narrowband(cls, M_max: int, xi0: float, ell: float, S0: float = 1.0) -> "TargetClass":
        """S(m) = S0 [exp(-(m-xi0)^2 / 2 l^2) + exp(-(m+xi0)^2 / 2 l^2)] (Thesis eq. I.17)."""
        m = np.arange(M_max + 1).astype(float)
        S = S0 * (np.exp(-((m - xi0) ** 2) / (2 * ell**2)) + np.exp(-((m + xi0) ** 2) / (2 * ell**2)))
        return cls(psd=S, name="narrowband", params={"xi0": xi0, "ell": ell, "S0": S0})


def fixed_target(y: np.ndarray, name: str = "fixed", M_max: int | None = None) -> TargetClass:
    """Wrap one deterministic target, given on an N-point grid, as a class with S(m) = |y_hat_m|^2.

    The coefficients are the unitary DFT of y (Thesis eq. 4.1), m = 0..floor(N/2).  For even N the
    Nyquist coefficient is halved in ``evaluate``'s 2 Re(...) convention, so it is stored halved.
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    yh = np.fft.fft(y) / np.sqrt(N)
    half = yh[: N // 2 + 1].copy()
    if N % 2 == 0:
        half[N // 2] = half[N // 2] / 2.0  # evaluate() doubles every m > 0 term
    S = np.abs(yh[: N // 2 + 1]) ** 2
    if M_max is not None and M_max > N // 2:
        half = np.concatenate([half, np.zeros(M_max - N // 2, complex)])
        S = np.concatenate([S, np.zeros(M_max - N // 2)])
    return TargetClass(psd=S, name=name, fixed_coeffs=half)

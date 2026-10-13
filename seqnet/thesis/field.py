"""The SOM-base field equation and the chain k_S -> gamma -> a -> g -> E (Thesis §9.3, App. K.2, App. L).

Field equation (Thesis eq. K.6), on the N-neuron ring with continuum-normalized convolutions:

    tau dr/dt = -r + phi( J_E * r - alpha k_S * r + I_ext(theta - omega t) ),
    (J * r)_i = sum_j J(r_i - r_j) r_j dr.

With phi = identity this is linear.  Fourier coefficients of connectivity kernels are continuum
coefficients J_hat(n) = dr sum_j J(r_j) e^{-i n r_j} (so that (J*r)^ = J_hat r^ exactly), and the
effective damping is gamma_n = 1 - J_E_hat(n) + alpha k_S_hat(n) (eq. 9.14, K.8).

Travelling steady state (spec 6.4 extension of eq. K.9).  In this module's convention the forward DFT
uses e^{-i n theta}; an input travelling toward increasing theta, I(theta - omega t), has bin-n
coefficient I_hat_n e^{-i n omega t}, and the linear steady state is

    r_hat_n(t) = I_hat_n e^{-i n omega t} / (gamma_n - i n omega tau).

(The spec writes gamma_n + i n omega tau, which is the same statement in the opposite sign
convention for n; see results/thesis/REPORT.md.)  For omega = 0 this is eq. K.9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .readout import spectral_filter

__all__ = [
    "continuum_coeffs",
    "effective_damping",
    "som_chain",
    "inverse_som_chain",
    "steady_state",
    "FieldResult",
    "simulate_field",
    "branching_ratios",
]


def continuum_coeffs(J: np.ndarray) -> np.ndarray:
    """J_hat(n) = dr sum_j J(r_j) e^{-i n r_j} on all N bins (continuum Fourier coefficient)."""
    J = np.asarray(J, dtype=float)
    return np.fft.fft(J) * (2 * np.pi / J.size)


def effective_damping(JE_hat: np.ndarray, kS_hat: np.ndarray, alpha: float) -> np.ndarray:
    """gamma_n = 1 - J_E_hat(n) + alpha k_S_hat(n) (Thesis eq. 9.13-9.14, K.8)."""
    return 1.0 - np.asarray(JE_hat) + alpha * np.asarray(kS_hat)


def som_chain(
    kS_hat: np.ndarray,
    JE_hat: np.ndarray,
    alpha: float,
    I_hat: np.ndarray,
    decay: float | None = None,
    S: np.ndarray | None = None,
) -> dict:
    """Forward chain k_S -> gamma -> a -> g -> E (Thesis eq. 9.16, K.8-K.10, 6.6).

    a_n = |I_hat_n|^2 / gamma_n^2 (eq. K.9).  To obtain a in the D10 operational units, pass I_hat as the
    *unnormalized* DFT of the input on the grid (so that a_n = |sum_i r_i e^{-i n r_i}|^2).
    """
    gamma = effective_damping(JE_hat, kS_hat, alpha)
    a = np.abs(I_hat) ** 2 / np.abs(gamma) ** 2
    out = {"gamma": gamma, "a": a}
    if decay is not None:
        g = spectral_filter(a, decay)
        out["g"] = g
        if S is not None:
            out["E"] = decay**2 * np.asarray(S) / (a + decay) ** 2
    return out


def inverse_som_chain(a: np.ndarray, JE_hat: np.ndarray, alpha: float, I_hat: np.ndarray) -> np.ndarray:
    """k_S_hat(n) achieving allocation a_n: gamma_n = |I_hat_n| / sqrt(a_n), k_S = (gamma - 1 + J_E)/alpha.

    The positive root gamma_n > 0 is taken (stable modes).  a_n = 0 requires infinite inhibition (inf).
    """
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore"):
        gamma = np.where(a > 0, np.abs(I_hat) / np.sqrt(np.where(a > 0, a, 1.0)), np.inf)
    return (gamma - 1.0 + np.real(JE_hat)) / alpha


def steady_state(I_hat: np.ndarray, gamma: np.ndarray, omega: float = 0.0, tau: float = 1.0, t: float = 0.0) -> np.ndarray:
    """Linear steady-state coefficients r_hat_n(t) = I_hat_n e^{-i n omega t} / (gamma_n - i n omega tau)."""
    N = len(I_hat)
    n = np.rint(np.fft.fftfreq(N, 1.0 / N))
    return I_hat * np.exp(-1j * n * omega * t) / (gamma - 1j * n * omega * tau)


@dataclass(frozen=True)
class FieldResult:
    t: np.ndarray
    r: np.ndarray  # (n_t, N)


def simulate_field(
    JE: np.ndarray,
    kS: np.ndarray,
    alpha: float,
    I_profile: np.ndarray | Callable[[np.ndarray], np.ndarray],
    omega: float = 0.0,
    tau: float = 1.0,
    dt: float = 0.01,
    t_max: float = 50.0,
    r0: np.ndarray | None = None,
    phi: Literal["linear", "sigmoid"] = "linear",
    kS_asym: np.ndarray | None = None,
    record_every: int = 10,
) -> FieldResult:
    """Integrate the field equation (Thesis eq. K.6) with FFT convolutions (forward Euler).

    ``I_profile`` is the input shape at t = 0 on the grid (or a function of angle); the drive at time t is
    I(theta - omega t), evaluated by spectral shift.  ``kS_asym`` adds an optional asymmetric SOM kernel.
    """
    JE = np.asarray(JE, dtype=float)
    N = JE.size
    theta = 2 * np.pi * np.arange(N) / N
    I0 = I_profile(theta) if callable(I_profile) else np.asarray(I_profile, dtype=float)
    I0_hat = np.fft.fft(I0)
    n = np.rint(np.fft.fftfreq(N, 1.0 / N))
    J_hat = continuum_coeffs(JE) - alpha * continuum_coeffs(kS)
    if kS_asym is not None:
        J_hat = J_hat - alpha * continuum_coeffs(kS_asym)
    f = (lambda u: u) if phi == "linear" else (lambda u: 1.0 / (1.0 + np.exp(-u)))
    r = np.zeros(N) if r0 is None else np.asarray(r0, dtype=float).copy()
    n_steps = int(round(t_max / dt))
    ts, rs = [0.0], [r.copy()]
    for s in range(1, n_steps + 1):
        t = (s - 1) * dt
        I_t = np.real(np.fft.ifft(I0_hat * np.exp(-1j * n * omega * t)))
        u = np.real(np.fft.ifft(J_hat * np.fft.fft(r))) + I_t
        r = r + dt / tau * (-r + f(u))
        if s % record_every == 0:
            ts.append(s * dt)
            rs.append(r.copy())
    return FieldResult(t=np.asarray(ts), r=np.asarray(rs))


def branching_ratios(JE_hat: np.ndarray, kS_hat: np.ndarray, alpha: float) -> np.ndarray:
    """Mode-resolved branching ratio m_n = J_E_hat(n) - alpha k_S_hat(n) = 1 - gamma_n (Thesis App. L, eq. L.1).

    [VERIFY] App. L is low priority in the spec and was not re-derived; this is the linear-response
    reading (each mode's recurrent gain), recorded for later checking against eq. L.1.
    """
    return 1.0 - effective_damping(JE_hat, kS_hat, alpha)

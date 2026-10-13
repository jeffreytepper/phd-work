"""The Braun et al. supplement map from target shape to connectivity (Braun Suppl. Sec. 2-5, 7; spec 6.10).

Conventions: f(theta, t) = sum_k f_hat_k e^{i k omega t} e^{-i k theta}; v_k = e^{-2 pi i k n / N};
lambda_k = sum_d cp_d e^{-2 pi i k d / N}.  Spectra of the scalar bump f0, x0* = phi^{-1}(f0) and y0* are computed by
the temporal DFT of one period sampled at ``n_t`` points (``conventions.temporal_fourier``).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import expit

from ..conventions import ifft, signed_modes, temporal_fourier
from ..targets import SequenceSpec, logit_target, wrapped_gaussian

__all__ = [
    "bump_spectra",
    "required_eigs",
    "ridge_shrunk_eigs",
    "profile_from_mode_eigs",
    "active_cutoff",
    "scaffolding_bound",
    "batch_ridge_W",
]


def bump_spectra(spec: SequenceSpec, tau: float, n_t: int = 4096) -> dict:
    """Temporal Fourier coefficients of f0, x0*, y0* = x0* + tau dx0*/dt on all n_t bins (Braun Suppl. eq. 2, Sec. 2).

    The bump of neuron 0 is sampled at t_b = b T / n_t.  y_hat_k = (1 + i k omega tau) x_hat_k exactly (Suppl. Sec. 2).
    """
    t = np.arange(n_t) * spec.T / n_t
    f0 = spec.amp * wrapped_gaussian(t, spec.width, spec.T, spec.n_images) + spec.offset
    x0 = logit_target(f0, spec.eps)
    k = signed_modes(n_t)
    w = 2 * np.pi / spec.T
    f_hat = temporal_fourier(f0)
    x_hat = temporal_fourier(x0)
    return {"k": k, "f_hat": f_hat, "x_hat": x_hat, "y_hat": (1 + 1j * k * w * tau) * x_hat,
            "t": t, "f0": f0, "x0": x0, "omega": w}


def required_eigs(
    spec: SequenceSpec,
    tau: float,
    K_max: int = 20,
    convention: Literal["supplement", "methods"] = "supplement",
    n_t: int = 4096,
) -> dict[int, complex]:
    """Sequence-required eigenvalues lambda_k for |k| <= K_max.

    * ``'supplement'``: lambda_k = y_hat_k / f_hat_k = (1 + i k omega tau) x_hat_k / f_hat_k (Braun Suppl. Cor. 3.2).
    * ``'methods'``: FORCE trains the *input* toward x* (Braun Methods eq. 7), so the activation follows the leaky
      filter h_hat_k = x_hat_k / (1 + i k omega tau) and the realized rates are F' = phi(h); support W F' = x* gives
      lambda_k = x_hat_k / F'_hat_k (spec 6.10; [VERIFY] A1).
    """
    sp = bump_spectra(spec, tau, n_t)
    k, w = sp["k"], sp["omega"]
    if convention == "supplement":
        num, den = sp["y_hat"], sp["f_hat"]
    elif convention == "methods":
        h_hat = sp["x_hat"] / (1 + 1j * k * w * tau)
        h0 = np.real(ifft(h_hat * n_t))
        num, den = sp["x_hat"], temporal_fourier(expit(h0))
    else:
        raise ValueError(convention)
    return {int(kk): complex(num[kk % n_t] / den[kk % n_t]) for kk in range(-K_max, K_max + 1)}


def ridge_shrunk_eigs(eig: dict[int, complex], f_hat: dict[int, complex] | np.ndarray, alpha: float) -> dict[int, complex]:
    """lambda_k^(alpha) = |f_hat_k|^2 / (|f_hat_k|^2 + alpha) lambda_k (Braun Suppl. Sec. 5)."""
    out = {}
    for k, lam in eig.items():
        fk = f_hat[k] if isinstance(f_hat, dict) else f_hat[k % len(f_hat)]
        out[k] = abs(fk) ** 2 / (abs(fk) ** 2 + alpha) * lam
    return out


def profile_from_mode_eigs(eig: dict[int, complex], N: int) -> np.ndarray:
    """cp_d = (1/N) sum_{k in eig} lambda_k e^{2 pi i k d / N} (Braun Suppl. Sec. 3, end), real part."""
    lam = np.zeros(N, complex)
    for k, v in eig.items():
        lam[k % N] += v
    return np.real(ifft(lam))


def active_cutoff(f_hat: np.ndarray, y_hat: np.ndarray, eps: float) -> dict:
    """D_f(eps), D_y*(eps), D_seq = max (Braun Suppl. Sec. 2): smallest D with sum_{|k|>D} |c_k|^2 <= eps^2."""
    n = f_hat.size
    k = np.abs(signed_modes(n))

    def D(c):
        p = np.abs(c) ** 2
        for d in range(n // 2 + 1):
            if p[k > d].sum() <= eps**2:
                return d
        return n // 2

    Df, Dy = D(f_hat), D(y_hat)
    return {"D_f": Df, "D_y": Dy, "D_seq": max(Df, Dy)}


def scaffolding_bound(m: np.ndarray, q: int, N_d: int, N_u: int, a_d: float, a_u: float, s_d: float, s_u: float,
                      omega_u: float) -> np.ndarray:
    """|J|_2 >= q sqrt(N_d/N_u) (a_d s_d)/(a_u s_u) exp[(q m omega_u)^2 (s_u^2 - s_d^2)/2] (Braun Suppl. eq. 6)."""
    m = np.asarray(m, dtype=float)
    return q * np.sqrt(N_d / N_u) * (a_d * s_d) / (a_u * s_u) * np.exp(0.5 * (q * m * omega_u) ** 2 * (s_u**2 - s_d**2))


def batch_ridge_W(F: np.ndarray, Y: np.ndarray, J0: np.ndarray | None, beta: float) -> np.ndarray:
    """W = (Y F^T + beta J0)(F F^T + beta I)^{-1}: ridge centred on J0 (spec Section 3.4 fact 4).

    F: (N, n_samples) presynaptic rates; Y: (N, n_samples) required inputs.  With P(0) = p I, RLS from J0 minimizes
    sum |W r - y|^2 + (1/p) |W - J0|_F^2, i.e. beta = 1/p (test T9).
    """
    N = F.shape[0]
    J0 = np.zeros((Y.shape[0], N)) if J0 is None else J0
    A = F @ F.T + beta * np.eye(N)
    return np.linalg.solve(A, (Y @ F.T + beta * J0).T).T


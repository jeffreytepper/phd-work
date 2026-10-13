"""Fixed conventions of the main package (spec Section 4.1) and the only module that calls ``np.fft`` directly.

* ``W[i, j]``: weight from presynaptic j to postsynaptic i (Braun Methods line 782).
* Neuron positions theta_n = 2 pi n / N; neuron n peaks at mu_n = n T / N, so the bump moves toward increasing n.
* Profile index d = (m - n) mod N = presynaptic minus postsynaptic (Braun eq. 11; Braun Suppl. Sec. 3):
  cp_d = (1/N) sum_n W[n, (n + d) mod N], which equals the circulant projection W_cp.
* Spatial eigenvalues (unnormalized DFT): lambda_k = sum_d cp_d e^{-2 pi i k d / N}; then W_cp v_k = lambda_k v_k with
  (v_k)_n = e^{-2 pi i k n / N} (Braun Suppl. Sec. 3).
* Temporal Fourier: f_hat_k = (1/T) int_0^T f(t) e^{-i k omega t} dt, omega = 2 pi / T (Braun Suppl. eq. 2), computed
  by the DFT over one period of samples.  Population expansion f(theta, t) = sum_k f_hat_k e^{i k omega t} e^{-i k theta}.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "profile_from_matrix",
    "circulant_from_profile",
    "eigs_from_profile",
    "profile_from_eigs",
    "centred_profile",
    "even_odd_split",
    "circulant_projection",
    "shift",
    "unitary_dft",
    "alloc_from_rates",
    "temporal_fourier",
    "signed_modes",
    "kernel_from_temporal_profile",
    "temporal_profile_from_kernel",
    "fft",
    "ifft",
]

# thin aliases so that other modules never touch np.fft (spec Section 4)
fft = np.fft.fft
ifft = np.fft.ifft


def profile_from_matrix(W: np.ndarray) -> np.ndarray:
    """cp_d = (1/N) sum_n W[n, (n + d) mod N], d = 0..N-1 (Braun eq. 11; circular shift of rows, mean across rows).

    Works on a stack of matrices with shape (..., N, N).
    """
    W = np.asarray(W)
    N = W.shape[-1]
    n = np.arange(N)
    idx = (n[:, None] + n[None, :]) % N  # idx[n, d] = (n + d) mod N
    return np.take_along_axis(W, np.broadcast_to(idx, W.shape), axis=-1).mean(axis=-2)


def circulant_from_profile(cp: np.ndarray) -> np.ndarray:
    """W[n, m] = cp[(m - n) mod N] (inverse of :func:`profile_from_matrix` on circulant matrices)."""
    cp = np.asarray(cp)
    N = cp.size
    n = np.arange(N)
    return cp[(n[None, :] - n[:, None]) % N]


def circulant_projection(W: np.ndarray) -> np.ndarray:
    """W_cp = (1/N) sum_r S^{-r} W S^r, the circulant average (Braun Suppl. Prop. 3.1)."""
    return circulant_from_profile(profile_from_matrix(W))


def eigs_from_profile(cp: np.ndarray) -> np.ndarray:
    """lambda_k = sum_d cp_d e^{-2 pi i k d / N} on all N bins (Braun Suppl. Sec. 3)."""
    return np.fft.fft(np.asarray(cp), axis=-1)


def profile_from_eigs(eig: np.ndarray) -> np.ndarray:
    """Inverse of :func:`eigs_from_profile` (real part; eigenvalues of a real profile are Hermitian)."""
    return np.real(np.fft.ifft(np.asarray(eig), axis=-1))


def signed_modes(N: int) -> np.ndarray:
    """Signed index of each DFT bin: 0, 1, ..., -1."""
    return np.rint(np.fft.fftfreq(N, d=1.0 / N)).astype(int)


def centred_profile(cp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d, cp_d) with d in [-floor(N/2), floor(N/2)] for plotting and feature extraction (Braun Fig. 1C)."""
    cp = np.asarray(cp)
    N = cp.shape[-1]
    d = np.arange(-(N // 2), N - N // 2)
    return d, np.take(cp, d % N, axis=-1)


def even_odd_split(cp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Even and odd parts about d = 0: e_d = (cp_d + cp_{-d})/2, o_d = (cp_d - cp_{-d})/2 (index d mod N)."""
    cp = np.asarray(cp)
    N = cp.shape[-1]
    rev = np.take(cp, (-np.arange(N)) % N, axis=-1)
    return (cp + rev) / 2, (cp - rev) / 2


def shift(x: np.ndarray, r: int, axis: int = -1) -> np.ndarray:
    """(S^r x)_n = x_{(n + r) mod N} (Braun Suppl. Sec. 3)."""
    return np.roll(x, -r, axis=axis)


def unitary_dft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unitary spatial DFT x_hat_m = N^{-1/2} sum_i x_i e^{-2 pi i m i / N} (Thesis eq. 4.1)."""
    x = np.asarray(x)
    return np.fft.fft(x, axis=axis) / np.sqrt(x.shape[axis])


def alloc_from_rates(r_snapshot: np.ndarray) -> np.ndarray:
    """Thesis operational power a_m = |sum_i r_i e^{-2 pi i m i/N}|^2 = N |k_hat_m|^2 of a population snapshot (D10)."""
    return np.abs(np.fft.fft(np.asarray(r_snapshot), axis=-1)) ** 2


def temporal_fourier(f: np.ndarray, axis: int = -1) -> np.ndarray:
    """f_hat_k = (1/n) sum_b f(t_b) e^{-i k omega t_b} over n samples of exactly one period (Braun Suppl. eq. 2)."""
    f = np.asarray(f)
    return np.fft.fft(f, axis=axis) / f.shape[axis]


def kernel_from_temporal_profile(f0: np.ndarray) -> np.ndarray:
    """Spatial kernel at fixed time from a temporal profile sampled on the neuron grid (spec Section 4.2).

    With f0 sampled at t_n = n T / N (n = 0..N-1), neuron n's rate at t = 0 is f0(-n T/N), so the population snapshot
    (bump position c = 0) is k(r_n) = f0((-n) mod N).  For symmetric f0 this equals f0 itself.
    """
    f0 = np.asarray(f0)
    N = f0.shape[-1]
    return np.take(f0, (-np.arange(N)) % N, axis=-1)


def temporal_profile_from_kernel(k: np.ndarray) -> np.ndarray:
    """Inverse of :func:`kernel_from_temporal_profile` (the map is an involution)."""
    return kernel_from_temporal_profile(k)

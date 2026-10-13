"""Fourier conventions for the thesis framework (Thesis §4.1) and the normalization resolver (spec 6A.5, D10).

Conventions used throughout ``seqnet.thesis``
---------------------------------------------
* Neuron positions r_i = 2*pi*i/N, i = 0..N-1 (Thesis §2.1).
* ``unitary_dft``: k_hat_m = N^{-1/2} sum_i k(r_i) exp(-i m r_i)  (Thesis eq. 4.1).
* ``circulant_eigs``: K_m = sum_i k(r_i) exp(-i m r_i) = sqrt(N) * k_hat_m.  These are the
  eigenvalues of the circulant design matrix X_{t,i} = k(r_i - r_t) at T_obs = N.

Normalization resolver (D10)
----------------------------
Thesis eq. 4.2 / 4.7 write X = F* diag(k_hat) F with F unitary, which would make the singular
values of X equal |k_hat_m|.  For a circulant X whose rows are k(r_i - c_t), the singular values
are |K_m| = sqrt(N) |k_hat_m| (test TT1).  The *operational* per-mode power that enters
g_m, E(m) and the optimum is therefore the squared singular value at T_obs = N:

    a_m := |K_m|^2 = N |k_hat_m|^2          (``alloc_from_kernel``)

and the spectral budget is K0 = sum_m a_m = N sum_i k(r_i)^2 (Parseval).

Half-spectrum storage (spec 6A.5, A6)
-------------------------------------
Real symmetric kernels have a_{-m} = a_m.  Allocations and spectra are stored on m = 0..floor(N/2)
together with a multiplicity vector (1 for m = 0 and for the Nyquist mode of even N, 2 otherwise),
so that sums over all signed modes are ``sum(mult * a)``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "unitary_dft",
    "unitary_idft",
    "circulant_eigs",
    "alloc_from_kernel",
    "half_modes",
    "mode_multiplicity",
    "signed_modes",
    "full_from_half",
    "kernel_from_half_spectrum",
    "trig_interpolate",
]


def unitary_dft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unitary DFT, Thesis eq. (4.1): x_hat_m = N^{-1/2} sum_i x_i exp(-2 pi i m i / N)."""
    x = np.asarray(x)
    n = x.shape[axis]
    return np.fft.fft(x, axis=axis) / np.sqrt(n)


def unitary_idft(x_hat: np.ndarray, axis: int = -1) -> np.ndarray:
    """Inverse of :func:`unitary_dft` (Thesis eq. 4.1 inverted)."""
    x_hat = np.asarray(x_hat)
    n = x_hat.shape[axis]
    return np.fft.ifft(x_hat, axis=axis) * np.sqrt(n)


def circulant_eigs(k: np.ndarray) -> np.ndarray:
    """Unnormalized DFT K_m = sum_i k(r_i) e^{-i m r_i} = sqrt(N) k_hat_m.

    These are the singular values (up to phase) of the circulant design matrix at T_obs = N
    (Thesis eq. 4.2/4.7 with the D10 correction; test TT1).
    """
    return np.fft.fft(np.asarray(k, dtype=float))


def half_modes(N: int) -> np.ndarray:
    """Non-negative mode indices m = 0..floor(N/2)."""
    return np.arange(N // 2 + 1)


def mode_multiplicity(N: int, n_modes: int | None = None) -> np.ndarray:
    """Number of signed modes represented by each stored m >= 0.

    1 for m = 0, 1 for m = N/2 when N is even, 2 otherwise.  If ``n_modes`` exceeds
    floor(N/2)+1 (spectra extending beyond the network Nyquist, e.g. S(m) for the network-floor
    term), the extra modes all have multiplicity 2.
    """
    n_half = N // 2 + 1
    n_modes = n_half if n_modes is None else int(n_modes)
    mult = np.full(n_modes, 2.0)
    mult[0] = 1.0
    if N % 2 == 0 and n_modes > N // 2:
        mult[N // 2] = 1.0 if n_modes == n_half else 2.0
        # When the array extends past Nyquist it represents a continuous-ring spectrum, where
        # +N/2 and -N/2 are distinct modes; keep multiplicity 2 there.
    return mult


def signed_modes(N: int) -> np.ndarray:
    """Signed mode index of each FFT bin: 0, 1, ..., -1 (numpy fftfreq order times N)."""
    return np.rint(np.fft.fftfreq(N, d=1.0 / N)).astype(int)


def alloc_from_kernel(k: np.ndarray, half: bool = True) -> np.ndarray:
    """Operational kernel power a_m = |K_m|^2 = N |k_hat_m|^2 (D10 resolution; Thesis eq. 7.1).

    Returns m = 0..floor(N/2) if ``half`` (assumes a real kernel so a_{-m} = a_m), else all N bins.
    """
    a = np.abs(circulant_eigs(k)) ** 2
    if half:
        return a[: len(k) // 2 + 1].copy()
    return a


def full_from_half(h: np.ndarray, N: int) -> np.ndarray:
    """Expand a half spectrum (m = 0..floor(N/2)) to all N FFT bins assuming h_{-m} = h_m."""
    h = np.asarray(h)
    m = np.abs(signed_modes(N))
    return h[m]


def kernel_from_half_spectrum(K_half: np.ndarray, N: int) -> np.ndarray:
    """Real symmetric kernel k(r_i) whose unnormalized DFT is K_m = K_half[|m|] (real).

    k(r_i) = (1/N) sum_m K_m e^{i m r_i}  (inverse of ``circulant_eigs``).
    """
    K_full = full_from_half(np.asarray(K_half, dtype=float), N)
    return np.real(np.fft.ifft(K_full))


def trig_interpolate(k: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate the band-limited (trigonometric) interpolant of grid samples k(r_i) at arbitrary u.

    k(u) = (1/N) sum_{m signed} K_m e^{i m u}; for even N the Nyquist term is split as a cosine so
    the interpolant is real.  Exact on the grid.  Used for off-grid bump positions c_t.
    """
    k = np.asarray(k, dtype=float)
    N = k.size
    K = np.fft.fft(k)
    m = signed_modes(N).astype(float)
    u = np.asarray(u, dtype=float)
    phase = np.exp(1j * np.multiply.outer(u, m))
    if N % 2 == 0:
        nyq = N // 2
        # numpy labels the Nyquist bin as -N/2; replace exp(-i N/2 u) by cos(N/2 u)
        phase[..., nyq] = np.cos(nyq * u)
    return np.real(phase @ K) / N

"""Spec Section 14 tests T1-T9 and T15 (main package)."""
import numpy as np
import pytest
from scipy.linalg import expm

from seqnet import conventions as cv
from seqnet.analysis.motifs import motifs
from seqnet.spectral import braun_map as bm
from seqnet.targets import SequenceSpec, fourier_coeffs_gaussian, wrapped_gaussian

rng = np.random.default_rng(0)


def test_T1_circulant_eigenvectors_and_round_trip():
    """Braun Suppl. Sec. 3: W_cp v_k = lambda_k v_k with (v_k)_n = e^{-2 pi i k n/N}; profile round trips (1e-12)."""
    N = 37
    cp = rng.normal(size=N)
    W = cv.circulant_from_profile(cp)
    lam = cv.eigs_from_profile(cp)
    n = np.arange(N)
    for k in range(N):
        v = np.exp(-2j * np.pi * k * n / N)
        np.testing.assert_allclose(W @ v, lam[k] * v, atol=1e-12)
    np.testing.assert_allclose(cv.profile_from_matrix(W), cp, atol=1e-12)
    np.testing.assert_allclose(cv.profile_from_eigs(lam), cp, atol=1e-12)
    # profile of a general matrix is the profile of its circulant projection
    M = rng.normal(size=(N, N))
    np.testing.assert_allclose(cv.profile_from_matrix(cv.circulant_projection(M)), cv.profile_from_matrix(M), atol=1e-12)


def test_T2_parseval_and_alloc():
    """Unitary Parseval; operational a_m sums to N sum_i x_i^2 (D10; corrected Section 3.4 fact 6)."""
    x = rng.normal(size=64)
    np.testing.assert_allclose(np.sum(np.abs(cv.unitary_dft(x)) ** 2), np.sum(x**2), rtol=1e-12)
    np.testing.assert_allclose(np.sum(cv.alloc_from_rates(x)), 64 * np.sum(x**2), rtol=1e-12)


def test_T3_kernel_temporal_conversion():
    """k(r_n) = f0(-n T/N); round trip exact; |k_hat_m|^2 = N |f_hat_m|^2 on the bin grid (spec 4.2 [VERIFY])."""
    N, T = 101, 8.5
    t = np.arange(N) * T / N
    f0 = 0.5 * wrapped_gaussian(t, 0.9, T) + 0.1 + 0.05 * np.sin(2 * np.pi * t / T)  # asymmetric on purpose
    k = cv.kernel_from_temporal_profile(f0)
    np.testing.assert_allclose(cv.temporal_profile_from_kernel(k), f0, atol=1e-14)
    # the population snapshot at t = 0 of the sequence f_n(t) = f0(t - n T/N)
    snap = np.array([np.interp((-(n * T / N)) % T, t, f0, period=T) for n in range(N)])
    np.testing.assert_allclose(k, snap, atol=1e-12)
    lhs, rhs = np.abs(cv.unitary_dft(k)) ** 2, N * np.abs(cv.temporal_fourier(f0)) ** 2
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12 * rhs.max())


def test_T4_gaussian_fourier_coefficients():
    """Numerical temporal coefficients of the wrapped-Gaussian target match Braun Suppl. eq. 3 (1e-8 relative)."""
    spec = SequenceSpec(N=501, amp=0.5, width=1.0, offset=0.2)
    sp = bm.bump_spectra(spec, 0.3224, n_t=4096)
    k = np.arange(-15, 16)
    num = sp["f_hat"][k % 4096]
    ana = fourier_coeffs_gaussian(k, spec.amp, spec.width, spec.offset, spec.T)
    big = np.abs(ana) > 1e-9 * np.abs(ana).max()
    np.testing.assert_allclose(num.real[big], ana[big], rtol=1e-8)  # 1e-8 relative where above roundoff
    np.testing.assert_allclose(num.real, ana, atol=1e-14)
    assert np.abs(num.imag).max() < 1e-12


def test_T5_image_truncation():
    t = np.linspace(-20, 20, 2001)
    for s in (0.5, 1.0, 1.9):
        np.testing.assert_allclose(wrapped_gaussian(t, s, 8.5, 6), wrapped_gaussian(t, s, 8.5, 20), atol=1e-12)


def _D_mode_sequence(spec, tau, D, N, ts):
    sp = bm.bump_spectra(spec, tau)
    n = np.arange(N)
    w = sp["omega"]
    F = np.zeros((N, ts.size))
    Y = np.zeros((N, ts.size))
    for k in range(-D, D + 1):
        v = np.exp(-2j * np.pi * k * n / N)
        F += np.real(sp["f_hat"][k % 4096] * np.exp(1j * k * w * ts)[None, :] * v[:, None])
        Y += np.real(sp["y_hat"][k % 4096] * np.exp(1j * k * w * ts)[None, :] * v[:, None])
    return F, Y


def test_T6_degenerate_family_supports_sequence():
    """Any W = W_D + W~ with W~ P_D = 0 satisfies W f^(D)(t) = y*^(D)(t) (Braun Suppl. Prop. 3.1, 4.1; 1e-10)."""
    N, D, tau = 64, 8, 0.3224
    spec = SequenceSpec(N=N, amp=0.5, width=1.0, offset=0.1)
    eig = bm.required_eigs(spec, tau, K_max=D)
    W_D = np.zeros((N, N))
    n = np.arange(N)
    for k, lam in eig.items():
        v = np.exp(-2j * np.pi * k * n / N)
        W_D += np.real(lam * np.outer(v, v.conj())) / N
    from seqnet.analysis.metrics import active_projector

    P = active_projector(N, D)
    Wt = rng.normal(size=(N, N)) @ (np.eye(N) - P)  # W~ P_D = 0
    ts = np.linspace(0, spec.T, 37)
    F, Y = _D_mode_sequence(spec, tau, D, N, ts)
    for W in (W_D, W_D + Wt):
        np.testing.assert_allclose(W @ F, Y, atol=1e-10 * np.abs(Y).max())


def test_T7_linear_support_reproduces_sequence():
    """Circulant from required eigenvalues, linear phi: tau x' = -x + W x reproduces the D-mode sequence for 3 periods
    from the exact initial state, without noise (Cor. 3.2; rel. error < 1e-4).  With phi linear, f = x* and
    lambda_k = 1 + i k omega tau on the active modes; integrated exactly with the matrix exponential."""
    N, D, tau, T = 64, 6, 0.3224, 8.5
    w = 2 * np.pi / T
    n = np.arange(N)
    rng_ = np.random.default_rng(1)
    c = {k: (rng_.normal() + 1j * rng_.normal()) for k in range(1, D + 1)}
    c[0] = 0.3
    for k in range(1, D + 1):
        c[-k] = np.conj(c[k])
    W = np.zeros((N, N), complex)
    for k in range(-D, D + 1):
        v = np.exp(-2j * np.pi * k * n / N)
        W += (1 + 1j * k * w * tau) * np.outer(v, v.conj()) / N
    W = np.real(W)

    def f(t):
        return np.real(sum(c[k] * np.exp(1j * k * w * t) * np.exp(-2j * np.pi * k * n / N) for k in c))

    A = (-np.eye(N) + W) / tau
    for t in np.linspace(0, 3 * T, 13)[1:]:
        x = expm(A * t) @ f(0.0)
        assert np.linalg.norm(x - f(t)) / np.linalg.norm(f(t)) < 1e-4


def test_T8_odd_part_is_derivative_of_even_part():
    """Supplement convention: odd(cp) = omega tau d/dtheta even(cp) (spec Section 3.4 fact 2; 1e-8).

    Restricted to modes with |f_hat_k| well above roundoff (here |k| <= 8 at width 1 s, |f_hat_8| ~ 1e-8): beyond them
    lambda_k = y_hat_k / f_hat_k is a ratio of roundoff-level numbers (spec 6.10 uses active modes only)."""
    N, tau = 501, 0.3224
    spec = SequenceSpec(N=N, amp=0.5, width=1.0, offset=0.25)
    K = 8
    eig = bm.required_eigs(spec, tau, K_max=K)
    cp = bm.profile_from_mode_eigs(eig, N)
    even, odd = cv.even_odd_split(cp)
    w = 2 * np.pi / spec.T
    lam = cv.eigs_from_profile(even)  # spectral derivative of the even part in theta = 2 pi d / N
    k = cv.signed_modes(N)
    d_even = np.real(cv.ifft(1j * k * lam))
    np.testing.assert_allclose(odd, w * tau * d_even, atol=1e-8 * np.abs(cp).max())
    # eigenvalue form: Im lambda_k = k omega tau Re lambda_k for a symmetric bump
    for kk in range(1, K + 1):
        assert abs(eig[kk].imag - kk * w * tau * eig[kk].real) < 1e-8 * abs(eig[kk])


def test_T9_batch_ridge_equals_rls():
    """Single open-loop pass of standard RLS from J0 with P(0) = p I equals (Y F^T + J0/p)(F F^T + I/p)^{-1} (1e-6).

    Standard update (no extra c factor).  The printed Braun variant (c with the updated P) does not equal it."""
    N, n_s, p = 30, 200, 0.08
    F = rng.uniform(0, 1, size=(N, n_s))
    Y = rng.normal(size=(N, n_s))
    J0 = rng.normal(size=(N, N)) / np.sqrt(N)

    def rls(c_updated):
        J, P = J0.copy(), p * np.eye(N)
        for s in range(n_s):
            r = F[:, s]
            e = J @ r - Y[:, s]
            Pr = P @ r
            den = 1 + r @ Pr
            P -= np.outer(Pr, Pr) / den
            Prn = Pr / den
            c = 1 / (1 + r @ Prn) if c_updated else 1.0
            J -= c * np.outer(e, Prn)
        return J

    ref = bm.batch_ridge_W(F, Y, J0, 1 / p)
    np.testing.assert_allclose(rls(False), ref, rtol=1e-6, atol=1e-8)
    assert np.linalg.norm(rls(True) - ref) / np.linalg.norm(ref) > 1e-3


def test_T15_motifs_of_centred_circulant():
    """Corrected T15: tau_div = tau_con = -1/(N-1), tau_chn = -2 tau_rec/(N-1) exactly (spec says 0)."""
    N = 101
    W = cv.circulant_from_profile(rng.normal(size=N))
    m = motifs(W)
    np.testing.assert_allclose(m["tau_div"], -1 / (N - 1), atol=1e-12)
    np.testing.assert_allclose(m["tau_con"], -1 / (N - 1), atol=1e-12)
    np.testing.assert_allclose(m["tau_chn"], -2 * m["tau_rec"] / (N - 1), atol=1e-12)
    # and against a direct implementation of Braun eq. 14-15
    X = W - W.mean()
    s2 = np.trace(X @ X.T) / N
    assert abs(m["tau_rec"] - np.trace(X @ X) / (s2 * N)) < 1e-12
    assert abs(m["tau_div"] - (np.sum(X @ X.T) - np.trace(X @ X.T)) / (s2 * N * (N - 1))) < 1e-12

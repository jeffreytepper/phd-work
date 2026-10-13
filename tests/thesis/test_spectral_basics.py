"""TT1, TT2, TT3, TT5: normalization resolver, SVD-DFT identity, per-mode ridge, effective rank."""
import numpy as np
import pytest

import seqnet.thesis as th

N = 64


def _kernels():
    return [
        th.Kernel.from_family("gaussian", N, sigma=0.3),
        th.Kernel.from_family("box", N, R=0.8),
        th.Kernel.from_family("bspline", N, p=3, h=0.25),
        th.Kernel.from_family("gabor", N, sigma=0.4, xi0=5.0),
        th.Kernel.from_family("point", N),
    ]


@pytest.mark.parametrize("k", _kernels(), ids=lambda k: k.name)
def test_TT1_singular_values_are_sqrtN_khat(k):
    """TT1 (Thesis eq. 4.1, 4.2, 4.7; D10): singular values of X at T_obs = N equal sqrt(N)|k_hat_m|, not |k_hat_m|."""
    X = th.design_matrix(k, th.Schedule(N, 1))
    s = np.sort(np.linalg.svd(X, compute_uv=False))
    k_hat = th.unitary_dft(k.values)
    np.testing.assert_allclose(s, np.sort(np.sqrt(N) * np.abs(k_hat)), rtol=1e-10, atol=1e-10 * s.max())
    # the thesis's literal statement (singular values = |k_hat_m|) is off by sqrt(N):
    assert not np.allclose(s, np.sort(np.abs(k_hat)))
    # operational a_m equals the squared singular values, and K0 = N sum k^2 (Parseval)
    np.testing.assert_allclose(np.sort(th.alloc_from_kernel(k.values, half=False)), s**2, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(k.energy(), N * np.sum(k.values**2), rtol=1e-12)


@pytest.mark.parametrize("k", _kernels(), ids=lambda k: k.name)
def test_TT2_right_singular_vectors_are_fourier_modes(k):
    """TT2 (Thesis eq. 4.7): F X^T X F* is diagonal with entries |K_m|^2."""
    X = th.design_matrix(k, th.Schedule(N, 1))
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    D = F @ X.T @ X @ F.conj().T
    off = D - np.diag(np.diag(D))
    assert np.abs(off).max() < 1e-10 * np.abs(D).max()
    np.testing.assert_allclose(np.real(np.diag(D)), np.abs(k.spectrum()) ** 2, rtol=1e-9, atol=1e-9 * np.abs(D).max())


def test_TT3_per_mode_ridge_equals_matrix_ridge():
    """TT3 (Thesis eq. 3.13 vs 4.5)."""
    rng = np.random.default_rng(1)
    for k in _kernels():
        X = th.design_matrix(k, th.Schedule(N, 1))
        y = rng.normal(size=N)
        for decay in (1e-3, 1e-1, 10.0):
            w = th.ridge_readout(X, y, decay)
            w_hat = th.per_mode_ridge(k.spectrum(), th.unitary_dft(y), decay)
            np.testing.assert_allclose(np.real(th.unitary_idft(w_hat)), w, rtol=1e-8, atol=1e-10 * np.abs(w).max())


def test_TT3_asymmetric_kernel_convention():
    """Per-mode ridge carries K_m (not conj) for X_{t,i} = k(r_i - r_t); checked on an asymmetric kernel."""
    rng = np.random.default_rng(2)
    k = th.Kernel(values=rng.random(N))
    X = th.design_matrix(k, th.Schedule(N, 1))
    y = rng.normal(size=N)
    w = th.ridge_readout(X, y, 0.3)
    w_hat = th.per_mode_ridge(k.spectrum(), th.unitary_dft(y), 0.3)
    np.testing.assert_allclose(np.real(th.unitary_idft(w_hat)), w, rtol=1e-8, atol=1e-10)


FAMILY_PARAMS = [
    ("gaussian", {"sigma": 0.15}),
    ("gaussian", {"sigma": 0.4}),
    ("box", {"R": 0.6}),
    ("bspline", {"p": 3, "h": 0.15}),
    ("gabor", {"sigma": 0.3, "xi0": 8.0}),
    ("point", {}),
]


@pytest.mark.parametrize("name,params", FAMILY_PARAMS)
def test_TT5_effective_rank_matches_analytic_spectrum(name, params):
    """TT5 (Thesis eq. 4.8-4.13): numerical r_k(eps) vs the count from the analytic continuous spectrum.

    Uses N = 256 so periodization effects are small; tolerance max(2 modes, 3% of the count) (the box's
    1/m sinc sidelobes hover near the threshold, so grid discretization of R moves several crossings).  For the Gabor the reference is the spectral maximum (its DC coefficient ~ 0).
    """
    Nn = 256
    k = th.Kernel.from_family(name, Nn, **params)
    ref = "max" if name == "gabor" else "dc"
    eps = 1e-2
    num = k.effective_rank(eps, reference=ref)
    m = np.rint(np.fft.fftfreq(Nn, 1.0 / Nn))
    ratio = th.analytic_spectrum_ratio(name, m, **params)
    if ref == "max":
        ratio = ratio / ratio.max()
    ana = int(np.sum(ratio > eps))
    assert abs(num - ana) <= max(2, 0.03 * ana), (num, ana)

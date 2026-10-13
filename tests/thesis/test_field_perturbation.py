"""TT14 (SOM chain and field equation), TT16 (coupling error ~ eps^2)."""
import numpy as np

import seqnet.thesis as th

N = 64


def _kernels():
    th_ = 2 * np.pi * np.arange(N) / N
    u = (th_ + np.pi) % (2 * np.pi) - np.pi
    JE = 0.8 * np.exp(-(u**2) / (2 * 0.5**2)) / (np.sqrt(2 * np.pi) * 0.5)
    kS = np.exp(-(u**2) / (2 * 1.0**2)) / (np.sqrt(2 * np.pi) * 1.0)
    I0 = np.exp(-(u**2) / (2 * 0.3**2))
    return JE, kS, I0


def test_TT14_chain_round_trip():
    JE, kS, I0 = _kernels()
    alpha = 0.7
    JE_hat, kS_hat, I_hat = th.continuum_coeffs(JE), th.continuum_coeffs(kS), np.fft.fft(I0)
    out = th.som_chain(kS_hat, JE_hat, alpha, I_hat, decay=1e-2)
    assert np.all(np.real(out["gamma"]) > 0)
    kS_back = th.inverse_som_chain(out["a"], JE_hat, alpha, I_hat)
    np.testing.assert_allclose(kS_back, np.real(kS_hat), rtol=1e-9, atol=1e-12)


def test_TT14_static_steady_state():
    """Linear field equation with a static input relaxes to r_hat = I_hat / gamma (eq. K.9)."""
    JE, kS, I0 = _kernels()
    alpha = 0.7
    gamma = th.effective_damping(th.continuum_coeffs(JE), th.continuum_coeffs(kS), alpha)
    res = th.simulate_field(JE, kS, alpha, I0, omega=0.0, tau=1.0, dt=0.01, t_max=60.0)
    pred = np.real(np.fft.ifft(th.steady_state(np.fft.fft(I0), gamma)))
    np.testing.assert_allclose(res.r[-1], pred, rtol=1e-6, atol=1e-8 * np.abs(pred).max())


def test_TT14_travelling_steady_state():
    """Travelling input I(theta - omega t): r_hat_n = I_hat_n e^{-i n omega t} / (gamma_n - i n omega tau)
    (spec 6.4 / T16; sign as in this module's DFT convention)."""
    JE, kS, I0 = _kernels()
    alpha, tau, omega = 0.7, 1.0, 2 * np.pi / 8.5
    gamma = th.effective_damping(th.continuum_coeffs(JE), th.continuum_coeffs(kS), alpha)
    dt = 0.002
    res = th.simulate_field(JE, kS, alpha, I0, omega=omega, tau=tau, dt=dt, t_max=60.0, record_every=50)
    # Forward Euler samples the input at t - dt; compare at the integration time of the last state
    t_last = res.t[-1]
    pred = np.real(np.fft.ifft(th.steady_state(np.fft.fft(I0), gamma, omega, tau, t_last)))
    err = np.abs(res.r[-1] - pred).max() / np.abs(pred).max()
    assert err < 5e-3  # Euler O(dt) phase error
    # the opposite sign convention (gamma + i n omega tau) is clearly worse
    n = np.rint(np.fft.fftfreq(N, 1 / N))
    wrong = np.real(np.fft.ifft(np.fft.fft(I0) * np.exp(-1j * n * omega * t_last) / (gamma + 1j * n * omega * tau)))
    assert np.abs(res.r[-1] - wrong).max() / np.abs(pred).max() > 10 * err


def test_TT16_coupling_error_scales_as_eps_squared():
    """App. E.4: with a smooth zero-mean position dependence k(r; theta) = k(r)(1 + eps cos theta), the excess
    expected error over the invariant kernel is O(eps^2) (exponent within 0.1 of 2)."""
    k = th.Kernel.from_family("gaussian", 32, sigma=0.4)
    S = th.TargetClass.gaussian_psd(16, beta=0.02).psd
    out = th.coupling_error(k, S, 0.5, np.array([0.01, 0.02, 0.04, 0.08]))
    assert abs(out["exponent"] - 2.0) < 0.1, out

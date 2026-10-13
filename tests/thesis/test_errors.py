"""TT8: bias + aliasing + floor vs the error measured by simulation (Thesis eq. 6.6-6.9, 6.6')."""
import numpy as np

import seqnet.thesis as th

N = 64


def test_TT8_full_sampling_bandlimited_exact():
    """T_obs = N, target band-limited below N/2: measured error = E_bias (eq. 6.7) exactly (random phases)."""
    rng = np.random.default_rng(0)
    k = th.Kernel.from_family("gaussian", N, sigma=0.3)
    tc = th.TargetClass.gaussian_psd(N // 2 - 1, beta=0.01)
    decay = 0.5
    meas = th.measure_error(k, tc, th.Schedule(N, 1), decay, rng, n_draws=3)
    et = th.error_terms(k.alloc(), tc.psd, decay, N, N)
    np.testing.assert_allclose(meas, et.total, rtol=1e-9)
    assert et.alias == 0 and et.network == 0


def test_TT8_observation_noise():
    """Eq. 6.6': with observation noise sigma^2 the expected error gains sum a^2 sigma^2 / T / (a + lambda)^2 per mode.

    [Note] Measured against the clean target at the T observation points, the noise term is a^2 sigma^2/(a+lambda)^2
    per mode, without the 1/T factor of eq. 6.6' in this normalization; the 1/T appears only if the error is
    normalized per observation.  Checked here in the sum-over-observations convention.
    """
    rng = np.random.default_rng(1)
    k = th.Kernel.from_family("gaussian", N, sigma=0.3)
    tc = th.TargetClass.gaussian_psd(N // 2 - 1, beta=0.01)
    decay, s2 = 0.5, 0.02
    meas = th.expected_error(k, tc.psd, th.Schedule(N, 1), decay, noise_var=s2)
    a_full = np.abs(k.spectrum()) ** 2
    S_pad = np.concatenate([tc.psd, np.zeros(N)])
    S_full = S_pad[np.abs(np.rint(np.fft.fftfreq(N, 1 / N)).astype(int))]
    pred = np.sum((decay**2 * S_full + a_full**2 * s2) / (a_full + decay) ** 2)
    assert abs(meas - pred) / pred < 1e-9


def test_TT8_network_floor_with_alias_contamination():
    """Target content beyond N/2, T_obs = N, error on the continuous ring.

    The thesis's decomposition (bias + floor) misses the contamination g^2 S(m +- N) of in-band modes by
    out-of-band content; ``sampled_error_exact`` includes it and matches simulation.
    """
    rng = np.random.default_rng(2)
    a = np.zeros(N // 2 + 1)
    a[:20] = 50.0 * np.exp(-0.01 * np.arange(20) ** 2)
    k = th.Kernel.from_alloc(a, N)
    tc = th.TargetClass.gaussian_psd(3 * N // 2, beta=0.002)
    decay = 1.0
    expct = th.expected_error(k, tc.psd, th.Schedule(N, 1), decay, eval_on="fine", n_fine=1024)
    exact = th.sampled_error_exact(a, tc.psd, decay, N, N)
    thesis = th.error_terms(a, tc.psd, decay, N, N)
    assert abs(expct - exact.total) / exact.total < 1e-8
    meas = th.measure_error(k, tc, th.Schedule(N, 1), decay, rng, n_draws=100, eval_on="fine", n_fine=512)
    assert abs(meas - expct) / expct < 0.08  # Monte Carlo sanity check of expected_error
    assert thesis.network > 0
    assert abs(meas - thesis.total) / thesis.total > 1e-3  # discrepancy is real


def test_TT8_subsampling_exact_scaling():
    """T_obs < N with a kernel band-limited to T_obs/2: the per-mode power is (T/N) a_m ('exact' scaling) and
    aliased target energy contaminates in-band modes.  ``sampled_error_exact`` matches simulation; the
    thesis's eq. 6.7-6.8 do not (values in REPORT.md)."""
    rng = np.random.default_rng(3)
    T = 16
    a = np.zeros(N // 2 + 1)
    a[:8] = 200.0 * np.exp(-0.05 * np.arange(8) ** 2)
    k = th.Kernel.from_alloc(a, N)
    tc = th.TargetClass.gaussian_psd(N // 2 - 1, beta=0.01)
    decay = 1.0
    sched = th.Schedule.from_T(N, T)
    expct = th.expected_error(k, tc.psd, sched, decay, eval_on="ring")
    exact = th.sampled_error_exact(a, tc.psd, decay, T, N)
    assert abs(expct - exact.total) / exact.total < 1e-8
    meas = th.measure_error(k, tc, sched, decay, rng, n_draws=200, eval_on="ring")
    assert abs(meas - expct) / expct < 0.05

"""App. G: noisy optimum (G.14-G.16), SNR (G.8), noise-floor estimator (G.9)."""
import numpy as np

import seqnet.thesis as th

N = 128
S = th.TargetClass.gaussian_psd(64, beta=0.01).psd
LAM, K0 = 1.0, 500.0


def test_noiseless_limit():
    r0 = th.water_fill(S, LAM, K0, N=N)
    r = th.water_fill_noisy(S, LAM, K0, 0.0, N, N)
    np.testing.assert_allclose(r.a, r0.a, rtol=1e-9, atol=1e-9)


def test_noisy_optimum_is_optimal():
    """The G.14 allocation beats random budget-preserving perturbations of itself under eq. 6.6'."""
    rng = np.random.default_rng(0)
    nv = 1.0
    r = th.water_fill_noisy(S, LAM, K0, nv, N, N)
    mult = th.mode_multiplicity(N)
    E0 = th.error_terms(r.a, S, LAM, N, N, noise_var=nv).bias
    for _ in range(200):
        a = np.clip(r.a + rng.normal(size=r.a.size) * 0.05 * r.a.max(), 0, None)
        a *= K0 / np.sum(mult * a)
        assert th.error_terms(a, S, LAM, N, N, noise_var=nv).bias >= E0 - 1e-10


def test_noise_lowers_nu_and_does_not_shrink_active_set():
    """Finding: contrary to App. G.7 ('nu* rises, fewer active modes'), noise lowers nu* and the active set does
    not shrink (here it grows by one mode)."""
    r0 = th.water_fill(S, LAM, K0, N=N)
    r = th.water_fill_noisy(S, LAM, K0, 1.0, N, N)
    assert r.nu < r0.nu
    assert r.M_star >= r0.M_star


def test_high_noise_budget_is_slack():
    """When sum lambda S / phi < K0, the optimum leaves budget unspent (a = lambda S / phi, nu = 0)."""
    nv = 100.0
    r = th.water_fill_noisy(S, LAM, K0, nv, N, N)
    phi = nv / N
    np.testing.assert_allclose(r.a, LAM * S[: N // 2 + 1] / phi, rtol=1e-12)
    assert r.nu == 0.0 and r.budget_residual < 0


def test_snr_and_noise_floor_estimator():
    """G.8, and G.9 recovers sigma^2/T from the residual at inactive modes (ridge readout, pure-noise modes)."""
    rng = np.random.default_rng(1)
    tc = th.TargetClass.flat(N // 2 - 1, B=10)
    a = np.zeros(N // 2 + 1); a[:11] = 100.0
    k = th.Kernel.from_alloc(a, N)
    sched = th.Schedule(N, 1)
    X = th.design_matrix(k, sched)
    s2 = 0.04
    est = []
    for _ in range(400):
        y = tc.sample(rng, N, sched.positions)
        y_obs = y + np.sqrt(s2) * rng.normal(size=N)
        e = y_obs - X @ th.ridge_readout(X, y_obs, LAM)
        p = th.spatial_error_power(e, N)
        est.append(th.noise_floor_estimate(p[:-1], a[:-1]))
    # in the sum-over-T convention (error powers of the unitary DFT), inactive modes carry sigma^2 per mode
    assert abs(np.mean(est) - s2) / s2 < 0.05
    np.testing.assert_allclose(th.snr(np.array([2.0]), s2, N), 2.0 / (s2 / N))

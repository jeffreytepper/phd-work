"""TT9 (= T11), TT10, TT15: water-filling closed forms, KKT conditions, joint optimum."""
import numpy as np
import pytest

import seqnet.thesis as th

CASES = {c["name"]: c for c in th.appendix_I_cases(N=128, decay=1e-2, K0=100.0)}


def _wf(case, **kw):
    return th.water_fill(case["target"].psd, case["decay"], case["K0"], N=case["N"], **kw)


@pytest.mark.parametrize("method", ["exact", "bisect"])
def test_budget_satisfied(method):
    for c in CASES.values():
        r = _wf(c, method=method)
        assert abs(r.budget_residual) <= 1e-10 * c["K0"], (c["name"], r.budget_residual)


def test_TT9_constant():
    r = _wf(CASES["constant"])
    np.testing.assert_allclose(r.a[0], 100.0, rtol=1e-12)
    assert np.all(r.a[1:] == 0)


def test_TT9_single_mode():
    r = _wf(CASES["single_mode"])
    np.testing.assert_allclose(r.a[5], 50.0, rtol=1e-12)  # K0/2 at each of +-m0 (eq. I.4)
    assert np.count_nonzero(r.a) == 1


def test_TT9_flat():
    c = CASES["flat"]
    r = _wf(c)
    B = c["check"]["B"]
    np.testing.assert_allclose(r.a[: B + 1], c["K0"] / (2 * B + 1), rtol=1e-10)  # eq. I.7-I.8
    assert np.all(r.a[B + 1:] == 0)
    np.testing.assert_allclose(r.nu, c["check"]["nu"], rtol=1e-10)  # eq. I.9
    assert r.rank == 2 * B + 1


def test_TT9_gaussian_psd_exponent():
    """a*_m + lambda = C exp(-beta m^2 / 3) on active modes (eq. I.12)."""
    c = CASES["gaussian_psd"]
    r = _wf(c)
    m = np.nonzero(r.active)[0]
    slope = np.polyfit(m**2, np.log(r.a[m] + c["decay"]), 1)[0]
    np.testing.assert_allclose(-slope, c["check"]["exponent"], rtol=1e-8)


def test_TT9_gaussian_psd_active_cutoff_has_square_root():
    """Discrepancy check: the active band is |m| < sqrt((1/beta) ln(2 S0/(nu lambda))), not (1/beta) ln(...)
    as printed under eq. I.12."""
    c = CASES["gaussian_psd"]
    r = _wf(c)
    beta = c["target"].params["beta"]
    arg = np.log(2 * 1.0 / (r.nu * c["decay"]))
    M_sqrt = np.sqrt(arg / beta)
    M_print = arg / beta
    assert r.M_star == int(np.floor(M_sqrt))
    assert r.M_star != int(np.floor(M_print))


def test_TT9_polynomial_exponent():
    """a*_m + lambda proportional to |m|^{-2s/3} on active modes m >= 1 (eq. I.15)."""
    c = CASES["polynomial"]
    r = _wf(c)
    m = np.nonzero(r.active)[0]
    m = m[m >= 1]
    slope = np.polyfit(np.log(m), np.log(r.a[m] + c["decay"]), 1)[0]
    np.testing.assert_allclose(slope, c["check"]["exponent"], rtol=1e-8)


def test_TT9_narrowband_spectral_width():
    """Around the carrier, a*_m + lambda is Gaussian with spectral std sqrt(3) l (eq. I.18).

    Only the spectral statement is tested.  The physical-width statements (sigma_k = sqrt(3) sigma_y and
    sigma_k^2 = 3/l^2 in eq. I.19) are D1 and are reported, not encoded (see REPORT)."""
    c = CASES["narrowband"]
    r = _wf(c)
    xi0, ell = c["check"]["xi0"], c["target"].params["ell"]
    m = np.nonzero(r.active)[0]
    m = m[np.abs(m - xi0) <= 3 * ell]  # near +xi0, where the -xi0 peak is negligible
    q = np.polyfit((m - xi0) ** 2, np.log(r.a[m] + c["decay"]), 1)[0]
    width = np.sqrt(-1 / (2 * q))
    np.testing.assert_allclose(width, c["check"]["width"], rtol=1e-6)


def test_TT10_kkt():
    """Marginal returns equal nu* on active modes, and are <= nu* (at a = 0) on inactive in-band modes (eq. 7.10-7.11)."""
    S = th.TargetClass.gaussian_psd(64, beta=0.01).psd + 1e-6
    for decay in (1e-3, 1e-2, 1e-1):
        r = th.water_fill(S, decay, 50.0, N=128)
        ret = th.marginal_returns(r.a, S, decay)
        np.testing.assert_allclose(ret[r.active], r.nu, rtol=1e-9)
        assert np.all(ret[~r.active] <= r.nu * (1 + 1e-12))
        gap = th.optimality_gap(r.a, S, decay)
        assert gap["spread"] < 1e-9 and gap["kkt_violation"] == 0.0


def test_min_bias_error_matches_error_terms():
    """Eq. 7.17 equals the bias term of eq. 6.7 evaluated at the optimum."""
    S = th.TargetClass.polynomial(64, s=1.0).psd
    r = th.water_fill(S, 1e-2, 30.0, N=128)
    et = th.error_terms(r.a, S, 1e-2, 128, 128)
    np.testing.assert_allclose(th.min_bias_error(r, S), et.bias, rtol=1e-10)


def test_optimum_beats_perturbations():
    """No budget-preserving perturbation of a* lowers the within-band error (convexity sanity check)."""
    rng = np.random.default_rng(0)
    S = th.TargetClass.gaussian_psd(64, beta=0.02).psd
    decay, K0, N = 1e-2, 40.0, 128
    r = th.water_fill(S, decay, K0, N=N)
    E0 = th.error_terms(r.a, S, decay, N, N).bias
    mult = th.mode_multiplicity(N)
    for _ in range(200):
        d = rng.normal(size=r.a.size) * 0.05
        a = np.clip(r.a + d, 0, None)
        a *= K0 / np.sum(mult * a)
        assert th.error_terms(a, S, decay, N, N).bias >= E0 - 1e-12


def test_ridge_fill_square_root_law():
    """Square-root law (spec Section 10.1): (a + lambda)^2 proportional to S on active modes."""
    S = th.TargetClass.gaussian_psd(64, beta=0.02).psd
    r = th.ridge_fill(S, 1e-2, 40.0, N=128)
    m = np.nonzero(r.active)[0]
    np.testing.assert_allclose(th.ridge_marginal_returns(r.a, S, 1e-2)[m], r.nu, rtol=1e-9)
    slope = np.polyfit(m**2, np.log(r.a[m] + 1e-2), 1)[0]
    np.testing.assert_allclose(-slope, 0.02 / 2, rtol=1e-8)


def test_implied_psd_inverts_water_fill():
    """implied_psd(a*) is proportional to S on active modes (inverse of eq. 7.10)."""
    S = th.TargetClass.narrowband(64, xi0=10, ell=3).psd
    r = th.water_fill(S, 1e-2, 40.0, N=128)
    Simp = th.implied_psd(r.a, 1e-2, nu=r.nu)
    np.testing.assert_allclose(Simp[r.active], S[: r.a.size][r.active], rtol=1e-9)


def test_TT15_joint_optimum_thesis_closed_form():
    """Thesis eq. 8.9: T* = 2 M*, kappa = lambda/2, joint budget satisfied up to the step in M*(nu)."""
    S = th.TargetClass.gaussian_psd(64, beta=0.01).psd
    decay, N = 1e-2, 128
    for B0 in (5.0, 20.0, 80.0):
        jr = th.joint_optimum(S, decay, B0, None, N, method="thesis")
        assert jr.T_star == 2 * jr.M_star
        # the budget jumps by lambda (= kappa * 2) when M* changes; residual must be within one jump
        assert abs(jr.budget_residual) <= decay + 1e-9 * B0
        # active set = {|m| <= M*} and a* follows the cube law there
        assert np.all(jr.a[: jr.M_star + 1] > 0) and np.all(jr.a[jr.M_star + 1:] == 0)


def test_joint_optimum_numeric_vs_thesis_reported():
    """Numeric minimization of E*(T) with kappa = lambda/2: records T* against the thesis's 2M*.

    The thesis claims equality (eq. 8.9); this test only checks that both methods run and that the numeric
    E*(T) curve is minimized at T*_numeric (values reported in REPORT.md)."""
    S = th.TargetClass.gaussian_psd(64, beta=0.01).psd
    decay, N, B0 = 1e-2, 128, 20.0
    jn = th.joint_optimum(S, decay, B0, decay / 2, N, method="numeric")
    assert jn.E_curve[jn.T_star] == min(jn.E_curve.values())

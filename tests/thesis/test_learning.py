"""TT11, TT12 (= T12), TT13 (= T13): observable gradient, kernel-rule convergence to the cube / square-root laws."""
import numpy as np
import pytest

import seqnet.thesis as th

N = 32


def _gauss_alloc(N, K0, sigma=0.6):
    k = th.Kernel.from_family("gaussian", N, sigma=sigma)
    a = k.alloc()
    return a * K0 / np.sum(th.mode_multiplicity(N) * a)


def test_TT11_observable_gradient_equals_analytic():
    """At ridge equilibrium, -2(1-g)/lambda * |eps_hat_m|^2 (eq. 9.6), with eps_hat from the simulated readout
    errors, equals -2 lambda^2 S / (a + lambda)^3 (eq. 9.5)."""
    rng = np.random.default_rng(0)
    decay = 0.3
    a = _gauss_alloc(N, 200.0)
    k = th.Kernel.from_alloc(a, N)
    tc = th.TargetClass.gaussian_psd(N // 2 - 1, beta=0.02)
    coeffs = tc.coeffs(rng)
    sched = th.Schedule(N, 1)
    y = tc.evaluate(coeffs, N, sched.positions)
    X = th.design_matrix(k, sched)
    e = y - X @ th.ridge_readout(X, y, decay)
    p = th.spatial_error_power(e, N)
    S = np.concatenate([np.abs(coeffs) ** 2, [0.0]])
    obs = th.observable_gradient(a, p, decay)
    ana = th.analytic_gradient(a, S, decay)
    np.testing.assert_allclose(obs[:-1], ana[:-1], rtol=1e-8, atol=1e-14)


def _run(rule_cls, law, decay=0.1, K0=150.0, n_steps=300, eta=0.3, budget="project", precondition="newton",
         target_mode="fixed", seed=0, readout="exact"):
    tc = th.TargetClass.gaussian_psd(N // 2, beta=0.03)
    S = tc.psd.copy()
    S[-1] = 0.0  # Nyquist mode: drop (its power is ill-defined on the ring)
    tc = th.TargetClass(psd=S, name="gaussian_psd")
    ref = th.water_fill(S, decay, K0, N=N, law=law)
    rule = rule_cls(eta=eta, decay=decay, N=N, K0=K0, budget=budget, precondition=precondition,
                    nu=ref.nu if budget == "fixed" else 0.0, eta_nu=0.0)
    ls = th.LearningSystem(N=N, decay=decay, K0=K0, target=tc, a0=_gauss_alloc(N, K0), rule=rule,
                           readout=readout, target_mode=target_mode, seed=seed)
    h = ls.run(n_steps)
    return ls, h, ref


@pytest.mark.parametrize("budget", ["project", "fixed"])
def test_TT12_thesis_rule_converges_to_cube_law(budget):
    """ThesisKernelRule (eq. 9.10), Newton-preconditioned (see learning.py), converges to eq. 7.13:
    allocation within 2% on active modes, exact active set.  The plain rule is tested separately below."""
    ls, h, ref = _run(th.ThesisKernelRule, "cube", budget=budget)
    a = ls.a
    act = ref.active
    np.testing.assert_array_equal(a > 1e-9 * a.max(), act)
    np.testing.assert_allclose(a[act], ref.a[act], rtol=0.02)


def test_TT12_thesis_rule_dual_budget():
    """Default budget enforcement (spec 6.9 option ii): nu updated toward the budget."""
    decay, K0 = 0.1, 150.0
    tc = th.TargetClass.gaussian_psd(N // 2, beta=0.03)
    S = tc.psd.copy()
    S[-1] = 0.0
    tc = th.TargetClass(psd=S)
    ref = th.water_fill(S, decay, K0, N=N)
    rule = th.ThesisKernelRule(eta=0.3, decay=decay, N=N, K0=K0, budget="dual", nu=0.5 * ref.nu,
                               eta_nu=0.05 * ref.nu / K0, precondition="newton")
    ls = th.LearningSystem(N=N, decay=decay, K0=K0, target=tc, a0=_gauss_alloc(N, K0), rule=rule)
    ls.run(3000)
    np.testing.assert_array_equal(ls.a > 1e-9 * ls.a.max(), ref.active)
    np.testing.assert_allclose(ls.a[ref.active], ref.a[ref.active], rtol=0.02)
    np.testing.assert_allclose(rule.nu, ref.nu, rtol=0.02)


def test_TT13_ridge_rule_converges_to_square_root_law():
    """RidgeObjectiveRule (spec Section 10.1) converges to a = sqrt(lambda S / nu) - lambda (within 2%)."""
    ls, h, ref = _run(th.RidgeObjectiveRule, "sqrt", budget="project")
    act = ref.active
    np.testing.assert_array_equal(ls.a > 1e-9 * ls.a.max(), act)
    np.testing.assert_allclose(ls.a[act], ref.a[act], rtol=0.02)
    cube = th.water_fill(ls.target.psd, 0.1, 150.0, N=N)
    assert np.max(np.abs(ls.a - cube.a)) > 0.05 * cube.a.max()  # the two laws are distinguishable here


def test_ensemble_and_fixed_targets_reach_same_fixed_point():
    """Spec 6A.5: 'ensemble' (new random-phase target each kernel step) and 'fixed' modes agree."""
    ls_f, _, ref = _run(th.ThesisKernelRule, "cube", target_mode="fixed")
    ls_e, _, _ = _run(th.ThesisKernelRule, "cube", target_mode="ensemble", seed=5)
    np.testing.assert_allclose(ls_e.a[ref.active], ls_f.a[ref.active], rtol=1e-6)


def test_plain_thesis_rule_is_stiff():
    """Finding: eq. 9.10 with one eta.  Stability near a = 0 needs eta < lambda^2 / (3 S_max); at that eta the
    slowest active mode relaxes at rate ~ eta * 3 nu / (a + lambda), i.e. >1e6 steps here.  The rule is
    stable and descends the error at such eta, but after 2000 steps it is still far from eq. 7.13; a larger
    eta destabilizes it (collapse onto few modes)."""
    decay, K0 = 0.1, 150.0
    S = th.TargetClass.gaussian_psd(N // 2, beta=0.03).psd.copy()
    S[-1] = 0.0
    ref = th.water_fill(S, decay, K0, N=N)
    mult = th.mode_multiplicity(N)

    def run(eta, n):
        rule = th.ThesisKernelRule(eta=eta, decay=decay, N=N, K0=K0, budget="project")
        a = _gauss_alloc(N, K0)
        E = []
        for _ in range(n):
            E.append(th.error_terms(a, S, decay, N, N).bias)
            a = rule.step(decay**2 * S / (a + decay) ** 2, a)
        return a, np.array(E)

    a_s, E_s = run(decay**2 / (3 * S.max()) * 0.9, 2000)
    assert np.all(np.diff(E_s) <= 1e-12 * E_s[0])  # stable, monotone descent
    assert np.max(np.abs(a_s - ref.a)[ref.active] / ref.a[ref.active]) > 0.1  # but far from the optimum
    a_u, _ = run(0.3 * decay / ref.nu, 500)
    assert np.sum(a_u > 0) < ref.active.sum() // 2  # large eta: collapse
    assert abs(np.sum(mult * a_u) - K0) < 1e-6 * K0

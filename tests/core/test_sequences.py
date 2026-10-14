"""seqnet.sequences: Braun targets unchanged, the Lehr et al. (2024) family, datasets, and the config/cache migration."""
import json
from pathlib import Path

import numpy as np
import pytest

from seqnet.sequences import (BRAUN_PERIOD, PARTICIPATION, PROFILES, SEQUENCES, TIMECOURSES, LehrRingNetwork,
                              MorphedCosine, RaisedCosine, RandomSilencing, RecordedSequence, SampledSequence,
                              Switching, TimeScaled, WrappedGaussian, braun_sequence, build_sequence, lehr_bump, logit,
                              logit_dt, measures, natural_period, training_target)
from seqnet.training.config import BRAUN_SYNTHETIC, ForceConfig, legacy_layout

TC_SPECS = [
    {"kind": "constant", "value": 1.7},
    {"kind": "step", "before": 1.0, "after": 3.0, "t_step": 0.4},
    {"kind": "exponential", "baseline": 1.0, "scale": 2.0, "rate": -1.5},
    {"kind": "exponential", "baseline": 0.0, "scale": 1.0, "rate": 0.7},
    {"kind": "sigmoid", "low": 1.0, "high": 4.0, "t_mid": 0.8, "width": 0.2},
    {"kind": "parabola", "vertex": 3.0, "curvature": -2.0, "t_vertex": 0.7},
    {"kind": "sinusoid", "mean": 2.0, "amplitude": 0.5, "period": 0.9, "phase": 0.3},
    {"kind": "gaussian_pulse", "baseline": 0.2, "height": 1.5, "t_peak": 0.6, "width": 0.15},
    {"kind": "scaled", "factor": -3.0, "course": {"kind": "sinusoid", "mean": 1.0, "amplitude": 0.4, "period": 0.5}},
]


# ---------------------------------------------------------------------------------------------- Braun members
def test_braun_sequence_matches_methods_formula():
    """f_n(t) = a sum_m exp(-(t - n T/N - m T)^2 / (2 sigma^2)) + b (Braun eq. 4-5), independently coded."""
    N, T, a, s, b = 37, 8.5, 0.6, 1.2, 0.15
    t = np.linspace(-5, 30, 301)
    u = t[None, :] - (np.arange(N) * T / N)[:, None]
    ref = a * sum(np.exp(-((u - m * T) ** 2) / (2 * s**2)) for m in range(-12, 13)) + b
    seq = braun_sequence(N, T, a, s, b)
    np.testing.assert_allclose(seq.rates(t), ref, atol=1e-13)
    refdt = a * sum(-(u - m * T) / s**2 * np.exp(-((u - m * T) ** 2) / (2 * s**2)) for m in range(-12, 13))
    np.testing.assert_allclose(seq.rates_dt(t), refdt, atol=1e-12)
    # stationary view: neuron 0's bump, period T, Braun Suppl. eq. 3 coefficients
    w = seq.stationary()
    assert w.period == pytest.approx(T) and w.omega == pytest.approx(2 * np.pi / T)
    np.testing.assert_allclose(w.profile(t), ref[0], atol=1e-13)
    k = np.arange(-6, 7)
    ana = a * s * np.sqrt(2 * np.pi) / T * np.exp(-(s**2) * (k * 2 * np.pi / T) ** 2 / 2) + np.where(k == 0, b, 0.0)
    np.testing.assert_allclose(w.fourier_coeffs(k), ana, atol=1e-15)


def test_spec_and_object_build_the_same_sequence():
    t = np.linspace(0, 10, 50)
    np.testing.assert_array_equal(build_sequence(dict(BRAUN_SYNTHETIC)).rates(t), braun_sequence().rates(t))
    seq = braun_sequence()
    assert build_sequence(seq) is seq
    assert {"braun", "traveling_bump", "lehr_bump", "lehr_network", "time_scaled", "bat", "mec", "npz"} <= set(SEQUENCES.kinds())


# ---------------------------------------------------------------------------------------------- parts
@pytest.mark.parametrize("spec", TC_SPECS, ids=lambda s: s["kind"])
def test_timecourse_derivative_and_integral(spec):
    tc = TIMECOURSES.build(spec)
    t = np.linspace(0.013, 1.5, 4001)
    h = 1e-6
    smooth = spec["kind"] != "step"
    if smooth:
        np.testing.assert_allclose(tc.derivative(t), (tc(t + h) - tc(t - h)) / (2 * h), atol=1e-5 * (1 + np.abs(tc(t)).max()))
    grid = np.linspace(0, 1.5, 300001)
    cum = np.concatenate([[0], np.cumsum((tc(grid[1:]) + tc(grid[:-1])) / 2 * np.diff(grid))])
    np.testing.assert_allclose(tc.integral(t), np.interp(t, grid, cum), atol=1e-4 if not smooth else 1e-7)
    assert tc.integral(np.zeros(1))[0] == pytest.approx(0.0, abs=1e-14)


def test_profiles():
    N, d = 1000, np.arange(1000) - 500.0
    rc = RaisedCosine()
    np.testing.assert_allclose(MorphedCosine(np.inf).value(d, 250, N), rc.value(d, 250, N))
    rect = MorphedCosine(0.0).value(d, 250, N)
    assert set(np.unique(rect)) == {0.0, 1.0} and rect.sum() == 249  # support |d| < N/(2k), k = 4
    for p in (rc, MorphedCosine(0.3), WrappedGaussian()):
        assert p.value(np.zeros(1), 100.0, N)[0] == pytest.approx(1.0)
    # sinusoid around the whole ring at k = 1 (Lehr Fig. 1a)
    np.testing.assert_allclose(rc.value(d, N, N), 0.5 * (1 + np.cos(2 * np.pi * d / N)), atol=1e-12)
    # closed-form Fourier series against the DFT of the samples
    k = np.arange(-8, 9)
    for p, w in ((rc, 300.0), (WrappedGaussian(), 40.0)):
        x = np.arange(N, dtype=float)
        dft = np.fft.fft(p.value(x, w, N)) / N
        np.testing.assert_allclose(p.fourier_coeffs(k, w, N), dft[k % N].real, atol=1e-6)
    # delta and width derivatives against finite differences
    for p in (rc, MorphedCosine(0.5), WrappedGaussian()):
        dd = np.linspace(-140, 140, 113) + 0.37
        h = 1e-5
        np.testing.assert_allclose(p.d_delta(dd, 120.0, N), (p.value(dd + h, 120.0, N) - p.value(dd - h, 120.0, N)) / (2 * h), atol=1e-6)
        np.testing.assert_allclose(p.d_width(dd, 120.0, N), (p.value(dd, 120.0 + h, N) - p.value(dd, 120.0 - h, N)) / (2 * h), atol=1e-6)
    assert set(PROFILES.kinds()) == {"wrapped_gaussian", "raised_cosine", "morphed_cosine"}


def test_participation():
    m = RandomSilencing(0.3, seed=7).mask(200)
    assert m.sum() == 140 and np.array_equal(m, RandomSilencing(0.3, seed=7).mask(200))
    assert not np.array_equal(m, RandomSilencing(0.3, seed=8).mask(200))
    sw = PARTICIPATION.build({"kind": "switching", "switch_times": [1.0],
                              "stages": [{"kind": "random_silencing", "p_inh": 0.5, "seed": 1},
                                         {"kind": "random_silencing", "p_inh": 0.5, "seed": 2}]})
    g = sw.gains(50, np.array([0.0, 0.99, 1.0, 2.0]))
    np.testing.assert_array_equal(g[:, 0], RandomSilencing(0.5, 1).mask(50))
    np.testing.assert_array_equal(g[:, 2], RandomSilencing(0.5, 2).mask(50))
    with pytest.raises(ValueError):
        Switching(stages=({"kind": "all_active"},), switch_times=(1.0,))


# ---------------------------------------------------------------------------------------------- Lehr family
def test_modulated_bump_has_exact_derivative_and_speed():
    """Gain-modulated speed, pulsed width and amplitude, morphed shape, sparse: rates_dt equals the finite difference
    and the measured bump speed follows v(t) (Lehr 2024 Fig. 2-4)."""
    gain = {"kind": "sigmoid", "low": 0.5, "high": 1.5, "t_mid": 0.5, "width": 0.1}
    seq = lehr_bump(N=400, k=3, eps=0.7, period=1.0, gain=gain, p_inh=0.2, seed=4,
                    width_scale={"kind": "gaussian_pulse", "baseline": 1.0, "height": 0.4, "t_peak": 0.6, "width": 0.1},
                    amplitude={"kind": "sinusoid", "mean": 0.8, "amplitude": 0.1, "period": 0.5})
    t = np.linspace(0.05, 1.0, 97)
    h = 1e-6
    fd = (seq.rates(t + h) - seq.rates(t - h)) / (2 * h)
    np.testing.assert_allclose(seq.rates_dt(t), fd, atol=1e-4 * np.abs(fd).max())
    dense = lehr_bump(N=400, k=3, period=1.0, gain=gain)
    tt = np.linspace(0, 1, 2001)
    v = measures.bump_speed(dense.rates(tt), tt[1] - tt[0])
    np.testing.assert_allclose(v, 400 * TIMECOURSES.build(gain)(0.5 * (tt[1:] + tt[:-1])), rtol=2e-3)
    with pytest.raises(ValueError):
        seq.stationary()


def _pcs(R, m):
    X = R - R.mean(1, keepdims=True)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :m], s


def test_lehr_manifold_geometry():
    """Lehr et al. 2024 Fig. 1: a k = 1 bump lies on a circle (2 PCs explain all variance); narrower bumps add
    harmonic dimensions; silencing rotates the subspace by about arccos(1 - p_inh) (Fig. 1j)."""
    t = np.arange(1000) / 1000 * BRAUN_PERIOD  # one cycle
    _, s = _pcs(lehr_bump(N=1000, k=1).rates(t), 6)
    assert (s[:2] ** 2).sum() / (s**2).sum() > 1 - 1e-12
    dims = []
    for k in (1, 2, 4):
        _, s = _pcs(lehr_bump(N=1000, k=k).rates(t), 40)
        dims.append(int(np.searchsorted(np.cumsum(s**2) / (s**2).sum(), 0.9) + 1))
    assert dims[0] < dims[1] < dims[2]
    for p in (0.3, 0.5):
        A, _ = _pcs(lehr_bump(N=1000, p_inh=p, seed=0).rates(t), 2)
        B, _ = _pcs(lehr_bump(N=1000, p_inh=p, seed=1).rates(t), 2)
        ang = np.degrees(np.arccos(np.linalg.svd(A.T @ B, compute_uv=False)[0]))
        assert abs(ang - np.degrees(np.arccos(1 - p))) < 4


def test_lehr_network_controls():
    """Lehr et al. 2024: gain S raises the speed (Fig. 3) up to one shift per step (Fig. S3, about 38 rot/s at
    shift 0.04, dt = 1 ms); E - I sets the bump width (Fig. 2); silenced neurons stay silent (Fig. 5)."""
    def run(**kw):
        q = LehrRingNetwork(N=1000, duration=0.6, **kw).simulate()
        return q, measures.bump_speed(q.samples[:, 300:], q.dt).mean() / 1000

    _, v1 = run(gain=1.0)
    _, v15 = run(gain=15.0)
    _, vmax = run(gain=1000.0)
    assert v1 < v15 < vmax and vmax == pytest.approx(0.04 / 1e-3 * 0.975, rel=0.05)
    width = [(run(gain=5.0, drive=d)[0].samples[:, -1] > 0.01).mean() for d in (-1.0, 0.0, 1.0)]
    assert width[0] < width[1] < width[2]
    p = RandomSilencing(0.8, 3)
    q, v = run(gain=5.0, participation=p)
    assert q.samples[p.mask(1000) == 0].max() == 0.0 and v > 1.0
    # weights scale with N: a 251-neuron ring runs like the paper's 1000-neuron one
    q251 = LehrRingNetwork(N=251, duration=0.6, gain=15.0).simulate()
    assert measures.bump_speed(q251.samples[:, 300:], q251.dt).mean() / 251 == pytest.approx(v15, rel=0.02)
    # the literal unbounded-current reading does not speed up (see seqnet.sequences.network)
    _, a = run(gain=1.0, bounded_state=False)
    _, b = run(gain=15.0, bounded_state=False)
    assert abs(a - b) < 0.05


# ---------------------------------------------------------------------------------------------- sampling
def test_sampled_and_recorded_snippets():
    rng = np.random.default_rng(0)
    R = rng.random((5, 101))
    s = SampledSequence(R, 0.1)
    assert s.duration == pytest.approx(10.0)
    np.testing.assert_allclose(s.rates(np.array([0.0, 0.25])), np.stack([R[:, 0], 0.5 * (R[:, 2] + R[:, 3])], 1))
    with pytest.raises(ValueError):
        s.rates(np.array([10.5]))
    snip = s.draw_snippet(np.random.default_rng(1), np.arange(20) * 0.3, start_max=100.0)
    assert 0 <= snip.t0 <= 10.0 - 19 * 0.3
    per = SampledSequence(R, 0.1, periodic=True)
    np.testing.assert_allclose(per.rates(np.array([10.1 + 3.0])), per.rates(np.array([3.0])))
    # recorded: one bin per update from a random window (Braun lines 913-925)
    rec = RecordedSequence(R, 0.129, snippet_bins=30)
    sn = rec.draw_snippet(np.random.default_rng(2), np.arange(30) * 0.1296, start_max=0.0)
    b0 = int(round(sn.t0 / 0.129))
    np.testing.assert_array_equal(sn.rates, R[:, b0:b0 + 30])
    assert sn.rates_dt is None
    y = training_target(sn, "supplement", 0.3, 0.01)
    np.testing.assert_allclose(y, logit(sn.rates) + 0.3 * np.gradient(logit(sn.rates), 0.1296, axis=1))


def test_training_target_conventions():
    seq = braun_sequence(N=20)
    snip = seq.draw_snippet(np.random.default_rng(0), np.arange(10) * 0.13, 50.0)
    np.testing.assert_array_equal(training_target(snip, "methods", 0.3), logit(snip.rates))
    np.testing.assert_allclose(training_target(snip, "supplement", 0.3),
                               logit(snip.rates) + 0.3 * logit_dt(snip.rates, seq.rates_dt(snip.times)))


# ---------------------------------------------------------------------------------------------- config and cache
def test_legacy_layout_round_trip():
    old = {**{k: v for k, v in ForceConfig().as_dict().items() if k != "sequence"},
           "N": 166, "T": 8.5, "amp": 0.83, "width": 1.72, "offset": 0.05, "data": None}
    cfg = ForceConfig.from_dict(old)
    assert cfg.sequence == {"kind": "braun", "N": 166, "T": 8.5, "amp": 0.83, "width": 1.72, "offset": 0.05}
    assert legacy_layout(cfg.as_dict()) == old
    data = {**old, "N": 501, "amp": 0.5, "width": 1.7, "offset": 0.0, "data": {"kind": "bat", "path": "x.mat"}}
    assert ForceConfig.from_dict(data).sequence == {"kind": "bat", "path": "x.mat"}
    assert legacy_layout(ForceConfig.from_dict(data).as_dict()) == data
    new = ForceConfig(sequence={"kind": "lehr_bump", "N": 100}).as_dict()
    assert legacy_layout(new) == new


def test_cached_runs_keep_their_directory():
    """Every cached run directory is still the one run_condition resolves for its stored config."""
    from seqnet.experiments.runner import repo_root, run_hashes

    base = repo_root() / "results" / "wp0" / "runs"
    metas = sorted(base.glob("*/meta.json")) if base.exists() else []
    if not metas:
        pytest.skip("no cached runs")
    for m in metas:
        meta = json.loads(m.read_text(encoding="utf-8"))
        assert any(m.parent.name.endswith("-" + h) for h in run_hashes(meta["config"], meta["n_realizations"])), m


def test_force_trains_on_lehr_sequences():
    """FORCE consumes any sequence kind: a sparse, morphed, gain-modulated Lehr bump and a simulated ring network."""
    from seqnet.training.rls import train_realization

    bump = {"kind": "lehr_bump", "N": 40, "k": 2, "eps": 1.0, "p_inh": 0.25, "amplitude": 0.8,
            "gain": {"kind": "sinusoid", "mean": 1.0, "amplitude": 0.4, "period": 4.0}}
    net = {"kind": "time_scaled", "period": 8.5,
           "sequence": {"kind": "lehr_network", "N": 40, "duration": 30.0, "dt": 0.01, "gain": 1.0}}
    for seq in (bump, net):
        cfg = ForceConfig(sequence=seq, n_train=2, n_test=1, steps_per_iter=120, snippet_start_max=10.0)
        out = train_realization(cfg, 0)
        assert out["J"].shape == (40, 40) and np.isfinite(out["corr"]).all()
        assert np.array_equal(out["f_target"], build_sequence(seq).rates(out["t0"] + np.arange(8) * 16 * cfg.dt))


def test_time_scaling():
    """One clock for the family: lehr_bump defaults to Braun's 8.5-s cycle; TimeScaled stretches time and derivative;
    a network rescaled to a period has that period; burn-in removes the start-up transient."""
    assert natural_period(lehr_bump(N=200)) == pytest.approx(BRAUN_PERIOD)
    assert natural_period(lehr_bump(N=200, period=2.0, gain=3.0)) == pytest.approx(2.0 / 3.0)
    base = lehr_bump(N=200, k=2, period=1.0, gain={"kind": "sinusoid", "mean": 1.0, "amplitude": 0.3, "period": 0.7})
    sc = TimeScaled(base, 4.0)
    t = np.linspace(0, 3, 40)
    np.testing.assert_array_equal(sc.rates(t), base.rates(t / 4))
    np.testing.assert_allclose(sc.rates_dt(t), base.rates_dt(t / 4) / 4)
    net = {"kind": "lehr_network", "N": 300, "duration": 6.0, "gain": 2.0}
    q = build_sequence({"kind": "time_scaled", "period": BRAUN_PERIOD, "sequence": net})
    assert q.duration == pytest.approx(6.0 * q.factor)
    assert natural_period(q, np.linspace(0, q.duration, 6001)) == pytest.approx(BRAUN_PERIOD, rel=1e-6)
    v = measures.bump_speed(build_sequence(net).samples[:, :200], 1e-3)
    assert np.std(v) < 0.02 * np.mean(v)  # settled from the first sample
    with pytest.raises(ValueError):
        build_sequence({"kind": "time_scaled", "sequence": net})


# ---------------------------------------------------------------------------------------------- pulsed testbed helpers
def test_local_profiles_average_to_the_mean_profile():
    from seqnet import conventions as cv
    from seqnet.analysis import neuron_profiles as npf

    rng = np.random.default_rng(3)
    W = rng.normal(size=(2, 31, 31))
    for direction in ("in", "out"):
        P = npf.local_profiles(W, direction)
        np.testing.assert_allclose(P.mean(-2), cv.profile_from_matrix(W), atol=1e-12)
    np.testing.assert_array_equal(npf.local_profiles(W[0])[4, 3], W[0, 4, 7])
    np.testing.assert_array_equal(npf.local_profiles(W[0], "out")[4, 3], W[0, 1, 4])
    C = cv.circulant_from_profile(rng.normal(size=31))  # every neuron has the same profile
    P = npf.local_profiles(C)
    np.testing.assert_allclose(P, np.broadcast_to(P[0], P.shape), atol=1e-12)
    g = npf.neuron_groups(30.4, 31, {"a": (-1, 2)})
    np.testing.assert_array_equal(g["a"], [29, 30, 0, 1])
    d, p = npf.window(P[:1], 3)
    s = npf.profile_summary(d, p[0])
    assert s["self"] == pytest.approx(C[0, 0]) and -1 <= s["asymmetry"] <= 1


def test_pulsed_specs_and_snippets():
    from seqnet.experiments import pulsed as pz

    base = {"kind": "lehr_bump", "N": 200, "k": 4}
    assert pz.pulsed_spec(base, None) == base
    pulse = pz.Pulse("gain", 1.0, 4.25, 0.5)
    spec = pz.pulsed_spec(base, pulse)
    assert spec["gain"] == {"kind": "gaussian_pulse", "baseline": 1.0, "height": 1.0, "t_peak": 4.25, "width": 0.5}
    assert "gain" not in base  # the base spec is not modified
    assert pz.pass_duration(build_sequence(base)) == pytest.approx(BRAUN_PERIOD, abs=1e-3)
    # a speed pulse of area h w sqrt(2 pi) shortens the pass by that many base periods' worth of time
    expected = BRAUN_PERIOD - 1.0 * 0.5 * np.sqrt(2 * np.pi)
    assert pz.pass_duration(build_sequence(spec)) == pytest.approx(expected, abs=1e-3)
    assert pz.pulse_neuron(build_sequence(base), 4.25) == pytest.approx(100, abs=0.5)
    # network pulses are converted to model time
    net = {"kind": "time_scaled", "period": 8.5, "sequence": {"kind": "lehr_network", "N": 100, "duration": 2.0}}
    ns = pz.pulsed_spec(net, pz.Pulse("drive", -0.5, 4.25, 1.0))
    f = ns["factor"]
    assert "period" not in ns and ns["sequence"]["drive"]["t_peak"] == pytest.approx(4.25 / f)
    cfg = pz.condition_config(spec, n_train=1, n_test=1)
    assert cfg.snippet_start_max == 0.0
    assert abs((cfg.steps_per_iter - 1) * cfg.dt - expected) < cfg.dt


def test_profile_change_views():
    from seqnet import conventions as cv
    from seqnet.analysis import neuron_profiles as npf
    from seqnet.spectral import braun_map as bm

    rng = np.random.default_rng(5)
    W = rng.normal(size=(40, 40))
    P = npf.local_profiles(W)
    lam = npf.local_spectra(P, drop_self=False)
    np.testing.assert_allclose(lam.mean(0), cv.eigs_from_profile(cv.profile_from_matrix(W)), atol=1e-10)
    # a circulant supporting Braun's sequence (supplement convention) implies exactly its speed N / T on every row
    N, tau = 64, 0.3224
    wave = braun_sequence(N=N, amp=0.5, width=1.0, offset=0.1).stationary()
    cp = bm.profile_from_mode_eigs(bm.required_eigs(wave, tau, K_max=4), N)
    lamc = npf.local_spectra(npf.local_profiles(cv.circulant_from_profile(cp)), drop_self=False)
    for k in (1, 2, 3, [1, 2, 3]):
        np.testing.assert_allclose(npf.implied_speed(lamc, k, tau, N), N / wave.period, rtol=1e-6)
    # windowed spectra ignore a change far from each row's own neurons (a column stripe)
    C = cv.circulant_from_profile(cp)
    Cs = C.copy(); Cs[:, 40] += 0.3
    lam_w, lam_ws = (npf.local_spectra(npf.local_profiles(X), d_max=10) for X in (C, Cs))
    far = np.abs((np.arange(N) - 40 + N // 2) % N - N // 2) > 10
    np.testing.assert_allclose(lam_ws[far], lam_w[far], atol=1e-12)
    # dilation: a profile stretched by 1.3 is recovered
    d = np.arange(-70, 71)
    ref = np.exp(-((d + 30) ** 2) / 200) + 0.8 * np.exp(-((d - 5) ** 2) / 50)
    fit = npf.dilation_fit(d, np.stack([2.0 * np.interp(d / 1.3, d, ref), ref]), ref)
    np.testing.assert_allclose(fit["scale"], [1.3, 1.0], rtol=0.02)
    np.testing.assert_allclose(fit["amplitude"][0], 2.0, rtol=0.02)
    # difference modes: a rank-one change is one mode carrying all of it
    shape, weight = rng.normal(size=141), np.exp(-((np.arange(200) - 100) ** 2) / 300)
    m = npf.difference_modes(np.outer(weight, shape), 2)
    assert m["explained"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(np.abs(m["modes"][0] @ shape) / np.linalg.norm(shape), 1.0, rtol=1e-10)
    # variance along the sequence: none for identical rows; binning
    Pc = np.broadcast_to(ref, (3, 50, d.size))
    v = npf.variance_along_sequence(Pc)
    assert np.allclose(v["across"], 0) and np.allclose(v["noise"], 0)
    B, c = npf.binned_profiles(np.arange(50.0)[:, None] * np.ones((1, 3)), 20)
    np.testing.assert_allclose(B[:, 0], [9.5, 29.5])
    np.testing.assert_allclose(c, [10, 30])

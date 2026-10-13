"""Data preprocessing (spec 6.18): MATLAB-equivalent helpers, bump fit, and the MEC pipeline on a synthetic stand-in."""
import numpy as np
import pytest

from seqnet.data import common, mec
from seqnet.targets import SequenceSpec


def test_movmean_matches_matlab():
    x = np.array([1.0, 2, 3, 4, 5])
    np.testing.assert_allclose(common.movmean(x, 3), [1.5, 2, 3, 4, 4.5])  # MATLAB movmean(1:5, 3)
    np.testing.assert_allclose(common.movmean(x, 4), [1.5, 2, 2.5, 3.5, 4])  # MATLAB movmean(1:5, 4)
    X = np.vstack([x, 2 * x])
    np.testing.assert_allclose(common.movmean(X, 3, axis=1)[1], 2 * np.array([1.5, 2, 3, 4, 4.5]))


def test_fit_bump_recovers_parameters():
    """Recovery when the bump decays to baseline inside the window (sigma = 1 s)."""
    t = np.linspace(0, 8.5, 200)
    y = 0.83 * np.exp(-((t - 4.2) ** 2) / (2 * 1.0**2)) + 0.05
    f = common.fit_bump(t, y)
    assert abs(f.width - 1.0) < 0.01 and abs(f.amp - 0.83) < 0.01 and abs(f.offset - 0.05) < 0.001 and f.r2 > 0.999


def test_fit_bump_bias_of_minimum_offset():
    """Finding: Braun's offset = min(bump) (Methods line 1114) biases a broad bump: at sigma = 1.72 s in an 8.5-s
    window the bump is still 0.04 above baseline at the edges, so b is overestimated and sigma underestimated."""
    t = np.linspace(0, 8.5, 200)
    y = 0.83 * np.exp(-((t - 4.2) ** 2) / (2 * 1.72**2)) + 0.05
    f = common.fit_bump(t, y)
    assert 1.55 < f.width < 1.70 and f.offset > 0.07 and f.r2 > 0.99


def _fake_session(rng, N=60, n_seq=24, T_bins=1000):
    """A synthetic stand-in for the MEC session: travelling bumps with sequence durations jittered around T_bins."""
    acts, bounds, pos = [], [], 0
    spec = SequenceSpec(N=N, T=1.0, amp=0.3, width=0.08, offset=0.1)
    for s in range(n_seq):
        L = int(T_bins * rng.uniform(0.8, 1.0))
        t = np.linspace(0, 1, L, endpoint=False)
        acts.append(spec.rates(t))
        bounds.append((pos, pos + L))
        pos += L
    act = np.concatenate(acts, axis=1) + 0.01 * rng.random((N, pos))
    perm = rng.permutation(N)
    return mec.MECSession(activity=act[perm], order=np.argsort(perm), sequence_bounds=np.array(bounds)), spec


def test_mec_pipeline_on_stand_in():
    rng = np.random.default_rng(0)
    s, spec = _fake_session(rng)
    tgt = mec.training_target(s)
    assert tgt.shape[0] == 60 and tgt.max() <= 1 + 1e-12
    # ordering restored: peaks of consecutive neurons advance in time (within one sequence)
    first = tgt[:, : tgt.shape[1] // 24]
    assert np.mean(np.diff(np.argmax(first, 1)) >= 0) > 0.8
    fit, centred = mec.bump_fit(s)
    assert fit.r2 > 0.9 and 0.1 < fit.width < 2.0


def test_mec_loader_reports_missing_data():
    with pytest.raises(FileNotFoundError):
        mec.load_session(None)
    assert not mec.available(None)

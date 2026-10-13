"""Mouse MEC sequences (Gonzalo Cogno et al. 2024), session 7 of mouse #60584: drop-in skeleton (spec 6.18; D8).

The data are not available yet.  Everything downstream of :func:`load_session` is implemented and tested on synthetic
stand-ins (tests/core/test_data.py), so plugging in the data only requires implementing the loader for the actual file
format and setting ``mec.path`` in ``configs/wp0/data.yaml``.

Expected content of a session (what :func:`load_session` must return, as a :class:`MECSession`):
* ``activity``: deconvolved activity, (N_cells, n_bins) at 0.129-s bins (N = 484 in Braun et al.);
* ``order``: neuron ordering from the PCA method of Gonzalo Cogno et al. (2024), PCs 1-2 (indices into the cells);
* ``sequence_bounds``: (24, 2) start/stop bin of each ultraslow sequence in the session.

Training-target pipeline (Braun Methods lines 872-893; spec 6.18 mouse 1-6):
1. deconvolved activity at 0.129-s bins; 2. moving mean over 120 bins (``smoothdata`` 'movmean'); 3. downsample to
1.29-s bins; 4. divide each neuron by its maximum; 5. order by the PCA method; 6. 50 random snippets of 250 bins
(322.5 s), updates every m = 160 integration steps, 39655 steps per iteration, g = 0.05.

Bump fit (Braun lines 1090-1106; spec 6.18 'Mouse bump fit'): drop the first 4 of 24 sequences; rescale the remaining
20 to the longest (177.8 s = 1378 bins); average across sequences; normalize each neuron by its maximum; threshold at
mean + 1 SD (values below set to 0); circularly shift each neuron to peak at bin 689; rescale to 8.5 s; average across
neurons; fit Gaussian + offset.  Expected a = 0.29, sigma = 1.68 s, b = 0.15, R^2 = 0.99.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import BumpFit, activity_bump, fit_bump, movmean, normalize_max, rescale_duration, resample_bins

__all__ = ["MECSession", "load_session", "training_target", "mean_sequence", "bump_fit", "available"]


@dataclass(frozen=True)
class MECSession:
    activity: np.ndarray  # (N, n_bins) deconvolved activity at bin_size
    order: np.ndarray  # PCA-method sequence ordering
    sequence_bounds: np.ndarray  # (n_seq, 2) [start, stop) bins of each sequence
    bin_size: float = 0.129


def available(path: str | Path | None) -> bool:
    """True when a data path is configured and exists (notebooks skip the mouse panels otherwise)."""
    return path is not None and Path(path).exists()


def load_session(path: str | Path) -> MECSession:
    """Load session 7 of mouse #60584.  [DATA] The file format is not known yet; implement here when available."""
    if not available(path):
        raise FileNotFoundError(f"MEC data not available at {path!r} (spec D8); set mec.path in configs/wp0/data.yaml")
    raise NotImplementedError(
        "MEC loader: implement reading of the delivered files into MECSession(activity, order, sequence_bounds)"
    )


def training_target(s: MECSession, smooth_bins: int = 120, bin_out: float = 1.29) -> np.ndarray:
    """Spec 6.18 mouse steps 1-5: (N, n_bins_out) normalized target at 1.29-s bins in PCA order."""
    sm = movmean(s.activity, smooth_bins)
    ds, _ = resample_bins(sm, s.bin_size, bin_out)
    return normalize_max(ds)[s.order]


def mean_sequence(s: MECSession, drop_first: int = 4) -> np.ndarray:
    """Spec 6.18 'Mouse bump fit' steps 1-4: rescale sequences to the longest, average, normalize, threshold."""
    bounds = np.asarray(s.sequence_bounds)[drop_first:]
    n_long = int(np.max(bounds[:, 1] - bounds[:, 0]))
    segs = [rescale_duration(s.activity[:, a:b], n_long) for a, b in bounds]
    m = normalize_max(np.mean(segs, axis=0))[s.order]
    thr = m.mean(1, keepdims=True) + m.std(1, keepdims=True)
    return np.where(m >= thr, m, 0.0)


def bump_fit(s: MECSession, T_target: float = 8.5) -> tuple[BumpFit, np.ndarray]:
    """Spec 6.18 'Mouse bump fit' steps 5-6 (centre at the middle bin, 689 for 1378 bins)."""
    seq = mean_sequence(s)
    centred, t, bump = activity_bump(seq, seq.shape[1] // 2, T_target)
    return fit_bump(t, bump), centred

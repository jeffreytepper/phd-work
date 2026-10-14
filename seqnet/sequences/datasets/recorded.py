"""Recorded sequences as training targets (Braun Methods lines 841-905, 913-925).

A recorded sequence is a preprocessed rate matrix in [0, 1] at a fixed bin size.  It plugs into training through
the :class:`~seqnet.sequences.base.Sequence` interface, with Braun's snippet rule: update u of an iteration takes
target bin start + u (not the bin nearest to the update time), and the start bin is 0 (bat: one 6-s sequence shared by
all realizations) or uniform over the windows of ``snippet_bins`` bins (mouse: random 250-bin snippets).

Sequence specs (kinds registered in ``SEQUENCES``):
  {"kind": "bat", "path": "data/bat_omer2023/Fig_2.mat", "field": "correct"}
  {"kind": "mec", "path": <session path>}                 # [DATA] skeleton (.mec)
  {"kind": "npz", "path": <file with 'rates' (N, n_bins) and 'bin_size'>}
each with an optional "snippet_bins".  Relative paths are taken from the repository root.
"""
from __future__ import annotations

import numpy as np

from ...paths import resolve
from ..base import SEQUENCES, Snippet
from ..sampled import SampledSequence

__all__ = ["RecordedSequence", "TRAINING_DEFAULTS"]

# Braun Methods: training settings that go with each dataset (g, update interval m, steps per iteration, bin, snippet)
TRAINING_DEFAULTS = {
    "bat": {"g_init": 0.05, "m": 16, "steps_per_iter": 732, "bin_size": 0.129, "snippet_bins": None},
    "mec": {"g_init": 0.05, "m": 160, "steps_per_iter": 39655, "bin_size": 1.29, "snippet_bins": 250},
}


class RecordedSequence(SampledSequence):
    """Binned rates (N, n_bins) at ``bin_size``; bin b sits at t = b * bin_size."""

    def __init__(self, rates: np.ndarray, bin_size: float, snippet_bins: int | None = None):
        super().__init__(rates, bin_size, periodic=False)
        self.snippet_bins = None if snippet_bins is None else int(snippet_bins)

    @property
    def bin_size(self) -> float:
        return self.dt

    def draw_snippet(self, rng, offsets, start_max):
        """Consecutive bins, one per update (Braun lines 913-925); ``start_max`` does not apply.  The y* derivative is
        a finite difference with the update spacing (no exact derivative for data)."""
        offsets = np.asarray(offsets, dtype=float)
        n_up, n_bins = offsets.size, self.n_samples
        sb = self.snippet_bins
        start = int(rng.integers(0, n_bins - sb + 1)) if sb else 0
        idx = np.clip(start + np.arange(n_up), 0, n_bins - 1)
        t0 = start * self.bin_size
        spacing = float(offsets[1] - offsets[0]) if n_up > 1 else 0.0
        return Snippet(t0=t0, times=idx * self.bin_size, rates=self.samples[:, idx], rates_dt=None, spacing=spacing)


@SEQUENCES.register("bat")
def bat_sequence(path: str, field: str = "correct", snippet_bins: int | None = None) -> RecordedSequence:
    """Omer et al. (2023) bat time cells after Braun's preprocessing (.bat)."""
    from . import bat

    t = bat.preprocess(bat.load_zscored(resolve(path), field))
    return RecordedSequence(t.rates, t.bin_size, snippet_bins)


@SEQUENCES.register("mec")
def mec_sequence(path: str, snippet_bins: int | None = None) -> RecordedSequence:
    """Gonzalo Cogno et al. (2024) MEC session after Braun's preprocessing (.mec; loader is a [DATA] stub)."""
    from . import mec

    return RecordedSequence(mec.training_target(mec.load_session(resolve(path))), 1.29, snippet_bins)


@SEQUENCES.register("npz")
def npz_sequence(path: str, snippet_bins: int | None = None) -> RecordedSequence:
    """Any preprocessed target saved as ``rates`` (N, n_bins) and ``bin_size``."""
    with np.load(resolve(path)) as z:
        return RecordedSequence(np.asarray(z["rates"], float), float(z["bin_size"]), snippet_bins)

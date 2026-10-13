"""Bat CA1 time cells (Omer, Las & Ulanovsky 2023) and Braun et al.'s preprocessing (Methods lines 841-870; spec 6.18).

Source used here: the public Zenodo deposit of the paper (doi:10.5281/zenodo.7213099, CC BY 4.0), file ``Fig_2.mat``,
field ``Fig_2.panel_a.correct``: z-scored firing-rate maps of **166** observer-bat time cells, 0.1-s bins from the
observer's landing (column 0), 15 s.  Braun et al. used 168 time cells (170 in one place, A6), averaged over all hanging
intervals after landing on ball A or B.  The deposit is figure-level data, so this matrix is the closest public proxy,
not Braun's exact input (recorded as a deviation in results/wp0/REPORT.md).

Pipeline (spec 6.18, bat):
1. first 6 s after landing (60 bins of 0.1 s);
2. order by peak time after a 15-bin moving mean at 0.1 s;
3. downsample the unsmoothed rates to 0.129-s bins;
4. set negative values to 0;
5. 15-bin moving mean;
6. divide each neuron by its maximum.
Bump fit: centre every neuron at bin 24, rescale to 8.5 s, mean across neurons, Gaussian + offset (expected
a = 0.83, sigma = 1.72 s, b = 0.05, R^2 = 0.99).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import BumpFit, activity_bump, fit_bump, movmean, normalize_max, order_by_peak, resample_bins

__all__ = ["load_zscored", "BatTarget", "preprocess", "bump_fit", "DEFAULT_PATH"]

DEFAULT_PATH = Path("data") / "bat_omer2023" / "Fig_2.mat"


def load_zscored(path: str | Path = DEFAULT_PATH, field: str = "correct") -> np.ndarray:
    """z-scored rate maps (cells x 0.1-s bins from landing) from the Zenodo ``Fig_2.mat`` (read-only)."""
    import scipy.io as sio

    d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["Fig_2"]
    return np.asarray(getattr(d.panel_a, field), dtype=float)


@dataclass(frozen=True)
class BatTarget:
    rates: np.ndarray  # (N, n_bins) normalized target rates at bin_size
    bin_size: float
    order: np.ndarray  # neuron order applied (indices into the loaded matrix)
    z_window: np.ndarray  # z-scored rates of the analysed window at 0.1 s, in sequence order

    @property
    def N(self) -> int:
        return self.rates.shape[0]

    @property
    def duration(self) -> float:
        return self.rates.shape[1] * self.bin_size


def preprocess(z: np.ndarray, dt_in: float = 0.1, window_s: float = 6.0, bin_size: float = 0.129,
               smooth_sort: int = 15, smooth_target: int = 15) -> BatTarget:
    """Braun Methods lines 853-864 (spec 6.18 bat steps 1-6)."""
    n_win = int(round(window_s / dt_in))
    zw = np.asarray(z, float)[:, :n_win]
    order = order_by_peak(movmean(zw, smooth_sort))
    zw = zw[order]
    r, _ = resample_bins(zw, dt_in, bin_size, duration=window_s)
    r = np.clip(r, 0.0, None)
    r = movmean(r, smooth_target)
    return BatTarget(rates=normalize_max(r), bin_size=bin_size, order=order, z_window=zw)


def bump_fit(target: BatTarget, centre_bin: int = 24, T_target: float = 8.5) -> tuple[BumpFit, np.ndarray]:
    """Centre each neuron at ``centre_bin``, rescale to 8.5 s, average, fit (Braun lines 1083-1089, 1107-1118)."""
    centred, t, bump = activity_bump(target.rates, centre_bin, T_target)
    return fit_bump(t, bump), centred

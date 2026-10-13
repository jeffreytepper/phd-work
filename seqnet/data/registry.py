"""Data-target registry for FORCE training (Braun Methods lines 841-905, 913-925).

A data target is a dict in ``ForceConfig.data``:
  {"kind": "bat", "path": "data/bat_omer2023/Fig_2.mat", "field": "correct"}
  {"kind": "mec", "path": <session path>}            # [DATA] skeleton (seqnet.data.mec)
  {"kind": "npz", "path": <file with 'rates' (N, n_bins) and 'bin_size'>}   # any pre-processed target

``load_target`` returns (rates (N, n_bins) in [0, 1], bin_size) and is cached per process.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["load_target", "DATA_DEFAULTS"]

# Braun Methods: bat = one 6-s sequence, all realizations share the target; mouse = 50 random 250-bin snippets
DATA_DEFAULTS = {
    "bat": {"g_init": 0.05, "m": 16, "steps_per_iter": 732, "bin_size": 0.129, "snippet_bins": None},
    "mec": {"g_init": 0.05, "m": 160, "steps_per_iter": 39655, "bin_size": 1.29, "snippet_bins": 250},
}


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    from ..experiments.runner import repo_root

    return repo_root() / p


@lru_cache(maxsize=8)
def _load(key: str) -> tuple[np.ndarray, float]:
    d = json.loads(key)
    kind = d["kind"]
    if kind == "bat":
        from . import bat

        t = bat.preprocess(bat.load_zscored(_resolve(d["path"]), d.get("field", "correct")))
        return t.rates, t.bin_size
    if kind == "mec":
        from . import mec

        s = mec.load_session(_resolve(d["path"]))
        return mec.training_target(s), 1.29
    if kind == "npz":
        with np.load(_resolve(d["path"])) as z:
            return np.asarray(z["rates"], float), float(z["bin_size"])
    raise ValueError(f"unknown data kind {kind!r}")


def load_target(spec: dict) -> tuple[np.ndarray, float]:
    """Load (and cache) a data target described by ``spec``."""
    return _load(json.dumps(spec, sort_keys=True))

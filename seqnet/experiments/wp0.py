"""WP0 conditions (spec Section 7) and a CLI to train them: ``python -m seqnet.experiments.wp0 <group> --settings dev|final``.

Settings (spec 5.5, D9): ``final`` = Braun's numbers (50 realizations, 300 training + 10 test iterations);
``dev`` = 10 realizations, 100 training + 10 test iterations.  Every run directory records which was used.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from ..training.rls import ForceConfig
from .runner import RunResult, run_condition

__all__ = ["SETTINGS", "condition", "GROUPS", "run_group", "get"]

SETTINGS = {
    "final": {"n_realizations": 50, "n_train": 300, "n_test": 10},
    "dev": {"n_realizations": 10, "n_train": 100, "n_test": 10},
}


def condition(amp: float, width: float, offset: float, settings: str = "dev", **overrides) -> tuple[ForceConfig, int]:
    s = SETTINGS[settings]
    cfg = ForceConfig(amp=amp, width=width, offset=offset, n_train=s["n_train"], n_test=s["n_test"])
    return replace(cfg, **overrides), s["n_realizations"]


def _r(x):
    return [round(float(v), 4) for v in x]


# (amp, width, offset) triples per group; spec Section 7, WP0.1-WP0.6
GROUPS = {
    "fig1": [(0.5, 1.7, 0.0)],
    "offset": [(0.5, 1.0, b) for b in _r(np.arange(0, 0.451, 0.05))],
    "width": [(0.5, s, 0.0) for s in [0.5, 0.9, 1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 1.9]],
    "amplitude": [(a, 1.0, 0.0) for a in _r(np.arange(0.2, 1.001, 0.1))],
    "amp_width": [(a, s, 0.0) for a in [0.4, 0.6, 0.8, 1.0] for s in [0.5, 0.9, 1.3, 1.7]],
    "offset_amp": [(a, 1.0, b) for b in [0.0, 0.1, 0.2, 0.3] for a in [0.4, 0.5, 0.6, 0.7]],
    "perturb": [(0.6, 1.0, 0.0)],
    "supp8": [(0.6, 1.0, 0.0), (0.6, 1.0, 0.2), (0.6, 1.0, 0.4), (0.6, 1.5, 0.0), (0.6, 2.0, 0.0), (0.8, 1.0, 0.0),
              (1.0, 1.0, 0.0)],
    "downstream": [(0.5, 1.0, 0.3), (0.6, 1.0, 0.3)],
}


def label_of(amp, width, offset) -> str:
    return f"a{amp:g}_s{width:g}_b{offset:g}"


def get(amp: float, width: float, offset: float, settings: str = "dev", store_full=5, **overrides) -> RunResult:
    """Train or load one condition."""
    cfg, R = condition(amp, width, offset, settings, **overrides)
    lab = label_of(amp, width, offset) + ("" if not overrides else "_" + "_".join(f"{k}{v}" for k, v in sorted(overrides.items())))
    return run_condition(cfg, R, label=f"{settings}-{lab}", store_full=store_full, settings=settings)


def run_group(group: str, settings: str, store_full=5) -> list[RunResult]:
    out = []
    for amp, width, offset in GROUPS[group]:
        res = get(amp, width, offset, settings, store_full=store_full)
        s = res.summary
        print(f"{group} {label_of(amp, width, offset)} [{settings}] test corr {s['corr'][:, -1].mean():.3f}  "
              f"profile corr {np.median(s['profile_corr']):.3f}  matrix corr {np.median(s['matrix_corr']):.3f}  "
              f"wall {res.meta['wall_time_total_s']:.0f}s  -> {res.path.name}", flush=True)
        out.append(res)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", choices=list(GROUPS) + ["all"])
    ap.add_argument("--settings", default="dev", choices=list(SETTINGS))
    ap.add_argument("--store-full", default="5")
    a = ap.parse_args(argv)
    sf = a.store_full if a.store_full == "all" else int(a.store_full)
    groups = list(GROUPS) if a.groups == ["all"] else a.groups
    for g in groups:
        run_group(g, a.settings, store_full=sf)


if __name__ == "__main__":
    main()

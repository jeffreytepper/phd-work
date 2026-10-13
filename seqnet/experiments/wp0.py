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


def data_config() -> dict:
    """configs/wp0/data.yaml."""
    import yaml

    from .runner import repo_root

    return yaml.safe_load((repo_root() / "configs" / "wp0" / "data.yaml").read_text(encoding="utf-8"))


def get_data(name: str, settings: str = "final", store_full=5, **overrides) -> RunResult:
    """Train or load networks on an experimental target (Braun Fig. 3; spec WP0.7).

    ``name`` is a key of configs/wp0/data.yaml ('bat' or 'mec').  g, m, steps per iteration and snippet length follow
    Braun's Methods (``seqnet.data.registry.DATA_DEFAULTS``).  Raises FileNotFoundError if the path is not set.
    """
    from pathlib import Path

    from ..data.registry import DATA_DEFAULTS
    from .runner import repo_root

    d = dict(data_config()[name])
    if not d.get("path") or not (repo_root() / Path(d["path"])).exists():
        raise FileNotFoundError(f"{name} data not available (configs/wp0/data.yaml: path = {d.get('path')!r})")
    kind = d["kind"]
    dd = DATA_DEFAULTS[kind]
    spec = {"kind": kind, "path": d["path"]}
    if "field" in d:
        spec["field"] = d["field"]
    if dd["snippet_bins"]:
        spec["snippet_bins"] = dd["snippet_bins"]
    s = SETTINGS[settings]
    cfg = ForceConfig(data=spec, g_init=dd["g_init"], m=dd["m"], steps_per_iter=dd["steps_per_iter"],
                      bin_size=dd["bin_size"], n_train=s["n_train"], n_test=s["n_test"])
    cfg = replace(cfg, **overrides)
    from .runner import run_condition as _rc

    return _rc(cfg, s["n_realizations"], label=f"{settings}-data_{name}", store_full=store_full, settings=settings)


def get_matched(name: str, which: str = "ours", settings: str = "final", store_full=5) -> RunResult:
    """Synthetic sequences matched to a dataset's fitted bump (Braun lines 1119-1123; Fig. 3C/E bottom).

    ``which='ours'``: parameters fitted here to the available data (stored by the data notebooks in
    results/wp0/data/braun_supp6/fit_<name>.json); ``'paper'``: Braun's printed fit.  N equals the dataset's cell count;
    synthetic settings otherwise (T = 8.5 s, 17-s snippets, g = 1.5).
    """
    import json

    from ..data.registry import load_target
    from .runner import repo_root

    d = data_config()[name]
    spec = {"kind": d["kind"], "path": d["path"], **({"field": d["field"]} if "field" in d else {})}
    N = load_target(spec)[0].shape[0]
    if which == "paper":
        p = d["fit_paper"]
    else:
        p = json.loads((repo_root() / "results" / "wp0" / "data" / "braun_supp6" / f"fit_{name}.json").read_text())
    amp, width, offset = (round(float(p[k]), 3) for k in ("amp", "width", "offset"))
    cfg, R = condition(amp, width, offset, settings, N=N)
    from .runner import run_condition as _rc

    return _rc(cfg, R, label=f"{settings}-matched_{name}_{which}_{label_of(amp, width, offset)}", store_full=store_full,
               settings=settings)


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
    ap.add_argument("groups", nargs="+", choices=list(GROUPS) + ["all", "bat"])
    ap.add_argument("--settings", default="dev", choices=list(SETTINGS))
    ap.add_argument("--store-full", default="5")
    a = ap.parse_args(argv)
    sf = a.store_full if a.store_full == "all" else int(a.store_full)
    groups = list(GROUPS) if a.groups == ["all"] else a.groups
    for g in groups:
        if g == "bat":
            r = get_data("bat", a.settings, store_full=sf)
            print("bat", r.path.name, "test corr", r.summary["corr"][:, -1].mean(), flush=True)
            continue
        run_group(g, a.settings, store_full=sf)


if __name__ == "__main__":
    main()

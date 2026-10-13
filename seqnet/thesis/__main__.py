"""CLI: ``python -m seqnet.thesis <experiment> [--config FILE]`` (spec 6A.2 rule 3).

Experiments
-----------
fig2 ... fig9   Execute the figure notebook ``seqnet/thesis/figures/figN.ipynb`` in place (needs ``nbconvert`` and a
                Jupyter kernel; optional dependencies).  Data -> results/thesis/data/figN, panels ->
                results/thesis/figures/figN, acceptance table -> results/thesis/data/figN/acceptance.json.
water_fill      Solve eq. 7.11-7.13 for a target class given in a config (JSON or YAML):
                {"N": 128, "decay": 0.1, "K0": 10, "units": "unitary"|"d10", "T_obs": null,
                 "target": {"kind": "narrowband", "xi0": 24, "ell": 3}, "law": "cube"|"sqrt"}
learn           Run the LearningSystem (eq. 9.31) from a config with the keys above plus
                {"kernel_epochs": 2000, "rule": "thesis"|"ridge", "budget": "project", "eta_k_factor": 10,
                 "precondition": "none", "initial_sigma": 0.6, "out": "results/thesis/runs/learn.npz"}
summary         Print every figure's acceptance table.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import (
    Kernel,
    LearningSystem,
    RidgeObjectiveRule,
    TargetClass,
    ThesisKernelRule,
    mode_multiplicity,
    water_fill,
)
from .figures import load_config, repo_root

FIGS = [f"fig{i}" for i in range(2, 10)]


def _read_config(path: str | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return json.loads(text)
    import yaml  # optional

    return yaml.safe_load(text)


def _target(spec: dict, M: int) -> TargetClass:
    spec = dict(spec)
    kind = spec.pop("kind")
    return getattr(TargetClass, kind)(M, **spec)


def _units(cfg: dict) -> tuple[int, float, float]:
    N = int(cfg["N"])
    f = N if cfg.get("units", "unitary") == "unitary" else 1
    return N, f * float(cfg["decay"]), f * float(cfg["K0"])


def run_figure(fig: str) -> int:
    nb = Path(__file__).parent / "figures" / f"{fig}.ipynb"
    exe = shutil.which("jupyter-nbconvert") or shutil.which("jupyter")
    if exe is None:
        print("nbconvert is not installed; install the 'notebooks' extra", file=sys.stderr)
        return 2
    cmd = [exe] + ([] if exe.endswith(("nbconvert", "nbconvert.exe")) else ["nbconvert"])
    cmd += ["--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=3600", str(nb)]
    r = subprocess.run(cmd, cwd=nb.parent)
    if r.returncode == 0:
        summary([fig])
    return r.returncode


def summary(figs=FIGS) -> int:
    root = repo_root() / "results" / "thesis" / "data"
    for fig in figs:
        f = root / fig / "acceptance.json"
        if not f.exists():
            print(f"{fig}: not run")
            continue
        for c in json.loads(f.read_text(encoding="utf-8")):
            mark = {True: "PASS", False: "FAIL", None: "info"}[c.get("passed")]
            print(f"{fig} [{mark}] {c['item']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m seqnet.thesis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=FIGS + ["water_fill", "learn", "summary"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if args.experiment in FIGS:
        if args.config:
            print("note: figure notebooks read configs/thesis/<fig>.yaml; --config is ignored", file=sys.stderr)
        load_config(args.experiment)  # fail early if the config is missing
        return run_figure(args.experiment)
    if args.experiment == "summary":
        return summary()
    cfg = _read_config(args.config)
    N, lam, K0 = _units(cfg)
    S = _target(cfg["target"], N // 2).psd.copy()
    S[-1] = 0.0 if N % 2 == 0 else S[-1]
    if args.experiment == "water_fill":
        r = water_fill(S, lam, K0, cfg.get("T_obs"), N, law=cfg.get("law", "cube"))
        out = {"nu": r.nu, "M_star": r.M_star, "rank": r.rank, "T_star": 2 * r.M_star,
               "budget_residual": r.budget_residual, "a": r.a.tolist()}
        print(json.dumps(out, indent=1))
        return 0
    # learn
    mult = mode_multiplicity(N)
    a0 = Kernel.from_family("gaussian", N, sigma=float(cfg.get("initial_sigma", 0.6))).alloc()
    a0 *= K0 / np.sum(mult * a0)
    rule_cls = ThesisKernelRule if cfg.get("rule", "thesis") == "thesis" else RidgeObjectiveRule
    rule = rule_cls(eta=float(cfg.get("eta_k_factor", 10.0)) * lam**2 / (3 * S.max()), decay=lam, N=N, K0=K0,
                    budget=cfg.get("budget", "project"), precondition=cfg.get("precondition", "none"))
    ls = LearningSystem(N=N, decay=lam, K0=K0, target=TargetClass(psd=S), a0=a0, rule=rule,
                        readout=cfg.get("readout", "exact"), seed=int(cfg.get("seed", 0)))
    h = ls.run(int(cfg.get("kernel_epochs", 2000))).as_arrays()
    out = repo_root() / cfg.get("out", "results/thesis/runs/learn.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: v for k, v in h.items() if v.size})
    ref = water_fill(S, lam, K0, N=N, law="cube" if rule_cls is ThesisKernelRule else "sqrt")
    rel = float(np.max(np.abs(ls.a - ref.a)[ref.active] / ref.a[ref.active]))
    print(json.dumps({"saved": str(out), "max_rel_error_vs_optimum": rel, "M_star": ref.M_star}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

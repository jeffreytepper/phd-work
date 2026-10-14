"""Cached, parallel execution of FORCE training conditions (spec 5.3-5.5).

``run_condition(cfg, n_realizations)`` trains ``n_realizations`` independent networks for one condition and stores a
run directory ``results/wp0/runs/<label>-<hash>/`` with

* ``meta.json``: resolved config, git commit (and dirty flag), package versions, seeds, wall time, settings label;
* ``summary.npz``: mean matrix, per-realization profiles, correlation and weight-change curves, mean self-weight,
  pairwise matrix and profile correlations, test-iteration rates of realization 0;
* ``W_<r>.npz``: full matrices for the first ``store_full`` realizations (all for ``store_full='all'``), plus J0 for
  realizations 0 and the last (Supp. Fig. 1G).

A second call with the same config and realization count loads the cache instead of retraining.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..conventions import profile_from_matrix
from ..paths import repo_root
from ..training.config import legacy_layout
from ..training.rls import ForceConfig, train_realization

__all__ = ["repo_root", "git_info", "run_condition", "RunResult", "config_hash", "run_hashes"]


def git_info() -> dict:
    root = repo_root()
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "seqnet"], cwd=root, capture_output=True,
                                    text=True).stdout.strip())
        return {"commit": h, "dirty_seqnet": dirty}
    except OSError:
        return {"commit": None, "dirty_seqnet": None}


# config fields added after runs were cached; omitted from the hash while at their default so old caches stay valid
_LATE_DEFAULTS = {"data": None}


def _strip_late(d):
    if isinstance(d, dict):
        out = {k: _strip_late(v) for k, v in d.items()}
        for k, default in _LATE_DEFAULTS.items():
            if k in out and out[k] == default:
                del out[k]
        return out
    return d


def config_hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(_strip_late(d), sort_keys=True).encode()).hexdigest()[:10]


def run_hashes(cfg: ForceConfig | dict, n_realizations: int) -> tuple[str, str]:
    """(hash, fallback hash) naming a condition's run directory.

    Targets the pre-sequences layout could express are hashed in that layout, so their cached runs stay valid.  The
    fallback is the hash of the full dict, used by runs cached between adding a config field and _LATE_DEFAULTS.
    """
    cfg_d = (cfg if isinstance(cfg, ForceConfig) else ForceConfig.from_dict(cfg)).as_dict()
    key = {"cfg": legacy_layout(cfg_d), "n_realizations": n_realizations}
    return config_hash(key), hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def _worker(args):
    cfg_dict, r, keep_W0 = args
    return r, train_realization(ForceConfig(**cfg_dict), r, keep_W0=keep_W0)


@dataclass
class RunResult:
    path: Path
    meta: dict
    summary: dict

    def W(self, r: int) -> np.ndarray:
        return np.load(self.path / f"W_{r}.npz")["J"]

    def W0(self, r: int) -> np.ndarray:
        return np.load(self.path / f"W_{r}.npz")["J0"]

    @property
    def config(self) -> ForceConfig:
        """The run's configuration (stored in the current or the legacy layout)."""
        return ForceConfig.from_dict(self.meta["config"])

    def has_W(self, r: int) -> bool:
        return (self.path / f"W_{r}.npz").exists()


def _offdiag_corr(G: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlations (upper triangle) from a stack of flattened, centred vectors' Gram matrix."""
    d = np.sqrt(np.diag(G))
    C = G / np.outer(d, d)
    iu = np.triu_indices(G.shape[0], 1)
    return C[iu]


def run_condition(
    cfg: ForceConfig | dict,
    n_realizations: int,
    label: str = "run",
    n_workers: int | None = None,
    store_full: int | str = 5,
    settings: str = "dev",
    root: Path | None = None,
    force: bool = False,
) -> RunResult:
    """Train (or load) ``n_realizations`` networks for one condition (Braun Fig. 1C procedure)."""
    cfg_d = (cfg if isinstance(cfg, ForceConfig) else ForceConfig.from_dict(cfg)).as_dict()
    h, h_full = run_hashes(cfg_d, n_realizations)
    base = (root or repo_root() / "results" / "wp0" / "runs")
    path = base / f"{label}-{h}"
    if not (path / "summary.npz").exists() and (base / f"{label}-{h_full}" / "summary.npz").exists():
        path, h = base / f"{label}-{h_full}", h_full
    if (path / "summary.npz").exists() and not force:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        with np.load(path / "summary.npz") as z:
            return RunResult(path, meta, {k: z[k] for k in z.files})
    path.mkdir(parents=True, exist_ok=True)
    # default: one worker per physical core (SMT siblings share caches; each worker streams J and P, 2 MB each at
    # N = 501, so more workers than physical cores is memory-bandwidth bound).  Override with SEQNET_WORKERS.
    n_workers = n_workers or int(os.environ.get("SEQNET_WORKERS", 0)) or max(1, min(n_realizations, (os.cpu_count() or 2) // 2))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"  # one BLAS thread per worker process
    t0 = time.time()
    n_full = n_realizations if store_full == "all" else int(store_full)
    keep_W0 = {0, n_realizations - 1}
    sumW = None  # sized from the first result (data targets set N from the data)
    profiles, corr, dW, diag, t0s, walls = [], [], [], [], [], []
    flat = []  # centred flattened matrices for pairwise correlations
    R_test0 = f_target0 = None
    jobs = [(cfg_d, r, r in keep_W0) for r in range(n_realizations)]
    results = {}
    # multiprocessing.Pool, not ProcessPoolExecutor: leaving the block terminates the workers instead of waiting for
    # them to exit; on Windows an executor occasionally left idle workers that never exited, hanging the run after
    # every result had arrived (2026-10-14).  Results are collected before the block ends, so nothing is lost.
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for r, out in pool.imap(_worker, jobs):
            results[r] = True
            J = out["J"]
            sumW = J.copy() if sumW is None else sumW + J
            profiles.append(profile_from_matrix(J))
            corr.append(out["corr"])
            dW.append(out["dW"])
            diag.append(float(np.mean(np.diag(J))))
            t0s.append(out["t0"])
            walls.append(out["wall_time"])
            flat.append((J - J.mean()).ravel().astype(np.float32))
            if r < n_full or r in keep_W0:
                save = {"J": J}
                if "J0" in out:
                    save["J0"] = out["J0"]
                if "keep" in out:
                    save["keep"] = out["keep"]
                np.savez(path / f"W_{r}.npz", **save)
            if r == 0:
                R_test0, f_target0 = out["R_test"], out["f_target"]
    F = np.stack(flat).astype(np.float64)
    G = F @ F.T
    P = np.stack(profiles)
    Pc = P - P.mean(1, keepdims=True)
    summary = {
        "mean_W": sumW / n_realizations,
        "profiles": P,
        "corr": np.stack(corr),
        "dW": np.stack(dW),
        "diag_mean": np.asarray(diag),
        "t0": np.asarray(t0s),
        "matrix_corr": _offdiag_corr(G),
        "profile_corr": _offdiag_corr(Pc @ Pc.T),
        "R_test0": R_test0,
        "f_target0": f_target0,
    }
    np.savez(path / "summary.npz", **summary)
    meta = {
        "label": label, "hash": h, "config": cfg_d, "n_realizations": n_realizations, "settings": settings,
        "git": git_info(), "python": platform.python_version(), "numpy": np.__version__,
        "wall_time_total_s": time.time() - t0, "wall_time_per_realization_s": float(np.mean(walls)),
        "n_workers": n_workers, "seeds": {"base": cfg_d["seed"], "realizations": list(range(n_realizations))},
        "verify_switches": {k: cfg_d[k] for k in ("error_target", "c_uses_updated_P", "state_reset", "target_time",
                                                  "weight_decay", "mask")},
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return RunResult(path, meta, summary)

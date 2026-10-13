"""Helpers for the thesis-figure notebooks (spec 6A.3, 6A.6).

One notebook per thesis figure lives next to this file (``fig2.ipynb`` ... ``fig9.ipynb``).  Each panel has
a *compute* cell that writes ``results/thesis/data/figN/<panel>.npz`` and a *plot* cell that reads only that
file and writes ``results/thesis/figures/figN/<panel>.png`` (spec 5.6: a figure function never computes).
Parameters come from ``configs/thesis/figN.yaml``; every run records them in ``<panel>.json`` next to the data.

Only the standard library, NumPy and matplotlib are imported here; YAML configs are read with PyYAML when it
is installed and otherwise must be JSON (a deviation recorded in results/thesis/REPORT.md).
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["repo_root", "load_config", "save_data", "load_data", "save_figure", "record_checks", "style"]


def repo_root(start: str | os.PathLike | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the directory holding pyproject.toml; env SEQNET_ROOT overrides."""
    env = os.environ.get("SEQNET_ROOT")
    if env:
        return Path(env)
    p = Path(start or os.getcwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists():
            return q
    return p


def load_config(fig: str) -> dict:
    """Read configs/thesis/<fig>.yaml (or .json)."""
    root = repo_root()
    for ext in (".yaml", ".yml", ".json"):
        f = root / "configs" / "thesis" / f"{fig}{ext}"
        if f.exists():
            text = f.read_text(encoding="utf-8")
            if ext == ".json":
                return json.loads(text)
            try:
                import yaml  # optional; not a hard dependency of seqnet.thesis
            except ImportError as e:  # pragma: no cover
                raise ImportError("PyYAML is needed to read YAML configs; use a .json config instead") from e
            return yaml.safe_load(text)
    raise FileNotFoundError(f"no config for {fig} under {root / 'configs' / 'thesis'}")


def _dirs(fig: str) -> tuple[Path, Path]:
    root = repo_root() / "results" / "thesis"
    d, f = root / "data" / fig, root / "figures" / fig
    d.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return d, f


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def save_data(fig: str, panel: str, params: dict | None = None, **arrays) -> Path:
    """Save a panel's arrays (npz) and its parameters plus run metadata (json)."""
    d, _ = _dirs(fig)
    path = d / f"{panel}.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    meta = {
        "figure": fig, "panel": panel, "params": _jsonable(params or {}),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "numpy": np.__version__, "python": platform.python_version(),
    }
    (d / f"{panel}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path


def load_data(fig: str, panel: str) -> dict:
    """Load a panel's arrays and parameters: returns {name: array, ..., '_params': dict}."""
    d, _ = _dirs(fig)
    with np.load(d / f"{panel}.npz", allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    meta = d / f"{panel}.json"
    out["_params"] = json.loads(meta.read_text(encoding="utf-8"))["params"] if meta.exists() else {}
    return out


def save_figure(fig: str, panel: str, mpl_fig) -> Path:
    """Write a panel's matplotlib figure to results/thesis/figures/<fig>/<panel>.png."""
    _, f = _dirs(fig)
    path = f / f"{panel}.png"
    mpl_fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def record_checks(fig: str, checks: list[dict]) -> Path:
    """Write the acceptance table of a figure: list of {item, criterion, value, passed}."""
    d, _ = _dirs(fig)
    path = d / "acceptance.json"
    path.write_text(json.dumps(_jsonable(checks), indent=1), encoding="utf-8")
    return path


def style() -> None:
    """Common matplotlib style for the reproductions."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    })

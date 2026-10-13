"""Notebook helpers for the main-package reproductions (Braun et al.; WP0 onward).

Same pattern as :mod:`seqnet.thesis.figures`: one notebook per figure (``notebooks/<wp>/<fig>.ipynb``), panel by panel;
compute cells write ``results/<wp>/data/<fig>/<panel>.npz`` (+ JSON parameters), plot cells read only those files and
write ``results/<wp>/figures/<fig>/<panel>.png``; configs live in ``configs/<wp>/<fig>.yaml``; each notebook ends with
``results/<wp>/data/<fig>/acceptance.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .experiments.runner import git_info, repo_root

__all__ = ["Figure", "style"]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


class Figure:
    """I/O for one reproduced figure: ``fig = Figure('wp0', 'braun_fig1')``."""

    def __init__(self, wp: str, name: str):
        self.wp, self.name = wp, name
        root = repo_root()
        self.data_dir = root / "results" / wp / "data" / name
        self.fig_dir = root / "results" / wp / "figures" / name
        self.cfg_path = root / "configs" / wp / f"{name}.yaml"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def config(self) -> dict:
        import yaml

        return yaml.safe_load(self.cfg_path.read_text(encoding="utf-8"))

    def save(self, panel: str, params: dict | None = None, **arrays) -> Path:
        np.savez(self.data_dir / f"{panel}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
        meta = {"figure": self.name, "panel": panel, "params": _jsonable(params or {}),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_info()}
        (self.data_dir / f"{panel}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return self.data_dir / f"{panel}.npz"

    def load(self, panel: str) -> dict:
        with np.load(self.data_dir / f"{panel}.npz", allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        meta = self.data_dir / f"{panel}.json"
        out["_params"] = json.loads(meta.read_text(encoding="utf-8"))["params"] if meta.exists() else {}
        return out

    def savefig(self, panel: str, mpl_fig) -> Path:
        p = self.fig_dir / f"{panel}.png"
        mpl_fig.savefig(p, dpi=150, bbox_inches="tight")
        return p

    def checks(self, checks: list[dict]) -> Path:
        p = self.data_dir / "acceptance.json"
        p.write_text(json.dumps(_jsonable(checks), indent=1), encoding="utf-8")
        for c in checks:
            mark = {True: "PASS", False: "FAIL", None: "info"}[c.get("passed")]
            print(f"[{mark}] {c['item']}: {c.get('value')}")
        return p


def style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "image.cmap": "viridis",
    })

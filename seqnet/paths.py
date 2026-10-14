"""Repository paths (shared by the experiment runner and the dataset loaders)."""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "resolve"]


def repo_root() -> Path:
    """$SEQNET_ROOT, else the nearest ancestor of the working directory holding pyproject.toml, else the package's."""
    env = os.environ.get("SEQNET_ROOT")
    if env:
        return Path(env)
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists():
            return q
    return Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    """Absolute paths unchanged; relative paths are taken relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p

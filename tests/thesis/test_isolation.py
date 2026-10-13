"""TT17 (import boundary) and TT18 (isolated run) for seqnet.thesis (spec 6A.2 rules 1-2)."""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "seqnet" / "thesis"
TESTS = Path(__file__).resolve().parent
ALLOWED_TOP = {"numpy", "scipy", "matplotlib", "yaml", "__future__"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or ""), node.level


def test_TT17_import_boundary():
    stdlib = set(sys.stdlib_module_names)
    for f in PKG.rglob("*.py"):
        for name, level in _imports(f):
            if level > 0:
                continue  # relative import inside seqnet.thesis
            top = name.split(".")[0]
            if name.startswith("seqnet"):
                assert name == "seqnet.thesis" or name.startswith("seqnet.thesis."), (f, name)
            else:
                assert top in stdlib or top in ALLOWED_TOP, (f, name)


@pytest.mark.skipif(os.environ.get("SEQNET_ISOLATED") == "1", reason="already running isolated")
def test_TT18_isolated_run(tmp_path):
    """Copy seqnet/thesis and tests/thesis into an empty directory and run the tests there."""
    (tmp_path / "seqnet").mkdir()
    (tmp_path / "seqnet" / "__init__.py").write_text("")
    shutil.copytree(PKG, tmp_path / "seqnet" / "thesis", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(TESTS, tmp_path / "tests" / "thesis", ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ, SEQNET_ISOLATED="1", PYTHONPATH=str(tmp_path))
    # -p no:cacheprovider keeps the copy clean; run from tmp so the installed package is shadowed
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/thesis",
         "-k", "not TT18 and not dual_budget"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1200,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    # confirm the copy (not the source tree) was imported
    chk = subprocess.run([sys.executable, "-c", "import seqnet.thesis as t; print(t.__file__)"],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert str(tmp_path) in chk.stdout, chk.stdout

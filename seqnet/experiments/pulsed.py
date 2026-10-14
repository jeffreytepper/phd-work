"""Testbed: FORCE training on sequences with a parameter pulsed inside the training snippet.

Braun's protocol draws one random snippet start per realization; a pulse at a fixed time would then fall at a
different place, or outside, each network's snippet.  Here every realization trains on the same snippet:

* start t0 = 0 (``snippet_start_max = 0``), so realizations differ only in J0 and noise, and their matrices can be
  averaged neuron by neuron;
* length = ``passes`` passes of the bump around the ring (default one: every neuron is active once, so the neurons
  active during the pulse are unambiguous), measured on the pulsed sequence itself (a speed pulse shortens it).

A pulse is a Gaussian bump in one control of the sequence spec: ``gain`` (speed), ``width_scale``, ``amplitude``
(parametric Lehr bumps) or ``drive`` (E - I of the Lehr ring network).  Times are on the training clock; for a
``time_scaled`` network they are converted to model time.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from ..paths import repo_root
from ..sequences import Sequence, build_sequence, measures
from ..training.config import ForceConfig
from .runner import RunResult, run_condition, run_hashes

__all__ = ["Pulse", "pulsed_spec", "resolve_time_scale", "pass_duration", "pulse_neuron", "snippet_start", "speed_along_snippet",
           "condition_config", "run", "sweep_conditions", "train_sweep", "RUNS"]

RUNS = ("results", "pulsed", "runs")
_BASELINE = {"gain": 1.0, "width_scale": 1.0, "amplitude": 1.0, "drive": 0.0}


@dataclass(frozen=True)
class Pulse:
    """Gaussian pulse of ``height`` in control ``param``, centred at ``t_peak`` (s) with SD ``width`` (s)."""

    param: str
    height: float
    t_peak: float
    width: float


def resolve_time_scale(spec: dict) -> dict:
    """A ``time_scaled`` spec given by ``period`` becomes one given by the equivalent ``factor`` (so that pulse times
    can be converted to model time before the pulse changes the measured period)."""
    if spec.get("kind") != "time_scaled" or "factor" in spec:
        return spec
    seq = build_sequence(spec)
    out = {k: v for k, v in spec.items() if k not in ("period", "measure_until")}
    out["factor"] = float(seq.factor)
    return out


def pulsed_spec(base: dict, pulse: Pulse | None) -> dict:
    """``base`` with ``pulse`` added to its control (a number or absent control is the pulse's baseline)."""
    spec = resolve_time_scale(copy.deepcopy(base))
    if pulse is None:
        return spec
    target, f = (spec["sequence"], spec["factor"]) if spec.get("kind") == "time_scaled" else (spec, 1.0)
    current = target.get(pulse.param, _BASELINE[pulse.param])
    if not isinstance(current, (int, float)):
        raise ValueError(f"{pulse.param} is already a time course; pulses are added to constant controls only")
    target[pulse.param] = {"kind": "gaussian_pulse", "baseline": float(current), "height": pulse.height,
                           "t_peak": pulse.t_peak / f, "width": pulse.width / f}
    return spec


def pass_duration(seq: Sequence, passes: float = 1.0, dt: float = 0.005, t_max: float = 600.0) -> float:
    """Time (s) for the bump to travel ``passes`` * N neurons from t = 0 (from the measured bump position)."""
    t = np.arange(0.0, t_max, dt)
    if seq.duration is not None:
        t = t[t <= seq.duration]
    travelled = np.abs(measures.bump_position(seq.rates(t)) - measures.bump_position(seq.rates(t[:1]))[0])
    hit = np.nonzero(travelled >= passes * seq.N)[0]
    if hit.size == 0:
        raise ValueError(f"the bump does not complete {passes} pass(es) within {t[-1]:g} s")
    i = hit[0]  # interpolate the crossing between samples i - 1 and i
    a, b = travelled[i - 1], travelled[i]
    return float(t[i - 1] + (passes * seq.N - a) / (b - a) * dt)


def pulse_neuron(seq: Sequence, t_peak: float) -> float:
    """Sequence position (neuron index, 0..N) of the bump centre at the pulse peak."""
    return float(measures.bump_position(seq.rates(np.array([t_peak])))[0] % seq.N)


def snippet_start(seq: Sequence) -> float:
    """Sequence position (neuron index, 0..N) of the bump centre at t = 0, where every snippet starts."""
    return pulse_neuron(seq, 0.0)


def speed_along_snippet(seq: Sequence, T_snip: float, dt: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """(snippet position 0..N, bump speed in neurons / s when the bump passes it), from the measured bump position."""
    t = np.arange(0.0, T_snip + dt, dt)
    pos = measures.bump_position(seq.rates(t))
    s = pos - pos[0]
    v = np.gradient(pos, dt)
    return s, v


def condition_config(spec: dict, n_train: int, n_test: int, passes: float = 1.0, **overrides) -> ForceConfig:
    """Braun's FORCE settings on one fixed snippet of ``passes`` bump passes starting at t = 0."""
    base = ForceConfig(sequence=spec, n_train=n_train, n_test=n_test, snippet_start_max=0.0)
    steps = int(np.ceil(pass_duration(build_sequence(spec), passes) / base.dt)) + 1
    return replace(base, steps_per_iter=steps, **overrides)


def run(name: str, cfg: ForceConfig, n_realizations: int, settings: str = "testbed", store_full="all") -> RunResult:
    """Train (or load) one condition under results/pulsed/runs.  A run of the same configuration cached under another
    name is reused rather than retrained."""
    root = repo_root().joinpath(*RUNS)
    label = f"{settings}-{name}"
    for h in run_hashes(cfg, n_realizations):
        hits = sorted(root.glob(f"*-{h}/summary.npz")) if root.exists() else []
        if hits and not (root / f"{label}-{h}").exists():
            label = hits[0].parent.name[: -len(h) - 1]
            break
    return run_condition(cfg, n_realizations, label=label, store_full=store_full, settings=settings, root=root)


def sweep_conditions(sweep: dict) -> dict[str, tuple[dict, Pulse | None]]:
    """Name -> (sequence spec, pulse) for a sweep config (configs/pulsed/pulsed_sweep.yaml): the control, then for
    each control parameter its heights at the default width and position, and its default height at every width and
    every position.  Names are canonical, e.g. ``gain+1.5_w0.5_t4.25``."""
    w0, t0 = sweep["default"]["width"], sweep["default"]["t_peak"]
    out: dict[str, tuple[dict, Pulse | None]] = {sweep["control"]: (pulsed_spec(sweep["base"], None), None)}
    for param, c in sweep["controls"].items():
        pulses = [Pulse(param, h, t0, w0) for h in c["heights"]]
        pulses += [Pulse(param, c["default_height"], t0, w) for w in sweep["widths"]]
        pulses += [Pulse(param, c["default_height"], t, w0) for t in sweep["positions"]]
        for p in pulses:
            out.setdefault(f"{p.param}{p.height:+g}_w{p.width:g}_t{p.t_peak:g}", (pulsed_spec(sweep["base"], p), p))
    return out


def train_sweep(sweep: dict, settings: str = "testbed") -> dict[str, RunResult]:
    """Train (or load) every condition of a sweep config."""
    tr = sweep["training"]
    out = {}
    for name, (spec, _) in sweep_conditions(sweep).items():
        cfg = condition_config(spec, tr["n_train"], tr["n_test"], tr.get("passes", 1.0))
        r = run(name, cfg, tr["n_realizations"], settings)
        print(f"{name:32s} test corr {r.summary['corr'][:, -tr['n_test']:].mean():.3f}  {r.path.name}", flush=True)
        out[name] = r
    return out


if __name__ == "__main__":
    import sys

    import yaml

    for path in sys.argv[1:]:
        train_sweep(yaml.safe_load(open(path, encoding="utf-8")))

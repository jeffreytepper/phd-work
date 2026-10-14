"""FORCE training configuration (Braun Methods 'FORCE-training of the RNNs'; spec 6.7) and its stored layouts.

The target is a sequence spec (``seqnet.sequences``), any kind.  Runs cached before the sequences package stored the
target as flat fields (N, T, amp, width, offset) plus an optional ``data`` dict; :func:`legacy_layout` maps a config
back to that layout so those runs keep their hashes, and :meth:`ForceConfig.from_dict` reads either layout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from ..models.rate_rnn import RNNParams
from ..sequences import Sequence, build_sequence

__all__ = ["ForceConfig", "BRAUN_SYNTHETIC", "legacy_layout"]

# Braun's synthetic target (Methods eq. 4-6, Fig. 1)
BRAUN_SYNTHETIC = {"kind": "braun", "N": 501, "T": 8.5, "amp": 0.5, "width": 1.7, "offset": 0.0}

_LEGACY_TARGET = ("N", "T", "amp", "width", "offset")
_RECORDED_KINDS = ("bat", "mec", "npz")


@dataclass
class ForceConfig:
    """One training condition (defaults = Braun's synthetic setting)."""

    # target (seqnet.sequences spec) and its logit (Braun eq. 6)
    sequence: dict = field(default_factory=lambda: dict(BRAUN_SYNTHETIC))
    eps: float = 0.01
    # network (eq. 1-3)
    tau: float = 0.3224
    dt: float = 0.0081
    noise_gain: float = 0.5
    # training (eq. 7-9)
    g_init: float = 1.5
    p0: float = 0.08
    m: int = 16
    steps_per_iter: int = 2086
    bin_size: float = 0.129
    snippet_start_max: float = 119.2
    n_train: int = 300
    n_test: int = 10
    # [VERIFY] switches (spec 6.7)
    error_target: Literal["methods", "supplement"] = "methods"
    c_uses_updated_P: bool = True
    state_reset: bool = True
    target_time: Literal["analytic", "bin"] = "analytic"
    weight_decay: float = 0.0
    mask: dict | None = None  # {"kind": "presyn_random"|"presyn_block"|"synapse_random", "fraction": f}
    seed: int = 0

    def build_sequence(self) -> Sequence:
        return build_sequence(self.sequence)

    def params(self) -> RNNParams:
        return RNNParams(self.tau, self.dt, self.noise_gain)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ForceConfig":
        """From the current layout or the legacy one (flat target fields and ``data``)."""
        d = dict(d)
        if "sequence" not in d:
            target = {k: d.pop(k) for k in _LEGACY_TARGET if k in d}
            data = d.pop("data", None)
            d["sequence"] = dict(data) if data else {**BRAUN_SYNTHETIC, **target}
        return cls(**d)


def legacy_layout(d: dict) -> dict:
    """The pre-sequences layout of a config dict, for targets that layout could express; others are returned as is.

    Braun targets go to the flat fields (``n_images`` must be the default 6); recorded targets to ``data`` with the
    unused flat fields at their old defaults.
    """
    seq = d["sequence"]
    kind = seq.get("kind")
    rest = {k: v for k, v in d.items() if k != "sequence"}
    if kind == "braun" and seq.get("n_images", 6) == 6:
        target = {**BRAUN_SYNTHETIC, **{k: v for k, v in seq.items() if k in _LEGACY_TARGET}}
        return {**{k: target[k] for k in _LEGACY_TARGET}, **rest, "data": None}
    if kind in _RECORDED_KINDS:
        return {**{k: BRAUN_SYNTHETIC[k] for k in _LEGACY_TARGET}, **rest, "data": dict(seq)}
    return d

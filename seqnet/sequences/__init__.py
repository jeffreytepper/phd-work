"""All sequence targets: synthetic generators and prerecorded datasets behind one interface.

Build any sequence from a spec (YAML configs, ``ForceConfig.sequence``)::

    build_sequence({"kind": "braun", "N": 501, "T": 8.5, "amp": 0.5, "width": 1.7, "offset": 0.0})
    build_sequence({"kind": "lehr_bump", "N": 1000, "k": 4, "eps": 0.5, "p_inh": 0.8, "seed": 3,
                    "speed": {"kind": "sigmoid", "low": 500, "high": 2000, "t_mid": 5, "width": 1}})
    build_sequence({"kind": "lehr_network", "N": 500, "gain": {"kind": "step", "before": 1, "after": 3, "t_step": 1}})
    build_sequence({"kind": "bat", "path": "data/bat_omer2023/Fig_2.mat"})

or construct the objects directly (:func:`braun_sequence`, :func:`lehr_bump`, :class:`TravelingBump`,
:class:`LehrRingNetwork`, :class:`RecordedSequence`).  The family runs from Braun et al.'s stationary Gaussian waves to
the shaped, sparse, speed-, width- and subspace-controlled sequences of Lehr, Kumar & Tetzlaff (2024).

Layout:
  base         Sequence interface, Snippet, StationaryWave, the SEQUENCES registry
  bump         parametric travelling bumps (Braun; Lehr Methods eq. 1-7)
  network      Lehr et al. ring network with gain, transient-current and clustered-inhibition control
  sampled      sequences on a time grid
  profiles, timecourses, participation   the swappable parts of a bump
  transforms   rates -> FORCE targets (x*, y*)
  measures     bump position and speed from a rate matrix
  datasets     prerecorded data (bat, MEC, npz) and Braun's preprocessing
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import datasets, measures
from .base import SEQUENCES, Sequence, Snippet, StationaryWave
from .bump import TravelingBump, braun_sequence, lehr_bump
from .datasets import TRAINING_DEFAULTS, RecordedSequence
from .network import LehrRingNetwork, lehr_network
from .participation import PARTICIPATION, AllActive, Participation, RandomSilencing, Switching
from .profiles import PROFILES, BumpProfile, MorphedCosine, RaisedCosine, WrappedGaussian
from .sampled import SampledSequence
from .timecourses import (TIMECOURSES, Constant, Exponential, GaussianPulse, Parabola, Sigmoid, Sinusoid, Step,
                          TimeCourse)
from .transforms import DEFAULT_EPS, logit, logit_dt, logit_target, training_target

__all__ = [
    "build_sequence", "SEQUENCES", "Sequence", "Snippet", "StationaryWave",
    "TravelingBump", "braun_sequence", "lehr_bump", "LehrRingNetwork", "lehr_network", "SampledSequence",
    "RecordedSequence", "TRAINING_DEFAULTS", "datasets", "measures",
    "PROFILES", "BumpProfile", "WrappedGaussian", "RaisedCosine", "MorphedCosine",
    "TIMECOURSES", "TimeCourse", "Constant", "Step", "Exponential", "Sigmoid", "Parabola", "Sinusoid", "GaussianPulse",
    "PARTICIPATION", "Participation", "AllActive", "RandomSilencing", "Switching",
    "DEFAULT_EPS", "logit", "logit_dt", "logit_target", "training_target",
]


@lru_cache(maxsize=16)
def _build_cached(key: str) -> Sequence:
    return SEQUENCES.build(json.loads(key))


def build_sequence(spec: dict | Sequence) -> Sequence:
    """Sequence from a spec ``{"kind": ..., **params}`` (see ``SEQUENCES.kinds()``); cached per process.

    Sequences are immutable, so cached instances (loaded datasets, simulated networks) are shared safely.
    """
    if isinstance(spec, Sequence):
        return spec
    return _build_cached(json.dumps(spec, sort_keys=True))

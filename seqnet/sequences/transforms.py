"""From target rates to the FORCE training targets (Braun Methods eq. 6; Suppl. Def. 1.1).

x* = phi^{-1}(f) = ln((f + eps) / (1 - f + eps)) is the input whose sigmoid gives the rates f; the support condition
asks for y* = x* + tau dx*/dt.  ``convention`` picks which one FORCE trains toward: 'methods' (x*, as printed) or
'supplement' (y*).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .base import Sequence, Snippet

__all__ = ["DEFAULT_EPS", "Convention", "logit", "logit_dt", "logit_target", "training_target"]

DEFAULT_EPS = 0.01
Convention = Literal["methods", "supplement"]


def logit(f: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """x* = ln((f + eps) / (1 - f + eps)) (Braun Methods eq. 6)."""
    f = np.asarray(f, dtype=float)
    return np.log((f + eps) / (1.0 - f + eps))


def logit_dt(f: np.ndarray, fdot: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """dx*/dt = f_dot [1/(f + eps) + 1/(1 - f + eps)] (spec 6.2)."""
    f = np.asarray(f, dtype=float)
    return np.asarray(fdot) * (1.0 / (f + eps) + 1.0 / (1.0 - f + eps))


def logit_target(sequence: Sequence, t, eps: float = DEFAULT_EPS) -> np.ndarray:
    """x*(t) of a sequence, (N, n_t).  ``logit_target(seq, [0.0])[:, 0]`` is the state that starts the sequence."""
    return logit(sequence.rates(np.atleast_1d(np.asarray(t, dtype=float))), eps)


def training_target(snippet: Snippet, convention: Convention, tau: float, eps: float = DEFAULT_EPS) -> np.ndarray:
    """x* ('methods') or y* = x* + tau dx*/dt ('supplement') over a snippet.

    The derivative is exact when the snippet carries ``rates_dt``, else a finite difference of x* (recorded data).
    """
    x = logit(snippet.rates, eps)
    if convention == "methods":
        return x
    if convention != "supplement":
        raise ValueError(f"unknown convention {convention!r}")
    if snippet.rates_dt is not None:
        return x + tau * logit_dt(snippet.rates, snippet.rates_dt, eps)
    return x + tau * np.gradient(x, snippet.spacing, axis=1)

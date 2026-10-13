"""Online learning of the kernel (Thesis §9, App. K.1) and the complete learning system (eq. 9.31).

Two slow-loop rules share the interface ``step(err_power, a) -> a``:

* :class:`ThesisKernelRule` (Thesis eq. 9.10): Delta a_m = eta_k (2 (1 - g_m)/lambda E|eps_m|^2 - nu_k).
  At ridge equilibrium this is gradient descent on the reconstruction error (eq. 9.5-9.6); fixed point
  is the cube-root law (eq. 7.13, 9.29).
* :class:`RidgeObjectiveRule` (spec Section 10.1, beyond the thesis): Delta a_m = eta_k (E|eps_m|^2/lambda - nu_k).
  Since E|eps_m|^2 / lambda = lambda S / (a + lambda)^2 = -d/da [lambda S / (a + lambda)] at ridge
  equilibrium, this descends the full ridge objective; fixed point is the square-root law.

Budget enforcement (spec 6.9): ``'dual'`` (default; nu <- nu + eta_nu (sum a - K0)), ``'fixed'`` (nu
constant) or ``'project'`` (projected step onto {a >= 0, sum a = K0}; nu is read off as the
projection's shift).

Stiffness (finding, results/thesis/REPORT.md).  The plain rule's curvature is
d^2E/da^2 = 6 lambda^2 S / (a + lambda)^4: 6 S / lambda^2 at a = 0 but 3 nu / (a + lambda) at the optimum.
Their ratio 2 S / (lambda nu) * ... is ~1e5-1e8 for typical parameters, so a single eta cannot be both
stable near a = 0 and fast near the optimum.  ``precondition='newton'`` scales each mode's step by the
inverse of the *observable* Hessian, (a_m + lambda) / (3 G_m) with G_m the rule's own gradient
term, which gives every mode the same local rate eta and keeps the fixed point.  It uses only quantities
the rule already has (g_m via a_m, and the error power), so it stays S-free.  Off by default
(the thesis rule as written).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .design import design_matrix
from .errors import error_terms
from .fourier import mode_multiplicity, unitary_dft
from .kernels import Kernel
from .readout import DeltaRule, ridge_readout, spectral_filter
from .ring import Schedule
from .targets import TargetClass

__all__ = [
    "ThesisKernelRule",
    "RidgeObjectiveRule",
    "spatial_error_power",
    "temporal_error_power",
    "EMAEstimator",
    "History",
    "LearningSystem",
    "observable_gradient",
    "analytic_gradient",
]

Budget = Literal["dual", "fixed", "project"]


def analytic_gradient(a: np.ndarray, S: np.ndarray, decay: float) -> np.ndarray:
    """dE(m)/da_m = -2 lambda^2 S(m) / (a_m + lambda)^3 (Thesis eq. 9.5, K.2)."""
    a = np.asarray(a, dtype=float)
    return -2 * decay**2 * np.asarray(S)[: a.size] / (a + decay) ** 3


def observable_gradient(a: np.ndarray, err_power: np.ndarray, decay: float) -> np.ndarray:
    """dE(m)/da_m = -2 (1 - g_m)/lambda E|eps_m|^2 (Thesis eq. 9.6, K.5) -- S(m) cancelled."""
    g = spectral_filter(a, decay)
    return -2 * (1 - g) / decay * np.asarray(err_power)


def _project_budget(a: np.ndarray, G: np.ndarray, D: np.ndarray, mult: np.ndarray, K0: float,
                    elig: np.ndarray) -> tuple[np.ndarray, float]:
    """Projected step in the diagonal metric D: a' = max(a + D (G - theta), 0) with sum mult a' = K0.

    theta (the budget price, i.e. nu) is found by bisection; the budget is monotone decreasing in theta.
    """
    ae, Ge, De, we = a[elig], G[elig], D[elig], mult[elig]
    f = lambda th: float(np.sum(we * np.clip(ae + De * (Ge - th), 0, None))) - K0  # noqa: E731
    lo, hi = min(Ge.min(), 0.0), max(Ge.max(), 1e-300)
    while f(lo) < 0:
        lo = lo - (hi - lo + 1.0) * 10
    while f(hi) > 0:
        hi = hi + (hi - lo + 1.0) * 10
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    th = 0.5 * (lo + hi)
    out = np.zeros_like(a)
    out[elig] = np.clip(ae + De * (Ge - th), 0, None)
    return out, th


Precondition = Literal["none", "linear", "newton"]


@dataclass
class _SlowRule:
    eta: float
    decay: float
    N: int
    K0: float
    budget: Budget = "dual"
    nu: float = 0.0
    eta_nu: float = 0.0
    precondition: Precondition | bool = "none"
    eligible: np.ndarray | None = None
    hessian_factor: float = 3.0  # d^2E/da^2 = hessian_factor * G / (a + lambda) at ridge equilibrium

    def gradient(self, err_power: np.ndarray, a: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _metric(self, G: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Per-mode step scale.  'none': 1 (the thesis rule as written).  'linear': (a + lambda).
        'newton': inverse of the observable Hessian, (a + lambda) / (h G), with h = 3 for the
        reconstruction error and h = 2 for the ridge objective.  Not in the thesis; see module docstring."""
        pc = self.precondition
        if pc is True:
            pc = "linear"
        if pc in (False, "none"):
            return np.ones_like(a)
        if pc == "linear":
            return a + self.decay
        if pc == "newton":
            return (a + self.decay) / (self.hessian_factor * np.maximum(G, 1e-300))
        raise ValueError(pc)

    def step(self, err_power: np.ndarray, a: np.ndarray) -> np.ndarray:
        """One slow-loop update from per-mode error powers E|eps_m|^2 on m = 0..floor(N/2)."""
        a = np.asarray(a, dtype=float)
        mult = mode_multiplicity(self.N)
        elig = np.ones(a.size, bool) if self.eligible is None else np.asarray(self.eligible, bool)
        G = self.gradient(err_power, a)  # descent direction (positive = increase a); G = marginal return
        D = self.eta * self._metric(G, a)
        if self.budget == "project":
            new, th = _project_budget(a, G, D, mult, self.K0, elig)
            self.nu = th
            return new
        new = np.where(elig, np.clip(a + D * (G - self.nu), 0.0, None), 0.0)
        if self.budget == "dual":
            self.nu = max(self.nu + self.eta_nu * (float(np.sum(mult * new)) - self.K0), 0.0)
        return new


@dataclass
class ThesisKernelRule(_SlowRule):
    """Thesis eq. 9.10: Delta a_m = eta_k (2 (1 - g_m) / lambda * E|eps_m|^2 - nu_k)."""

    def gradient(self, err_power, a):
        return -observable_gradient(a, err_power, self.decay)


@dataclass
class RidgeObjectiveRule(_SlowRule):
    """Spec Section 10.1: Delta a_m = eta_k (E|eps_m|^2 / lambda - nu_k); fixed point a = sqrt(lambda S/nu) - lambda."""

    hessian_factor: float = 2.0

    def gradient(self, err_power, a):
        return np.asarray(err_power) / self.decay


def spatial_error_power(errors: np.ndarray, N: int) -> np.ndarray:
    """Per-mode error power |eps_hat_m|^2 from one epoch's errors at T_obs observation positions.

    Unitary DFT over the T_obs observations (Thesis §9.1, eq. 9.2), folded to m = 0..floor(N/2) and
    rescaled by N / T_obs so that it is in the same units as S(m) (normalized to the N-ring).  Modes
    above T_obs/2 are returned as zero (they are aliased into the band).
    """
    e = np.asarray(errors, dtype=float)
    T = e.size
    p = np.abs(unitary_dft(e)) ** 2 * (N / T)
    out = np.zeros(N // 2 + 1)
    n = min(T // 2 + 1, N // 2 + 1)
    out[:n] = p[:n]
    return out


def temporal_error_power(e_t: np.ndarray, n_per_cycle: int, N: int, n_modes: int | None = None) -> np.ndarray:
    """Per-mode error power from a travelling bump: temporal DFT of e(t) over whole cycles at frequency m omega.

    ``e_t`` holds an integer number of cycles sampled ``n_per_cycle`` times per cycle (spec 6A.5, Section
    3.4 fact 5).  The power at harmonic m is averaged over cycles and expressed in N-ring units, so that
    with n_per_cycle = T_obs it equals :func:`spatial_error_power` (test T14 in the main suite).
    """
    e = np.asarray(e_t, dtype=float)
    n_cyc = e.size // n_per_cycle
    if n_cyc < 1:
        raise ValueError("need at least one full cycle")
    cyc = e[: n_cyc * n_per_cycle].reshape(n_cyc, n_per_cycle)
    p = (np.abs(np.fft.fft(cyc, axis=1)) ** 2 / n_per_cycle).mean(0) * (N / n_per_cycle)
    nm = N // 2 + 1 if n_modes is None else n_modes
    out = np.zeros(nm)
    n = min(n_per_cycle // 2 + 1, nm)
    out[:n] = p[:n]
    return out


@dataclass
class EMAEstimator:
    """Exponential moving average of per-mode error powers (spec 6A.5); window = 1/beta epochs."""

    window: float = 1.0
    value: np.ndarray | None = None

    def update(self, x: np.ndarray) -> np.ndarray:
        beta = 1.0 / max(self.window, 1.0)
        self.value = np.array(x, dtype=float) if self.value is None else (1 - beta) * self.value + beta * x
        return self.value


@dataclass
class History:
    """Per-kernel-step record (spec 6A.4)."""

    alloc: list = field(default_factory=list)
    nu: list = field(default_factory=list)
    err_power: list = field(default_factory=list)
    readout_dist: list = field(default_factory=list)
    total_error: list = field(default_factory=list)  # measured sum of squared errors over observations
    bias: list = field(default_factory=list)
    alias: list = field(default_factory=list)
    network: list = field(default_factory=list)
    inner_dist: list = field(default_factory=list)  # per inner sweep |w - w*|/|w*| (when record_inner=True)

    def as_arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


@dataclass
class LearningSystem:
    """The complete learning system (Thesis eq. 9.31): fast readout loop inside a slow kernel loop.

    Parameters
    ----------
    N, decay (lambda), K0 : ring size, ridge regularizer, spectral budget (D10 units).
    target : TargetClass; ``target_mode='fixed'`` draws one target and keeps it (S = |y_hat|^2 of that
        draw), ``'ensemble'`` draws a new random-phase target at every kernel step (spec 6A.5).
    a0 : initial allocation on m = 0..floor(N/2).
    rule : a ThesisKernelRule or RidgeObjectiveRule.
    T_obs : observations per epoch (default N).
    readout : ``'exact'`` (ridge solve each step), ``'batch'`` (mean-field delta-rule epochs) or
        ``'delta'`` (stochastic delta rule, eq. 3.18, random order each epoch).
    eta_w, inner_epochs : fast-loop learning rate and epochs per kernel step (timescale ratio, eq. 9.28).
    ema_window : epochs averaged in the per-mode error estimate.
    """

    N: int
    decay: float
    K0: float
    target: TargetClass
    a0: np.ndarray
    rule: _SlowRule
    T_obs: int | None = None
    readout: Literal["exact", "batch", "delta"] = "exact"
    eta_w: float = 1.0
    inner_epochs: int = 1
    target_mode: Literal["fixed", "ensemble"] = "fixed"
    ema_window: float = 1.0
    sign_policy: str = "positive"
    seed: int = 0
    warn_tol: float = 1e-2
    record_inner: bool = False

    def __post_init__(self) -> None:
        self.T_obs = self.N if self.T_obs is None else self.T_obs
        self.schedule = Schedule.from_T(self.N, self.T_obs)
        self.rng = np.random.default_rng(self.seed)
        self.a = np.asarray(self.a0, dtype=float).copy()
        self.w = np.zeros(self.N)
        self.ema = EMAEstimator(self.ema_window)
        self._coeffs = self.target.coeffs(self.rng)
        self.n_warnings = 0

    def _target_values(self) -> tuple[np.ndarray, np.ndarray]:
        if self.target_mode == "ensemble":
            self._coeffs = self.target.coeffs(self.rng)
        y = self.target.evaluate(self._coeffs, self.N, self.schedule.positions)
        return y, np.abs(self._coeffs) ** 2

    def run(self, n_steps: int) -> History:
        h = History()
        for _ in range(n_steps):
            kernel = Kernel.from_alloc(self.a, self.N, sign_policy=self.sign_policy)  # type: ignore[arg-type]
            X = design_matrix(kernel, self.schedule)
            y, S_draw = self._target_values()
            w_star = ridge_readout(X, y, self.decay)
            if self.readout == "exact":
                self.w = w_star
            else:
                dr = DeltaRule(self.N, self.eta_w, self.decay, self.T_obs, w=self.w.copy())
                nrm = max(np.linalg.norm(w_star), 1e-300)
                inner = [float(np.linalg.norm(dr.w - w_star) / nrm)]
                for _e in range(self.inner_epochs):
                    if self.readout == "batch":
                        dr.batch_epoch(X, y)
                    else:
                        dr.epoch(X, y, order=self.rng.permutation(self.T_obs))
                    if self.record_inner:
                        inner.append(float(np.linalg.norm(dr.w - w_star) / nrm))
                if self.record_inner:
                    h.inner_dist.append(inner)
                self.w = dr.w
            dist = float(np.linalg.norm(self.w - w_star) / max(np.linalg.norm(w_star), 1e-300))
            if dist > self.warn_tol:
                self.n_warnings += 1
            e = y - X @ self.w
            p = self.ema.update(spatial_error_power(e, self.N))
            S_ref = self.target.psd if self.target_mode == "ensemble" else S_draw
            et = error_terms(self.a, S_ref, self.decay, self.T_obs, self.N)
            h.alloc.append(self.a.copy())
            h.nu.append(self.rule.nu)
            h.err_power.append(p.copy())
            h.readout_dist.append(dist)
            h.total_error.append(float(e @ e))
            h.bias.append(et.bias)
            h.alias.append(et.alias)
            h.network.append(et.network)
            self.a = self.rule.step(p, self.a)
        return h

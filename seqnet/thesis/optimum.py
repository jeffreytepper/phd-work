"""The Wiener-optimal kernel and the joint optimum (Thesis §7, §8, App. H, I, J).

All allocations use the D10 operational normalization (a_m = squared singular value of the design
matrix at T_obs = N) and live on m = 0..floor(N/2); budgets count signed modes with
:func:`.fourier.mode_multiplicity`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .errors import error_terms
from .fourier import mode_multiplicity
from .targets import TargetClass

__all__ = [
    "WaterFillResult",
    "water_fill",
    "ridge_fill",
    "min_bias_error",
    "JointResult",
    "joint_optimum",
    "marginal_returns",
    "ridge_marginal_returns",
    "optimality_gap",
    "implied_psd",
    "appendix_I_cases",
    "noisy_alloc",
    "water_fill_noisy",
    "snr",
    "noise_floor_estimate",
]

Law = Literal["cube", "sqrt"]


@dataclass(frozen=True)
class WaterFillResult:
    """Wiener-optimal allocation (Thesis eq. 7.11-7.13) or its square-root analogue (spec Section 10.1)."""

    a: np.ndarray  # allocation on m = 0..floor(N/2)
    nu: float  # budget multiplier nu*
    active: np.ndarray  # bool mask, a_m > 0
    M_star: int  # largest active mode (-1 if none)
    N: int
    T_obs: int
    K0: float
    decay: float
    law: str
    budget_residual: float

    @property
    def rank(self) -> int:
        """r* = 2 M* + 1 (Thesis eq. 10.1)."""
        return 2 * self.M_star + 1 if self.M_star >= 0 else 0

    @property
    def n_active_signed(self) -> int:
        """Number of active signed modes (equals ``rank`` when the active set is contiguous from DC)."""
        mult = mode_multiplicity(self.N)
        return int(round(mult[self.active].sum()))


def _law_constants(law: Law, decay: float) -> tuple[float, float, float]:
    """(c, p, threshold factor) with a = (c S / nu)^p - lambda, active iff S > thr * nu."""
    if law == "cube":
        # Thesis eq. 7.11: a = (2 lambda^2 S / nu)^{1/3} - lambda, active iff S > nu lambda / 2 (eq. 7.8)
        return 2 * decay**2, 1.0 / 3.0, decay / 2.0
    if law == "sqrt":
        # spec Section 10.1: a = sqrt(lambda S / nu) - lambda, active iff S > nu lambda
        return decay, 0.5, decay
    raise ValueError(law)


def _alloc(S: np.ndarray, nu: float, decay: float, law: Law) -> np.ndarray:
    c, p, _ = _law_constants(law, decay)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (c * np.clip(S, 0, None) / nu) ** p - decay
    return np.clip(np.nan_to_num(a, nan=0.0), 0.0, None)


def _fill(S_e: np.ndarray, w_e: np.ndarray, K0: float, decay: float, law: Law) -> float:
    """Exact multiplier: grow the active set in order of decreasing S until KKT-consistent.

    For a fixed active set A the budget sum_A w ((c S/nu)^p - lambda) = K0 gives
    nu^{-p} = (K0 + lambda W_A) / (c^p sum_A w S^p).
    """
    c, p, thr = _law_constants(law, decay)
    order = np.argsort(-S_e, kind="stable")
    S_s, w_s = S_e[order], w_e[order]
    pos = S_s > 0
    S_s, w_s = S_s[pos], w_s[pos]
    if S_s.size == 0 or K0 <= 0:
        return float(S_s[0] / thr) if S_s.size else np.inf
    cum_W = np.cumsum(w_s)
    cum_Sp = np.cumsum(w_s * S_s**p)
    for j in range(S_s.size):
        nu_mp = (K0 + decay * cum_W[j]) / (c**p * cum_Sp[j])  # nu^{-p}
        nu = nu_mp ** (-1.0 / p)
        ok_in = S_s[j] > thr * nu
        ok_out = (j + 1 == S_s.size) or (S_s[j + 1] <= thr * nu)
        if ok_in and ok_out:
            return float(nu)
    raise RuntimeError("water-filling failed to find a consistent active set")


def _bisect_nu(S_e, w_e, K0, decay, law, tol=1e-12, max_iter=400) -> float:
    """Log-space bisection on the budget equation with automatic bracket expansion (spec 6A.5)."""
    c, p, thr = _law_constants(law, decay)
    budget = lambda nu: float(np.sum(w_e * _alloc(S_e, nu, decay, law)))  # noqa: E731
    hi = S_e.max() / thr  # nothing active at or above this nu
    lo = hi
    while budget(lo) < K0:
        lo /= 10.0
        if lo < 1e-300:
            raise RuntimeError("cannot bracket nu")
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(max_iter):
        mid = 0.5 * (llo + lhi)
        r = budget(np.exp(mid)) - K0
        if abs(r) <= tol * max(K0, 1.0):
            return float(np.exp(mid))
        if r > 0:
            llo = mid
        else:
            lhi = mid
    return float(np.exp(0.5 * (llo + lhi)))


def water_fill(
    S: np.ndarray,
    decay: float,
    K0: float,
    T_obs: int | None = None,
    N: int | None = None,
    eligible: np.ndarray | None = None,
    law: Law = "cube",
    method: Literal["exact", "bisect"] = "exact",
) -> WaterFillResult:
    """Wiener-optimal kernel allocation (Thesis eq. 7.11-7.13).

        a*_m = max((2 lambda^2 S(m) / nu*)^{1/3} - lambda, 0)  for |m| <= T_obs/2,  sum_m a*_m = K0.

    Parameters
    ----------
    S : PSD on m = 0..M (M >= floor(N/2) is fine; modes beyond N/2 are ignored here).
    decay : lambda (synaptic decay = ridge regularizer).
    K0 : spectral budget, summed over signed modes (D10 normalization: K0 = N sum_i k_i^2).
    T_obs : observations per epoch; only |m| <= T_obs/2 may receive energy (eq. 7.2).  Default N.
    N : ring size.  Default 2 (len(S) - 1) when T_obs is None, else T_obs.
    eligible : optional bool mask on m = 0..floor(N/2) (e.g. exclude DC, decision D4).
    law : ``'cube'`` (thesis) or ``'sqrt'`` (ridge-objective law of spec Section 10.1).
    method : ``'exact'`` active-set solve (default) or log-space bisection.
    """
    S = np.asarray(S, dtype=float)
    if N is None:
        N = T_obs if T_obs is not None else 2 * (S.size - 1)
    if T_obs is None:
        T_obs = N
    n_half = N // 2 + 1
    if S.size < n_half:
        S = np.concatenate([S, np.zeros(n_half - S.size)])
    S_h = S[:n_half]
    m = np.arange(n_half)
    mult = mode_multiplicity(N)
    elig = m <= T_obs / 2
    if eligible is not None:
        elig &= np.asarray(eligible, dtype=bool)
    S_e = np.where(elig, S_h, 0.0)
    if method == "exact":
        nu = _fill(S_e[elig], mult[elig], K0, decay, law)
    else:
        nu = _bisect_nu(S_e[elig], mult[elig], K0, decay, law)
    a = np.where(elig, _alloc(S_e, nu, decay, law), 0.0)
    active = a > 0
    M_star = int(m[active].max()) if active.any() else -1
    resid = float(np.sum(mult * a) - K0)
    return WaterFillResult(
        a=a, nu=nu, active=active, M_star=M_star, N=N, T_obs=T_obs, K0=K0, decay=decay, law=law,
        budget_residual=resid,
    )


def ridge_fill(S, decay, K0, T_obs=None, N=None, eligible=None) -> WaterFillResult:
    """Square-root water-filling, the stationary point of the full ridge objective (spec Section 10.1).

    Minimizing sum_m lambda S(m) / (a_m + lambda) subject to sum a_m = K0 gives
    a_m = max(sqrt(lambda S(m) / nu) - lambda, 0).
    """
    return water_fill(S, decay, K0, T_obs, N, eligible, law="sqrt")


def min_bias_error(res: WaterFillResult, S: np.ndarray) -> float:
    """Minimum within-band error at the optimum (Thesis eq. 7.17).

    E*_bias = sum_{a*>0} (nu*/2)^{2/3} (lambda^2 S)^{1/3} + sum_{a*=0, |m|<=T/2} S(m).
    """
    S = np.asarray(S, dtype=float)[: res.N // 2 + 1]
    mult = mode_multiplicity(res.N)
    m = np.arange(res.N // 2 + 1)
    band = m <= res.T_obs / 2
    act = res.active
    lam = res.decay
    e_act = (res.nu / 2) ** (2.0 / 3.0) * (lam**2 * S) ** (1.0 / 3.0)
    return float(np.sum((mult * e_act)[act]) + np.sum((mult * S)[band & ~act]))


@dataclass(frozen=True)
class JointResult:
    """Joint kernel-and-sampling optimum (Thesis eq. 8.3-8.9)."""

    a: np.ndarray
    T_star: int
    M_star: int
    nu: float
    budget_residual: float
    method: str
    kappa: float
    E_curve: dict = field(default_factory=dict)  # T -> E*(T) for the numeric method


def joint_optimum(
    S: np.ndarray,
    decay: float,
    B0: float,
    kappa: float | None,
    N: int,
    method: Literal["thesis", "numeric"] = "thesis",
    T_grid: np.ndarray | None = None,
) -> JointResult:
    """Joint optimum over the allocation and the sampling rate T (Thesis eq. 8.3-8.9).

    * ``method='thesis'``: the closed form (8.9) with kappa = lambda/2 (the thesis's identity; D6).
      nu* solves sum_{|m|<=M*(nu)} a*_m(nu) + (lambda/2) 2 M*(nu) = B0 with
      M*(nu) = max{|m| : S(m) >= nu lambda/2}, and T* = 2 M*.  ``kappa`` is ignored (recorded only).
      Because M*(nu) is a step function the budget can jump over B0; the residual is reported.
    * ``method='numeric'``: direct minimization of E*(T) = E*_bias(T) + E_alias(T) + E_network (eq. 8.2)
      over integer T in ``T_grid`` (default even T in [2, N]) with K0 = B0 - kappa T and the inner
      water-filling solved at each T.  Uses the given kappa (D6: kappa is a parameter).
    """
    S = np.asarray(S, dtype=float)
    n_half = N // 2 + 1
    mult = mode_multiplicity(N)
    m = np.arange(n_half)
    S_h = np.concatenate([S, np.zeros(max(0, n_half - S.size))])[:n_half]
    if method == "thesis":
        lam = decay

        def M_of(nu):
            idx = np.nonzero(S_h >= nu * lam / 2)[0]
            return int(idx.max()) if idx.size else -1

        def lhs(nu):
            M = M_of(nu)
            band = m <= M
            a = np.where(band, _alloc(S_h, nu, lam, "cube"), 0.0)
            return float(np.sum(mult * a) + lam / 2 * 2 * max(M, 0)), a, M

        hi = 2 * S_h.max() / lam
        lo = hi
        while lhs(lo)[0] < B0:
            lo /= 10.0
            if lo < 1e-300:
                raise RuntimeError("cannot bracket nu")
        llo, lhi = np.log(lo), np.log(hi)
        for _ in range(500):
            mid = 0.5 * (llo + lhi)
            if lhs(np.exp(mid))[0] > B0:
                llo = mid
            else:
                lhi = mid
        nu = float(np.exp(llo))
        val, a, M = lhs(nu)
        return JointResult(a=a, T_star=2 * M, M_star=M, nu=nu, budget_residual=val - B0, method="thesis",
                           kappa=lam / 2)
    if method == "numeric":
        if kappa is None:
            raise ValueError("numeric joint optimum needs kappa")
        T_grid = np.arange(2, N + 1, 2) if T_grid is None else np.asarray(T_grid)
        best = None
        curve = {}
        for T in T_grid:
            K0 = B0 - kappa * T
            if K0 <= 0:
                continue
            wf = water_fill(S, decay, K0, int(T), N)
            et = error_terms(wf.a, S, decay, int(T), N)
            E = et.total
            curve[int(T)] = E
            if best is None or E < best[0]:
                best = (E, int(T), wf)
        if best is None:
            raise ValueError("budget too small for any T")
        E, T, wf = best
        return JointResult(a=wf.a, T_star=T, M_star=wf.M_star, nu=wf.nu,
                           budget_residual=float(np.sum(mult * wf.a) + kappa * T - B0), method="numeric",
                           kappa=float(kappa), E_curve=curve)
    raise ValueError(method)


def marginal_returns(a: np.ndarray, S: np.ndarray, decay: float) -> np.ndarray:
    """Delta_m(a_m) = 2 lambda^2 S(m) / (a_m + lambda)^3 (Thesis eq. 7.5, 7.15)."""
    a = np.asarray(a, dtype=float)
    S = np.asarray(S, dtype=float)[: a.size]
    return 2 * decay**2 * S / (a + decay) ** 3


def ridge_marginal_returns(a: np.ndarray, S: np.ndarray, decay: float) -> np.ndarray:
    """-d/da of the ridge objective lambda S / (a + lambda): lambda S / (a + lambda)^2 (spec Section 10.1)."""
    a = np.asarray(a, dtype=float)
    S = np.asarray(S, dtype=float)[: a.size]
    return decay * S / (a + decay) ** 2


def optimality_gap(a: np.ndarray, S: np.ndarray, decay: float, law: Law = "cube", tol: float = 0.0) -> dict:
    """Marginal-return diagnostic (Thesis §7.5): uniform on active modes, below nu on inactive ones.

    Returns ``spread`` = (max - min)/mean of the returns on active modes, ``nu_hat`` = their mean, and
    ``kkt_violation`` = max over inactive modes of (return at a = 0 - nu_hat)_+ / nu_hat.
    """
    ret = marginal_returns(a, S, decay) if law == "cube" else ridge_marginal_returns(a, S, decay)
    act = np.asarray(a) > tol
    if not act.any():
        return {"spread": np.nan, "nu_hat": np.nan, "kkt_violation": np.nan}
    r = ret[act]
    nu_hat = float(r.mean())
    zero = ret[~act]  # a = 0 there, so ret is the return at zero
    viol = float(np.clip(zero - nu_hat, 0, None).max() / nu_hat) if zero.size else 0.0
    return {"spread": float((r.max() - r.min()) / nu_hat), "nu_hat": nu_hat, "kkt_violation": viol}


def implied_psd(a: np.ndarray, decay: float, nu: float = 1.0, law: Law = "cube") -> np.ndarray:
    """Target spectrum for which allocation a is optimal, up to the scale nu (inverse of Thesis eq. 7.10).

    Cube law: S(m) = nu (a_m + lambda)^3 / (2 lambda^2) on active modes.  Square-root law:
    S(m) = nu (a_m + lambda)^2 / lambda.  Inactive modes (a_m = 0) are only bounded above by the
    threshold (S <= nu lambda/2, resp. nu lambda); they are returned as NaN.
    """
    a = np.asarray(a, dtype=float)
    if law == "cube":
        S = nu * (a + decay) ** 3 / (2 * decay**2)
    else:
        S = nu * (a + decay) ** 2 / decay
    return np.where(a > 0, S, np.nan)


def appendix_I_cases(N: int = 128, decay: float = 1e-2, K0: float = 100.0) -> list[dict]:
    """The six closed-form target classes of Thesis App. I.7 with their predicted optimum features.

    Each entry: ``name``, ``target`` (TargetClass), ``decay``, ``K0``, ``N`` and ``check`` (a dict of the
    closed-form statements that tests compare against; see tests/thesis/test_optimum.py).
    """
    M = N // 2
    cases = [
        {"name": "constant", "target": TargetClass.constant(M, S0=1.0),
         "check": {"a_dc": K0}},
        {"name": "single_mode", "target": TargetClass.single_mode(M, m0=5, S0=1.0),
         "check": {"m0": 5, "a_m0": K0 / 2}},
        {"name": "flat", "target": TargetClass.flat(M, B=10, S0=1.0),
         "check": {"B": 10, "a_band": K0 / 21, "nu": 2 * decay**2 * 1.0 / (decay + K0 / 21) ** 3}},
        {"name": "gaussian_psd", "target": TargetClass.gaussian_psd(M, beta=0.02, S0=1.0),
         "check": {"exponent": 0.02 / 3}},
        {"name": "polynomial", "target": TargetClass.polynomial(M, s=1.0, S0=1.0),
         "check": {"exponent": -2.0 / 3}},
        {"name": "narrowband", "target": TargetClass.narrowband(M, xi0=20.0, ell=2.0, S0=1.0),
         "check": {"xi0": 20.0, "width": 2.0 * np.sqrt(3)}},
    ]
    for c in cases:
        c.update({"decay": decay, "K0": K0, "N": N})
    return cases


def snr(S: np.ndarray, noise_var: float, T_obs: int) -> np.ndarray:
    """Per-mode signal-to-noise ratio SNR(m) = S(m) / (sigma_eta^2 / T) (Thesis eq. G.8)."""
    return np.asarray(S, dtype=float) / (noise_var / T_obs)


def noise_floor_estimate(err_power: np.ndarray, a: np.ndarray, tol: float = 0.0) -> float:
    """sigma_eta^2 / T estimated as the mean residual power over inactive modes I = {m : a_m = 0} (Thesis eq. G.9)."""
    err_power = np.asarray(err_power, dtype=float)
    inactive = np.asarray(a) <= tol
    if not inactive.any():
        return float("nan")
    return float(err_power[: inactive.size][inactive].mean())


def noisy_alloc(S: np.ndarray, nu: float, decay: float, phi: float) -> np.ndarray:
    """Per-mode root of the noisy optimality condition 2 lambda (lambda S - a phi) / (a + lambda)^3 = nu (eq. G.14).

    phi = sigma_eta^2 / T.  The left side is strictly decreasing in a >= 0, so there is a unique root when
    S > nu lambda / 2 (equivalently, the cubic G.16 has one positive root) and a = 0 otherwise.  Solved by bisection.
    """
    S = np.asarray(S, dtype=float)
    lam = decay
    out = np.zeros_like(S)
    for i, s in enumerate(S):
        if s <= nu * lam / 2:
            continue
        g = lambda x: 2 * lam * (lam * s - x * phi) / (x + lam) ** 3 - nu  # noqa: E731
        hi = max((2 * lam**2 * s / nu) ** (1 / 3), lam)  # noiseless root + lambda bounds the noisy root
        lo = 0.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


def water_fill_noisy(
    S: np.ndarray, decay: float, K0: float, noise_var: float, T_obs: int | None = None, N: int | None = None,
) -> WaterFillResult:
    """Wiener-optimal allocation under observation noise (Thesis App. G.7, eq. G.10-G.16).

    Minimizes sum_m (lambda^2 S + a^2 phi)/(a + lambda)^2 with phi = sigma_eta^2/T subject to sum a <= K0, solving the
    per-mode condition (G.14) for each trial nu and bisecting nu (log space) on the budget.

    Budget as an upper bound (a correction to App. G.7, see REPORT): each mode's noisy error has a finite minimum at
    a = lambda S / phi.  If sum_m lambda S(m)/phi <= K0, spending the whole budget would *raise* the error (the
    equality-constrained problem is then non-convex and needs nu < 0), so the optimum is a = lambda S / phi with nu = 0
    and the budget is slack (reported as a negative ``budget_residual``).
    """
    S = np.asarray(S, dtype=float)
    if N is None:
        N = T_obs if T_obs is not None else 2 * (S.size - 1)
    T_obs = N if T_obs is None else T_obs
    n_half = N // 2 + 1
    S_h = np.concatenate([S, np.zeros(max(0, n_half - S.size))])[:n_half]
    m = np.arange(n_half)
    mult = mode_multiplicity(N)
    S_e = np.where(m <= T_obs / 2, S_h, 0.0)
    phi = noise_var / T_obs
    if phi > 0:
        a_free = decay * S_e / phi  # nu = 0 optimum of each mode
        if float(np.sum(mult * a_free)) <= K0:
            active = a_free > 0
            return WaterFillResult(a=a_free, nu=0.0, active=active, M_star=int(m[active].max()) if active.any() else -1,
                                   N=N, T_obs=T_obs, K0=K0, decay=decay, law="cube_noisy",
                                   budget_residual=float(np.sum(mult * a_free) - K0))
    budget = lambda nu: float(np.sum(mult * noisy_alloc(S_e, nu, decay, phi)))  # noqa: E731
    hi = 2 * S_e.max() / decay
    lo = hi
    while budget(lo) < K0:
        lo /= 10.0
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(200):
        mid = 0.5 * (llo + lhi)
        if budget(np.exp(mid)) > K0:
            llo = mid
        else:
            lhi = mid
    nu = float(np.exp(0.5 * (llo + lhi)))
    a = noisy_alloc(S_e, nu, decay, phi)
    active = a > 0
    return WaterFillResult(a=a, nu=nu, active=active, M_star=int(m[active].max()) if active.any() else -1, N=N,
                           T_obs=T_obs, K0=K0, decay=decay, law="cube_noisy", budget_residual=float(np.sum(mult * a) - K0))

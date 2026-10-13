"""The error decomposition (Thesis §6, App. G): bias, aliasing and network-floor terms.

Mode bookkeeping: allocations ``a`` live on m = 0..floor(N/2); the PSD ``S`` lives on m = 0..M_max
(M_max may exceed N/2).  Sums run over signed modes using the multiplicities of
:func:`.fourier.mode_multiplicity`, with band membership decided by |m|.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .design import design_matrix
from .fourier import mode_multiplicity
from .kernels import Kernel
from .readout import ridge_readout
from .ring import Ring, Schedule
from .targets import TargetClass

__all__ = ["ErrorTerms", "per_mode_error", "error_terms", "sampled_error_exact", "measure_error", "expected_error"]

SamplingScaling = Literal["thesis", "exact"]


def per_mode_error(a: np.ndarray, S: np.ndarray, decay: float, noise_var: float = 0.0, T_obs: int | None = None) -> np.ndarray:
    """E(m) = (lambda^2 S(m) + a_m^2 sigma_eta^2 / T) / (a_m + lambda)^2 (Thesis eq. 6.6, 6.6')."""
    a = np.asarray(a, dtype=float)
    S = np.asarray(S, dtype=float)
    num = decay**2 * S
    if noise_var:
        if T_obs is None:
            raise ValueError("noise term needs T_obs")
        num = num + a**2 * noise_var / T_obs
    return num / (a + decay) ** 2


@dataclass(frozen=True)
class ErrorTerms:
    """Bias, aliasing and network-floor terms (Thesis eq. 6.21-6.22)."""

    bias: float
    alias: float
    network: float
    per_mode: np.ndarray  # per-mode bias contribution on m = 0..T/2 (signed-mode multiplicity applied)

    @property
    def total(self) -> float:
        return self.bias + self.alias + self.network


def error_terms(
    a: np.ndarray,
    S: np.ndarray,
    decay: float,
    T_obs: int,
    N: int,
    noise_var: float = 0.0,
    sampling_scaling: SamplingScaling = "thesis",
) -> ErrorTerms:
    """Three-term decomposition E = E_bias + E_alias + E_network (Thesis eq. 6.7-6.9, 6.21-6.22).

    ``sampling_scaling`` [VERIFY, spec 6A.5]:
      * ``'thesis'``: use a_m as given in eq. 6.7 for every T_obs.
      * ``'exact'``: for T_obs < N the squared singular values of the sampled T x N design matrix are
        (T/N) sum_{m' = m mod T} a_{m'}; for a kernel band-limited to |m| <= T/2 this is (T/N) a_m.
        The per-mode bias then uses a_m T/N.  (Derived in results/thesis/REPORT.md.)
    """
    a = np.asarray(a, dtype=float)
    S = np.asarray(S, dtype=float)
    n_half = N // 2 + 1
    if a.size != n_half:
        raise ValueError("a must live on m = 0..floor(N/2)")
    M_max = S.size - 1
    multS = mode_multiplicity(N, S.size)
    mS = np.arange(S.size)
    a_ext = np.zeros(S.size)
    a_ext[: min(n_half, S.size)] = a[: min(n_half, S.size)]
    a_eff = a_ext * (T_obs / N) if sampling_scaling == "exact" else a_ext
    in_band = mS <= T_obs / 2
    alias_band = (mS > T_obs / 2) & (mS <= N / 2)
    beyond = mS > N / 2
    pm = per_mode_error(a_eff, S, decay, noise_var, T_obs) * multS
    bias = float(pm[in_band].sum())
    alias = float((multS * S)[alias_band].sum())
    network = float((multS * S)[beyond].sum()) if M_max > N / 2 else 0.0
    return ErrorTerms(bias=bias, alias=alias, network=network, per_mode=np.where(in_band, pm, 0.0))


def sampled_error_exact(a: np.ndarray, S: np.ndarray, decay: float, T_obs: int, N: int) -> ErrorTerms:
    """Exact expected error on the continuous ring for a kernel band-limited to |m| <= T_obs/2.

    Derived in results/thesis/REPORT.md (not in the thesis).  With T_obs uniform observations the sampled
    Gram matrix is diagonal in Fourier with entries a~_m = (T_obs/N) a_m, and target content at the aliases
    m + jT_obs (j != 0) is fitted as if it were mode m.  Per in-band mode, with g~ = a~/(a~ + lambda):

        E(m) = (1 - g~)^2 S(m) + g~^2 sum_{j != 0} S(m + j T_obs).

    Out-of-band modes contribute S(m) as in eq. 6.8-6.9.  The thesis's eq. 6.7-6.9 drop the second
    (alias-contamination) term and use a_m instead of a~_m.  The ``alias`` field of the result holds
    the out-of-band energy plus the contamination term; ``network`` holds |m| > N/2.
    """
    a = np.asarray(a, dtype=float)
    S = np.asarray(S, dtype=float)
    M = S.size - 1
    n_half = N // 2 + 1
    mS = np.arange(S.size)
    multS = mode_multiplicity(N, S.size)
    if np.any(a[np.arange(n_half) > T_obs / 2] > 0) or (T_obs % 2 == 0 and T_obs // 2 < n_half and a[T_obs // 2] > 0 and T_obs < N):
        raise ValueError("kernel must be band-limited to |m| < T_obs/2 (strictly, for even T_obs < N)")
    Sfun = lambda m: S[abs(m)] if abs(m) <= M else 0.0  # noqa: E731
    bias = 0.0
    contam = 0.0
    per_mode = np.zeros(S.size)
    for m in range(0, int(T_obs // 2) + 1):
        if m >= n_half:
            break
        at = a[m] * T_obs / N
        g = at / (at + decay)
        aliases = sum(Sfun(m + j * T_obs) for j in range(-(M // T_obs) - 2, M // T_obs + 3) if j != 0)
        w = multS[m]
        b = (1 - g) ** 2 * Sfun(m) * w
        c = g**2 * aliases * w
        bias += b
        contam += c
        per_mode[m] = b + c
    out_band = float((multS * S)[(mS > T_obs / 2) & (mS <= N / 2)].sum())
    network = float((multS * S)[mS > N / 2].sum())
    return ErrorTerms(bias=float(bias), alias=out_band + float(contam), network=network, per_mode=per_mode)


def measure_error(
    kernel: Kernel,
    target: TargetClass,
    schedule: Schedule,
    decay: float,
    rng: np.random.Generator,
    n_draws: int = 1,
    eval_on: Literal["observations", "ring", "fine"] = "observations",
    n_fine: int = 4096,
    noise_var: float = 0.0,
    amplitude: str = "phase",
) -> float:
    """Total reconstruction error measured by simulation (for TT8), averaged over ``n_draws`` targets.

    The readout is the ridge solution (eq. 3.13) on the T_obs observations, optionally with additive
    observation noise of variance ``noise_var``.  The squared error against the clean target is
      * ``'observations'``: summed over the T_obs training positions (the loss of eq. 3.10);
      * ``'ring'``: summed over the N neuron positions c = r_i (units of S(m), N-ring normalization);
      * ``'fine'``: averaged over ``n_fine`` uniform positions and multiplied by N (continuous-ring error
        in the same units).
    """
    N = kernel.N
    X = design_matrix(kernel, schedule)
    c = schedule.positions
    r = Ring(N).positions
    tot = 0.0
    for _ in range(n_draws):
        coeffs = target.coeffs(rng, amplitude)  # type: ignore[arg-type]
        y = target.evaluate(coeffs, N, c)
        y_obs = y + (np.sqrt(noise_var) * rng.normal(size=y.shape) if noise_var else 0.0)
        w = ridge_readout(X, y_obs, decay)
        if eval_on == "observations":
            tot += float(np.sum((X @ w - y) ** 2))
            continue
        ce = r if eval_on == "ring" else 2 * np.pi * np.arange(n_fine) / n_fine
        Xe = kernel((r[None, :] - ce[:, None]).ravel()).reshape(ce.size, N)
        ye = target.evaluate(coeffs, N, ce)
        tot += float(np.sum((Xe @ w - ye) ** 2)) * N / ce.size
    return tot / n_draws


def _target_cov(S: np.ndarray, N: int, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Cov(y(c1), y(c2)) = (1/N) [S(0) + 2 sum_{m>=1} S(m) cos(m (c1 - c2))] for a stationary target (A4)."""
    S = np.asarray(S, dtype=float)
    D = np.subtract.outer(np.asarray(c1, float), np.asarray(c2, float))
    m = np.arange(1, S.size)
    return (S[0] + 2 * np.cos(D[..., None] * m) @ S[1:]) / N


def expected_error(
    kernel: Kernel,
    S: np.ndarray,
    schedule: Schedule,
    decay: float,
    eval_on: Literal["observations", "ring", "fine"] = "observations",
    n_fine: int = 1024,
    noise_var: float = 0.0,
) -> float:
    """Exact expectation of the error measured by :func:`measure_error`, over a stationary target class.

    Linear algebra instead of sampling: w = A y_obs with A = (X^T X + lambda I)^{-1} X^T, so the error at the
    evaluation positions is linear in the Gaussian/random-phase target and its expectation is a trace
    against the target covariance (any distribution with the same second moments gives the same value).
    """
    N = kernel.N
    X = design_matrix(kernel, schedule)
    co = schedule.positions
    r = Ring(N).positions
    A = np.linalg.solve(X.T @ X + decay * np.eye(N), X.T)
    if eval_on == "observations":
        ce, Xe, scale = co, X, 1.0
    else:
        ce = r if eval_on == "ring" else 2 * np.pi * np.arange(n_fine) / n_fine
        Xe = kernel((r[None, :] - ce[:, None]).ravel()).reshape(ce.size, N)
        scale = N / ce.size
    B = Xe @ A
    Coo = _target_cov(S, N, co, co)
    Coe = _target_cov(S, N, co, ce)
    Cee_diag = np.full(ce.size, _target_cov(S, N, np.zeros(1), np.zeros(1))[0, 0])
    val = np.trace(B @ Coo @ B.T) - 2 * np.trace(B @ Coe) + Cee_diag.sum()
    if noise_var:
        val += noise_var * np.sum(B**2)
    return float(val * scale)

"""Activity vectors, the design matrix and its SVD (Thesis §2.2-2.3, §4.4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import Kernel
from .ring import Ring, Schedule

__all__ = ["activity", "design_matrix", "DesignSVD", "design_svd", "observation_subspace", "sampled_effective_rank"]


def activity(kernel: Kernel, c: float | np.ndarray) -> np.ndarray:
    """Activity vector(s) x_i(c) = k(r_i - c) (Thesis eq. 2.1).  Shape (..., N)."""
    r = Ring(kernel.N).positions
    c = np.asarray(c, dtype=float)
    u = r[None, :] - c.reshape(-1, 1)
    out = kernel(u.ravel()).reshape(u.shape)
    return out[0] if c.ndim == 0 else out.reshape(c.shape + (kernel.N,))


def design_matrix(kernel: Kernel, schedule: Schedule) -> np.ndarray:
    """Design matrix X in R^{T x N}, X_{t,i} = k(r_i - c_t) (Thesis eq. 2.4).

    On-grid schedules are built exactly by rolling the kernel; off-grid positions use the
    trigonometric interpolant of the kernel.
    """
    if schedule.N != kernel.N:
        raise ValueError("schedule and kernel disagree on N")
    if schedule.on_grid:
        shifts = np.rint(schedule.positions / (2 * np.pi / kernel.N)).astype(int)
        return np.stack([np.roll(kernel.values, s) for s in shifts])
    return activity(kernel, schedule.positions)


@dataclass(frozen=True)
class DesignSVD:
    """Thin SVD X = U diag(s) V^T restricted to singular values above tolerance (Thesis eq. 2.6)."""

    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.s.size)


def design_svd(X: np.ndarray, rtol: float = 1e-12) -> DesignSVD:
    """Thin SVD of the design matrix keeping singular values > rtol * s_max (Thesis eq. 2.6)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > rtol * s.max()
    return DesignSVD(U=U[:, keep], s=s[keep], Vt=Vt[keep])


def observation_subspace(X: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Orthonormal basis (columns) of S_k = rowspace(X) (Thesis eq. 2.5)."""
    return design_svd(X, rtol).Vt.T


def sampled_effective_rank(X: np.ndarray, eps: float = 1e-2) -> int:
    """Effective rank under sampling: #eigenvalues of G = X X^T above eps * max (Thesis eq. 5.4)."""
    ev = np.linalg.eigvalsh(X @ X.T)
    return int(np.sum(ev > eps * ev.max()))

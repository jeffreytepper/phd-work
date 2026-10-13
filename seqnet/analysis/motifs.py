"""Second-order connectivity motifs (Braun Methods eq. 14-15; spec 6.14).

X = W - mean(W); sigma^2 = tr(X X^T)/N; tau_rec = tr(X X)/(sigma^2 N);
tau_div = (sum_ij (X X^T)_ij - tr(X X^T)) / (sigma^2 N (N-1)); tau_con the same with X^T X;
tau_chn = 2 (sum_ij (X X)_ij - tr(X X)) / (sigma^2 N (N-1)).

For a matrix whose rows and columns all sum to the same value (e.g. any circulant), the centred X has zero row and
column sums, so sum(X X^T) = sum(X^T X) = sum(X X) = 0 and the off-diagonal motifs reduce *exactly* to
tau_div = tau_con = -1/(N-1) and tau_chn = -2 tau_rec/(N-1), i.e. O(1/N), not identically zero (test T15; this
corrects spec 6.14's "vanish identically").
"""
from __future__ import annotations

import numpy as np

__all__ = ["motifs"]


def motifs(W: np.ndarray) -> dict:
    """Return tau_rec, tau_div, tau_con, tau_chn and sigma^2 of W (Braun eq. 14-15)."""
    W = np.asarray(W, dtype=float)
    N = W.shape[0]
    X = W - W.mean()
    s2 = np.sum(X * X) / N  # tr(X X^T) / N
    XXt_sum = np.sum(X.sum(1) ** 2)  # sum_ij (X X^T)_ij = sum_i (row sum)^2
    XtX_sum = np.sum(X.sum(0) ** 2)  # sum_ij (X^T X)_ij = sum_j (column sum)^2
    XX_sum = float(X.sum(0) @ X.sum(1))  # sum_ij (X X)_ij = sum_k colsum_k * rowsum_k
    tr_XXt = np.sum(X * X)
    tr_XX = np.sum(X * X.T)
    den = s2 * N * (N - 1)
    return {
        "sigma2": float(s2),
        "tau_rec": float(tr_XX / (s2 * N)),
        "tau_div": float((XXt_sum - tr_XXt) / den),
        "tau_con": float((XtX_sum - tr_XXt) / den),
        "tau_chn": float(2 * (XX_sum - tr_XX) / den),
    }

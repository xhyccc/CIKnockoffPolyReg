"""PoSI alpha-spending sequences and knockoff+ threshold computation.

Two spending sequences are provided (Section 3 of the paper):

1. Riemann Zeta: q_t = Q * 6 / (pi^2 * t^2)
   Default choice; mathematically sound for unknown iteration count.

2. Geometric:    q_t = Q * (1-gamma) * gamma^(t-1)  with gamma=0.5
   Better statistical power in early steps; optimal for very sparse models.

Both satisfy sum_{t=1}^{inf} q_t <= Q.

The knockoff+ threshold at iteration t is computed as:

    tau_t = min{ tau > 0 :
        (1 + |{j not in A_poly : W_j <= -tau}|)
        / max(1, |{j not in A_poly : W_j >= tau}|)  <= q_t }

where W_j = |beta_hat_j| - |beta_hat_j_tilde| are feature importance
statistics computed from the cross-validated Lasso.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class SpendingSequence(str, Enum):
    """Available alpha-spending sequence types."""

    RIEMANN_ZETA = "riemann_zeta"
    GEOMETRIC = "geometric"


class AlphaSpending:
    """Manages the per-iteration FDR budget q_t.

    Parameters
    ----------
    Q : float
        Global target FDR level in (0, 1).
    sequence : SpendingSequence or str
        ``"riemann_zeta"`` (default) or ``"geometric"``.
    gamma : float
        Geometric decay parameter (only used when sequence="geometric").
        Default 0.5.
    """

    def __init__(
        self,
        Q: float = 0.10,
        sequence: str = "riemann_zeta",
        gamma: float = 0.5,
    ) -> None:
        if not 0 < Q < 1:
            raise ValueError("Q must be in (0, 1)")
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        self.Q = Q
        self.sequence = SpendingSequence(sequence)
        self.gamma = gamma

    def budget(self, t: int) -> float:
        """Return the FDR budget q_t for iteration t (1-indexed).

        Parameters
        ----------
        t : int
            Iteration index, starting from 1.
        """
        if t < 1:
            raise ValueError("t must be >= 1")
        if self.sequence == SpendingSequence.RIEMANN_ZETA:
            return self.Q * 6.0 / (math.pi ** 2 * t ** 2)
        else:  # geometric
            return self.Q * (1.0 - self.gamma) * (self.gamma ** (t - 1))

    def budgets(self, max_t: int) -> NDArray[np.float64]:
        """Return array of budgets q_1, ..., q_{max_t}."""
        return np.array([self.budget(t) for t in range(1, max_t + 1)])


def compute_knockoff_threshold(
    W: NDArray[np.float64],
    q_t: float,
    active_poly: Optional[set[int]] = None,
    offset: int = 1,
) -> float:
    """Compute the knockoff+ threshold tau_t for iteration t.

    Uses the knockoff+ formula (Equation from Section 3):

        tau_t = min{ tau > 0 :
            (offset + |{j not in A_poly : W_j <= -tau}|)
            / max(1, |{j not in A_poly : W_j >= tau}|)  <= q_t }

    where ``offset=1`` gives the knockoff+ (conservative) variant.
    Set ``offset=0`` for the plain knockoff threshold.

    Parameters
    ----------
    W : 1-D array
        Feature importance statistics W_j for all candidate features.
    q_t : float
        FDR budget for current iteration (from AlphaSpending.budget(t)).
    active_poly : set of int or None
        Indices of features already in the active polynomial set.
        These are excluded from the threshold computation.
    offset : int
        1 for knockoff+ (default), 0 for standard knockoff.

    Returns
    -------
    tau : float
        The threshold value.  Returns ``np.inf`` if no tau satisfies the
        condition (i.e., no features are selected this iteration).
    """
    W = np.asarray(W, dtype=np.float64)
    n_features = len(W)
    if active_poly is None:
        active_poly = set()

    # Candidate indices (not already in active set)
    candidate_mask = np.ones(n_features, dtype=bool)
    for idx in active_poly:
        if 0 <= idx < n_features:
            candidate_mask[idx] = False
    W_cand = W[candidate_mask]

    if len(W_cand) == 0:
        return np.inf

    # Candidate threshold values: all unique positive |W| values, sorted asc
    tau_candidates = np.sort(np.unique(np.abs(W_cand[W_cand != 0])))
    if len(tau_candidates) == 0:
        return np.inf

    for tau in tau_candidates:
        n_neg = int(np.sum(W_cand <= -tau))
        n_pos = int(np.sum(W_cand >= tau))
        ratio = (offset + n_neg) / max(1, n_pos)
        if ratio <= q_t:
            return float(tau)

    return np.inf

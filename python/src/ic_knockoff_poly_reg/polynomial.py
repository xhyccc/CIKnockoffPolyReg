"""Polynomial dictionary expansion: Φ(X) = (x, x^2, ..., x^d, 1/x, ..., 1/x^d).

For each base feature x_j the dictionary creates:
  - Positive powers:  x_j^1, x_j^2, ..., x_j^degree
  - Negative powers:  x_j^{-1}, x_j^{-2}, ..., x_j^{-degree}
  - Optional bias/intercept column

This represents the rational polynomial basis described in the paper as
Φ(·) = (·, 1/·, 1)^d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ExpandedFeatures:
    """Result of polynomial dictionary expansion."""

    matrix: NDArray[np.float64]       # (n_samples, n_expanded_features)
    feature_names: list[str]          # descriptive names for each column
    base_feature_indices: list[int]   # which base feature produced each column (-2 for interactions)
    power_exponents: list[int]        # exponent for each column (signed int, or total degree for interactions)
    interaction_indices: list[Optional[list[int]]]  # None for monomials, [i,j,...] for interactions like x_i*x_j


class PolynomialDictionary:
    """Rational polynomial dictionary expansion Φ(X) with feature interactions.

    Expands features into:
    1. Individual monomials: x_j^d, x_j^{-d} for d=1..degree
    2. Interaction terms: products of monomials like x_i^a * x_j^b where |a|+|b| <= degree
    
    Entries that would produce infinities (|x| < clip_threshold)
    are hard-clipped to avoid numerical instabilities.

    Parameters
    ----------
    degree : int
        Maximum absolute exponent (>= 1).
    include_bias : bool
        Whether to append an intercept (constant 1) column.
    include_interactions : bool
        Whether to include interaction terms (e.g., x_i*x_j, x_i^2*x_j).
        Default True for full polynomial basis.
    clip_threshold : float
        Features whose absolute value falls below this are clipped to
        ``±clip_threshold`` before computing negative powers.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        include_interactions: bool = True,
        clip_threshold: float = 1e-8,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.include_bias = include_bias
        self.include_interactions = include_interactions
        self.clip_threshold = clip_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
        self,
        X: NDArray[np.float64],
        base_names: Optional[list[str]] = None,
    ) -> ExpandedFeatures:
        """Expand base feature matrix X (n, p) to polynomial dictionary with interactions.

        Parameters
        ----------
        X : (n_samples, n_features) array
        base_names : optional list of p strings naming the base features

        Returns
        -------
        ExpandedFeatures dataclass
        """
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        if base_names is None:
            base_names = [f"x{j}" for j in range(p)]
        if len(base_names) != p:
            raise ValueError("len(base_names) must equal X.shape[1]")

        # Clip small absolute values to avoid ±inf in negative powers
        X_safe = self._safe_clip(X)

        columns: list[NDArray[np.float64]] = []
        names: list[str] = []
        base_indices: list[int] = []
        exponents: list[int] = []
        interaction_indices: list[Optional[list[int]]] = []

        # Generate individual monomials first
        for j in range(p):
            xj = X_safe[:, j]
            name_j = base_names[j]
            # Positive powers: 1 to degree
            for d in range(1, self.degree + 1):
                columns.append(xj ** d)
                names.append(f"{name_j}^{d}" if d > 1 else name_j)
                base_indices.append(j)
                exponents.append(d)
                interaction_indices.append(None)
            # Negative powers: -1 to -degree
            for d in range(1, self.degree + 1):
                columns.append(xj ** (-d))
                names.append(f"{name_j}^(-{d})")
                base_indices.append(j)
                exponents.append(-d)
                interaction_indices.append(None)

        # Generate interaction terms if requested
        if self.include_interactions and p >= 2:
            # Generate all pairwise interactions with total degree <= self.degree
            for i in range(p):
                for j in range(i + 1, p):
                    xi, xj = X_safe[:, i], X_safe[:, j]
                    name_i, name_j = base_names[i], base_names[j]
                    
                    # Generate all combinations of exponents (di, dj) where |di| + |dj| <= degree
                    # and di, dj are non-zero (otherwise it's just a monomial)
                    for di in range(-self.degree, self.degree + 1):
                        if di == 0:
                            continue
                        for dj in range(-self.degree, self.degree + 1):
                            if dj == 0:
                                continue
                            if abs(di) + abs(dj) <= self.degree:
                                # Create interaction term: xi^di * xj^dj
                                columns.append((xi ** di) * (xj ** dj))
                                
                                # Format name
                                term_i = f"{name_i}^{di}" if di != 1 else name_i
                                if di < 0:
                                    term_i = f"{name_i}^({di})"
                                term_j = f"{name_j}^{dj}" if dj != 1 else name_j
                                if dj < 0:
                                    term_j = f"{name_j}^({dj})"
                                
                                names.append(f"{term_i}*{term_j}")
                                base_indices.append(-2)  # -2 indicates interaction term
                                exponents.append(abs(di) + abs(dj))  # Total degree
                                interaction_indices.append([i, j])

        if self.include_bias:
            columns.append(np.ones(n))
            names.append("1")
            base_indices.append(-1)
            exponents.append(0)
            interaction_indices.append(None)

        matrix = np.column_stack(columns) if columns else np.zeros((n, 0))
        return ExpandedFeatures(
            matrix=matrix,
            feature_names=names,
            base_feature_indices=base_indices,
            power_exponents=exponents,
            interaction_indices=interaction_indices,
        )

    def n_expanded_features(self, n_base: int) -> int:
        """Number of columns produced for n_base base features."""
        return n_base * 2 * self.degree + (1 if self.include_bias else 0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_clip(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip values near zero to ±clip_threshold preserving sign."""
        X_out = X.copy()
        near_zero = np.abs(X_out) < self.clip_threshold
        X_out[near_zero & (X_out >= 0)] = self.clip_threshold
        X_out[near_zero & (X_out < 0)] = -self.clip_threshold
        return X_out

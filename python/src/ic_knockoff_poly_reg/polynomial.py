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
    base_feature_indices: list[int]   # which base feature produced each column
    power_exponents: list[int]        # exponent for each column (signed int)


class PolynomialDictionary:
    """Rational polynomial dictionary expansion Φ(X).

    Expands each feature into positive and negative integer powers up to
    `degree`.  Entries that would produce infinities (|x| < clip_threshold)
    are hard-clipped to avoid numerical instabilities.

    Parameters
    ----------
    degree : int
        Maximum absolute exponent (>= 1).
    include_bias : bool
        Whether to append an intercept (constant 1) column.
    clip_threshold : float
        Features whose absolute value falls below this are clipped to
        ``±clip_threshold`` before computing negative powers.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        clip_threshold: float = 1e-8,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.include_bias = include_bias
        self.clip_threshold = clip_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
        self,
        X: NDArray[np.float64],
        base_names: Optional[list[str]] = None,
    ) -> ExpandedFeatures:
        """Expand base feature matrix X (n, p) to polynomial dictionary.

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

        for j in range(p):
            xj = X_safe[:, j]
            name_j = base_names[j]
            # Positive powers: 1 to degree
            for d in range(1, self.degree + 1):
                columns.append(xj ** d)
                names.append(f"{name_j}^{d}" if d > 1 else name_j)
                base_indices.append(j)
                exponents.append(d)
            # Negative powers: -1 to -degree
            for d in range(1, self.degree + 1):
                columns.append(xj ** (-d))
                names.append(f"{name_j}^(-{d})")
                base_indices.append(j)
                exponents.append(-d)

        if self.include_bias:
            columns.append(np.ones(n))
            names.append("1")
            base_indices.append(-1)
            exponents.append(0)

        matrix = np.column_stack(columns) if columns else np.zeros((n, 0))
        return ExpandedFeatures(
            matrix=matrix,
            feature_names=names,
            base_feature_indices=base_indices,
            power_exponents=exponents,
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

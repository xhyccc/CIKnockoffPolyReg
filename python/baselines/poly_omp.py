"""Baseline: Polynomial Dictionary Expansion + Orthogonal Matching Pursuit.

Expands X with the same rational polynomial dictionary Φ(·) as IC-Knock-Poly,
then fits Orthogonal Matching Pursuit (OMP) with cross-validated sparsity
selection.

OMP is greedy and does not provide FDR guarantees; it serves as a fast
comparison baseline.
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
from ic_knockoff_poly_reg.evaluation import ResultBundle, _compute_fit_stats


class PolyOMP:
    """Polynomial expansion + cross-validated Orthogonal Matching Pursuit.

    Parameters
    ----------
    degree : int
        Maximum absolute polynomial exponent.  Default 2.
    include_bias : bool
        Include constant-1 column.  Default True.
    cv : int
        Cross-validation folds for ``OrthogonalMatchingPursuitCV``.  Default 5.
    max_iter : int or None
        Maximum number of OMP iterations (= maximum sparsity).  ``None`` means
        min(n_samples, n_features).
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        cv: int = 5,
        max_iter: Optional[int] = None,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.cv = cv
        self.max_iter = max_iter

        self.omp_: Optional[OrthogonalMatchingPursuitCV] = None
        self.scaler_: Optional[StandardScaler] = None
        self.poly_dict_: Optional[PolynomialDictionary] = None
        self._feature_names: list[str] = []
        self._base_feature_indices: list[int] = []
        self._power_exponents: list[int] = []

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> "PolyOMP":
        """Fit OMP on the polynomial-expanded feature matrix."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.poly_dict_ = PolynomialDictionary(
            degree=self.degree, include_bias=self.include_bias
        )
        expanded = self.poly_dict_.expand(X)
        Z = expanded.matrix
        self._feature_names = expanded.feature_names
        self._base_feature_indices = expanded.base_feature_indices
        self._power_exponents = expanded.power_exponents

        self.scaler_ = StandardScaler()
        Z_sc = self.scaler_.fit_transform(Z)

        n, p_exp = Z_sc.shape
        max_iter = self.max_iter if self.max_iter is not None else min(n - 1, p_exp)
        max_iter = max(1, max_iter)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.omp_ = OrthogonalMatchingPursuitCV(cv=self.cv, max_iter=max_iter)
            self.omp_.fit(Z_sc, y)

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted OMP model."""
        X = np.asarray(X, dtype=np.float64)
        expanded = self.poly_dict_.expand(X)
        Z_sc = self.scaler_.transform(expanded.matrix)
        return self.omp_.predict(Z_sc)

    @property
    def selected_indices(self) -> list[int]:
        """Indices of expanded features with non-zero OMP coefficients."""
        return [j for j, c in enumerate(self.omp_.coef_) if c != 0.0]

    def to_result_bundle(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        dataset: str = "",
        true_base_indices: Optional[set] = None,
        elapsed_seconds: float = float("nan"),
        peak_memory_mb: float = float("nan"),
    ) -> ResultBundle:
        """Produce a ``ResultBundle`` from this fitted OMP baseline."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        sel_idx = self.selected_indices
        sel_names = [self._feature_names[j] for j in sel_idx]
        sel_base = sorted(set(
            self._base_feature_indices[j]
            for j in sel_idx
            if self._base_feature_indices[j] >= 0
        ))
        sel_terms = [
            [self._base_feature_indices[j], self._power_exponents[j]]
            for j in sel_idx
        ]
        coef = [float(self.omp_.coef_[j]) for j in sel_idx]

        y_pred = self.predict(X)
        n_params = len(coef) + 1
        r2, adj_r2, ss_res, ss_tot, bic, aic = _compute_fit_stats(y, y_pred, n_params)

        fdr = tpr = n_tp = n_fp = n_fn = None
        if true_base_indices is not None:
            true_set = set(true_base_indices)
            sel_set = set(sel_base)
            n_tp = len(sel_set & true_set)
            n_fp = len(sel_set - true_set)
            n_fn = len(true_set - sel_set)
            fdr = n_fp / max(1, len(sel_set))
            tpr = n_tp / max(1, len(true_set))

        return ResultBundle(
            method="poly_omp",
            dataset=dataset,
            selected_names=sel_names,
            selected_base_indices=sel_base,
            selected_terms=sel_terms,
            coef=coef,
            intercept=float(self.omp_.intercept_),
            n_selected=len(sel_idx),
            r_squared=r2,
            adj_r_squared=adj_r2,
            residual_ss=ss_res,
            total_ss=ss_tot,
            bic=bic,
            aic=aic,
            elapsed_seconds=elapsed_seconds,
            peak_memory_mb=peak_memory_mb,
            fdr=fdr,
            tpr=tpr,
            n_true_positives=n_tp,
            n_false_positives=n_fp,
            n_false_negatives=n_fn,
            params={
                "degree": self.degree,
                "include_bias": self.include_bias,
                "cv": self.cv,
                "n_nonzero_coefs": int(self.omp_.n_nonzero_coefs_),
            },
        )

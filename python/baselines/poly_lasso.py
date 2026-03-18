"""Baseline: Polynomial Dictionary Expansion + Cross-validated Lasso.

The simplest baseline: expand X with the same rational polynomial dictionary
Φ(·) used by IC-Knock-Poly, then fit a cross-validated Lasso.

No FDR control is performed; features are selected by non-zero Lasso
coefficients after CV alpha selection.
"""

from __future__ import annotations

import sys
import os
import time
import tracemalloc
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
from ic_knockoff_poly_reg.evaluation import ResultBundle, _compute_fit_stats


class PolyLasso:
    """Polynomial expansion + cross-validated Lasso.

    Parameters
    ----------
    degree : int
        Maximum absolute polynomial exponent.  Default 2.
    include_bias : bool
        Include constant-1 column in the expansion.  Default True.
    cv : int
        Number of cross-validation folds for LassoCV.  Default 5.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        cv: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.cv = cv
        self.random_state = random_state

        self.lasso_: Optional[LassoCV] = None
        self.scaler_: Optional[StandardScaler] = None
        self.poly_dict_: Optional[PolynomialDictionary] = None
        self._feature_names: list[str] = []
        self._base_feature_indices: list[int] = []
        self._power_exponents: list[int] = []

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> "PolyLasso":
        """Fit Lasso on the polynomial-expanded feature matrix.

        Parameters
        ----------
        X : (n_samples, p) feature matrix
        y : (n_samples,) response vector

        Returns
        -------
        self
        """
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lasso_ = LassoCV(
                cv=self.cv, max_iter=5000, random_state=self.random_state
            )
            self.lasso_.fit(Z_sc, y)

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted model."""
        X = np.asarray(X, dtype=np.float64)
        expanded = self.poly_dict_.expand(X)
        Z_sc = self.scaler_.transform(expanded.matrix)
        return self.lasso_.predict(Z_sc)

    @property
    def selected_indices(self) -> list[int]:
        """Indices of expanded features with non-zero Lasso coefficients."""
        return [j for j, c in enumerate(self.lasso_.coef_) if c != 0.0]

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
        """Produce a ``ResultBundle`` from this fitted baseline.

        Parameters
        ----------
        X : feature matrix used during ``fit``
        y : response vector used during ``fit``
        dataset : str
            Dataset name/path for reporting.
        true_base_indices : set of int or None
            Ground-truth base feature indices for FDR/TPR computation.
        elapsed_seconds, peak_memory_mb : float
            Timing and memory from the ``fit`` call.
        """
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
        coef = [float(self.lasso_.coef_[j]) for j in sel_idx]

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
            method="poly_lasso",
            dataset=dataset,
            selected_names=sel_names,
            selected_base_indices=sel_base,
            selected_terms=sel_terms,
            coef=coef,
            intercept=float(self.lasso_.intercept_),
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
                "alpha": float(self.lasso_.alpha_),
            },
        )

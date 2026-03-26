"""Baseline: Sequential Thresholded Least Squares on Polynomial Features.

Implements the STLSQ (Sequential Thresholded Least Squares) algorithm from
SINDy (Brunton et al. 2016) applied to a rational polynomial dictionary.

Algorithm:
  1. Expand X via Φ(·) = (x, 1/x, 1)^d.
  2. Fit OLS on all expanded features.
  3. Set to zero any coefficient with |β_j| < threshold.
  4. Repeat until no coefficients are eliminated or max_iter reached.

STLSQ does not provide FDR guarantees.  The threshold controls sparsity.
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
from ic_knockoff_poly_reg.evaluation import ResultBundle, _compute_fit_stats


class SparsePolySTLSQ:
    """Sequential Thresholded Least Squares on polynomial features (SINDy-style).

    Parameters
    ----------
    degree : int
        Maximum absolute polynomial exponent.  Default 2.
    include_bias : bool
        Include constant-1 column.  Default True.
    threshold : float
        Coefficient magnitude threshold below which terms are pruned.
        If ``None``, determined automatically via cross-validated Ridge as
        ``threshold = 0.1 * max(|β|)`` after the first OLS fit.
    max_iter : int
        Maximum number of thresholding iterations.  Default 20.
    ridge_cv_alphas : list of float or None
        Regularisation candidates for the initial Ridge fit used when
        n > p (prevents blow-up).  Default ``[0.001, 0.01, 0.1, 1.0, 10.0]``.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        threshold: Optional[float] = None,
        max_iter: int = 20,
        ridge_cv_alphas: Optional[list] = None,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.threshold = threshold
        self.max_iter = max_iter
        self.ridge_cv_alphas = ridge_cv_alphas or [0.001, 0.01, 0.1, 1.0, 10.0]

        self.poly_dict_: Optional[PolynomialDictionary] = None
        self._active_mask: Optional[NDArray] = None
        self._coef_full: Optional[NDArray] = None
        self._intercept: float = 0.0
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: list[str] = []
        self._base_feature_indices: list[int] = []
        self._power_exponents: list[int] = []

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> "SparsePolySTLSQ":
        """Fit STLSQ on the polynomial-expanded feature matrix."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        self.poly_dict_ = PolynomialDictionary(
            degree=self.degree, include_bias=self.include_bias
        )
        expanded = self.poly_dict_.expand(X)
        Z = expanded.matrix
        self._feature_names = expanded.feature_names
        self._base_feature_indices = expanded.base_feature_indices
        self._power_exponents = expanded.power_exponents
        n_feat = Z.shape[1]

        self._scaler = StandardScaler()
        Z_sc = self._scaler.fit_transform(Z)

        # Initial fit: Ridge if n < p (prevents rank-deficiency), else OLS
        active = np.ones(n_feat, dtype=bool)
        coef = np.zeros(n_feat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n < n_feat:
                ridge = RidgeCV(alphas=self.ridge_cv_alphas)
                ridge.fit(Z_sc[:, active], y)
                coef[active] = ridge.coef_
                intercept = float(ridge.intercept_)
            else:
                ols = LinearRegression()
                ols.fit(Z_sc[:, active], y)
                coef[active] = ols.coef_
                intercept = float(ols.intercept_)

        # Determine threshold
        threshold = self.threshold
        if threshold is None:
            max_abs = float(np.max(np.abs(coef)))
            threshold = 0.1 * max_abs if max_abs > 0 else 1e-4

        # Iterative thresholding
        for _ in range(self.max_iter):
            new_active = active & (np.abs(coef) >= threshold)
            if not new_active.any():
                # Nothing survives: keep the single largest coefficient
                idx = int(np.argmax(np.abs(coef)))
                new_active = np.zeros(n_feat, dtype=bool)
                new_active[idx] = True

            if np.array_equal(new_active, active):
                break  # Converged

            active = new_active
            coef = np.zeros(n_feat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ols = LinearRegression()
                ols.fit(Z_sc[:, active], y)
                coef[active] = ols.coef_
                intercept = float(ols.intercept_)

        self._active_mask = active
        self._coef_full = coef
        self._intercept = intercept

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted STLSQ model."""
        X = np.asarray(X, dtype=np.float64)
        expanded = self.poly_dict_.expand(X)
        Z_sc = self._scaler.transform(expanded.matrix)
        return Z_sc @ self._coef_full + self._intercept

    @property
    def selected_indices(self) -> list[int]:
        """Indices of active (non-zero) polynomial features."""
        return list(np.where(self._active_mask)[0])

    def to_result_bundle(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        dataset: str = "",
        true_poly_terms: Optional[list] = None,
        elapsed_seconds: float = float("nan"),
        peak_memory_mb: float = float("nan"),
    ) -> ResultBundle:
        """Produce a ``ResultBundle`` from this fitted STLSQ baseline."""
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
        coef = [float(self._coef_full[j]) for j in sel_idx]

        y_pred = self.predict(X)
        n_params = len(coef) + 1
        r2, adj_r2, ss_res, ss_tot, bic, aic = _compute_fit_stats(y, y_pred, n_params)

        # Use polynomial term-level evaluation (CORRECT)
        fdr = tpr = n_tp = n_fp = n_fn = None
        if true_poly_terms is not None:
            from ic_knockoff_poly_reg.evaluation import compute_polynomial_term_metrics
            metrics = compute_polynomial_term_metrics(
                selected_terms=sel_terms,
                true_poly_terms=true_poly_terms,
            )
            fdr = metrics.fdr
            tpr = metrics.tpr
            n_tp = metrics.n_true_positives
            n_fp = metrics.n_false_positives
            n_fn = metrics.n_false_negatives

        return ResultBundle(
            method="sparse_poly_stlsq",
            dataset=dataset,
            selected_names=sel_names,
            selected_base_indices=sel_base,
            selected_terms=sel_terms,
            coef=coef,
            intercept=self._intercept,
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
                "threshold": float(self.threshold) if self.threshold is not None else None,
                "max_iter": self.max_iter,
            },
        )

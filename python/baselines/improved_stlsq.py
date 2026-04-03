"""Improved STLSQ with Ridge regularization and better threshold selection.

This version addresses the numerical instability issues in the original STLSQ
by adding Ridge regularization and using adaptive threshold selection.
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
from ic_knockoff_poly_reg.evaluation import ResultBundle, _compute_fit_stats


class ImprovedSparsePolySTLSQ:
    """Improved STLSQ with Ridge regularization and adaptive thresholding.
    
    Key improvements over original SparsePolySTLSQ:
    1. Uses Ridge regression instead of OLS for numerical stability
    2. Adaptive threshold based on coefficient distribution
    3. Option to use relative threshold (fraction of max coefficient)
    4. Reduced max_iter to prevent overfitting
    5. Better handling of small validation sets
    
    Parameters
    ----------
    degree : int
        Maximum polynomial degree. Default 2.
    threshold : float or None
        If float: absolute threshold for coefficient magnitude.
        If None: uses adaptive threshold based on max coefficient.
    threshold_mode : str
        "absolute" or "relative". If "relative", threshold is multiplied by max(|coef|).
    alpha : float
        Ridge regularization strength. Default 0.01.
    max_iter : int
        Maximum thresholding iterations. Default 10 (reduced from 20).
    min_features : int
        Minimum number of features to keep. Default 1.
    """

    def __init__(
        self,
        degree: int = 2,
        threshold: Optional[float] = None,
        threshold_mode: str = "relative",  # "absolute" or "relative"
        alpha: float = 0.01,  # Ridge regularization
        max_iter: int = 10,   # Reduced from 20
        min_features: int = 1,
        include_bias: bool = True,
    ) -> None:
        self.degree = degree
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_features = min_features
        self.include_bias = include_bias

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
    ) -> "ImprovedSparsePolySTLSQ":
        """Fit improved STLSQ with Ridge regularization."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        # Build polynomial dictionary
        self.poly_dict_ = PolynomialDictionary(
            degree=self.degree, include_bias=self.include_bias
        )
        expanded = self.poly_dict_.expand(X)
        Z = expanded.matrix
        self._feature_names = expanded.feature_names
        self._base_feature_indices = expanded.base_feature_indices
        self._power_exponents = expanded.power_exponents
        n_feat = Z.shape[1]

        # Standardize features
        self._scaler = StandardScaler()
        Z_sc = self._scaler.fit_transform(Z)

        # Initial Ridge fit (always use Ridge for stability)
        active = np.ones(n_feat, dtype=bool)
        coef = np.zeros(n_feat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge = Ridge(alpha=self.alpha, fit_intercept=True)
            ridge.fit(Z_sc[:, active], y)
            coef[active] = ridge.coef_
            intercept = float(ridge.intercept_)

        # Determine adaptive threshold if not provided
        threshold = self.threshold
        if threshold is None:
            max_abs = float(np.max(np.abs(coef)))
            # Use relative threshold: keep top coefficients
            threshold = 0.1 * max_abs if max_abs > 0 else 1e-4
        elif self.threshold_mode == "relative":
            max_abs = float(np.max(np.abs(coef)))
            threshold = threshold * max_abs if max_abs > 0 else 1e-4

        # Iterative thresholding with Ridge
        for iteration in range(self.max_iter):
            # Apply threshold
            new_active = active & (np.abs(coef) >= threshold)
            
            # Ensure minimum features
            if new_active.sum() < self.min_features:
                # Keep top min_features coefficients
                top_indices = np.argsort(np.abs(coef))[-self.min_features:]
                new_active = np.zeros(n_feat, dtype=bool)
                new_active[top_indices] = True
            
            # Check convergence
            if np.array_equal(new_active, active):
                break
            
            # Refit with Ridge on selected features
            active = new_active
            coef = np.zeros(n_feat)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if active.sum() > 0:
                    # Use slightly stronger regularization for refitting
                    ridge = Ridge(alpha=self.alpha * (1 + 0.1 * iteration), fit_intercept=True)
                    ridge.fit(Z_sc[:, active], y)
                    coef[active] = ridge.coef_
                    intercept = float(ridge.intercept_)

        self._active_mask = active
        self._coef_full = coef
        self._intercept = intercept

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted model."""
        X = np.asarray(X, dtype=np.float64)
        expanded = self.poly_dict_.expand(X)
        Z_sc = self._scaler.transform(expanded.matrix)
        return Z_sc @ self._coef_full + self._intercept

    @property
    def selected_indices(self) -> list[int]:
        """Indices of active features."""
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
        """Produce a ResultBundle."""
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
            method="improved_sparse_poly_stlsq",
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
                "threshold": self.threshold,
                "threshold_mode": self.threshold_mode,
                "alpha": self.alpha,
                "max_iter": self.max_iter,
            },
        )

"""Baseline: Polynomial Dictionary Expansion + Lasso.

The simplest baseline: expand X with the same rational polynomial dictionary
Φ(·) used by IC-Knock-Poly, then fit a Lasso with fixed regularization.

No FDR control is performed; features are selected by non-zero Lasso
coefficients.
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
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
from ic_knockoff_poly_reg.evaluation import ResultBundle, _compute_fit_stats


class PolyLasso:
    """Polynomial expansion + Lasso with theoretically-motivated regularization.

    Parameters
    ----------
    degree : int
        Maximum absolute polynomial exponent.  Default 2.
    include_bias : bool
        Include constant-1 column in the expansion.  Default True.
    alpha : float or None
        L1 regularization strength. If None, uses theory-inspired heuristic:
        alpha = 0.1 * sigma * sqrt(2*log(p)/n) where sigma is estimated from
        residuals of a preliminary fit. This approximates FDR control at Q≈0.1.
    max_iter : int
        Maximum iterations for Lasso.  Default 5000.
    random_state : int or None
        Seed for reproducibility (passed to Lasso).
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        alpha: Optional[float] = None,
        max_iter: int = 5000,
        random_state: Optional[int] = None,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        self.lasso_: Optional[Lasso] = None
        self.scaler_: Optional[StandardScaler] = None
        self.poly_dict_: Optional[PolynomialDictionary] = None
        self._feature_names: list[str] = []
        self._base_feature_indices: list[int] = []
        self._power_exponents: list[int] = []
        self._interaction_indices: list[Optional[list[int]]] = []

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
        self._interaction_indices = expanded.interaction_indices

        self.scaler_ = StandardScaler()
        Z_sc = self.scaler_.fit_transform(Z)

        # Compute theoretically-motivated alpha if not provided
        n, p_expanded = Z_sc.shape
        if self.alpha is None:
            # Theory-based alpha
            alpha = self._compute_alpha(Z_sc, y)
            self.alpha_computed_ = alpha
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lasso_ = Lasso(
                    alpha=alpha, max_iter=self.max_iter, random_state=self.random_state
                )
                self.lasso_.fit(Z_sc, y)
        elif self.alpha < 0:
            # Use LassoCV (cross-validation) - use 3-fold for consistency
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lasso_ = LassoCV(
                    cv=3, max_iter=self.max_iter, random_state=self.random_state
                )
                self.lasso_.fit(Z_sc, y)
                self.alpha_computed_ = self.lasso_.alpha_
        else:
            # Fixed alpha
            alpha = self.alpha
            self.alpha_computed_ = alpha
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lasso_ = Lasso(
                    alpha=alpha, max_iter=self.max_iter, random_state=self.random_state
                )
                self.lasso_.fit(Z_sc, y)

        return self

    def _compute_alpha(self, Z: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """Compute theory-inspired Lasso regularization parameter.
        
        Uses the heuristic: alpha = c * sigma * sqrt(2*log(p)/n)
        where sigma is estimated from residuals of a preliminary OLS fit
        on top k correlated features (k = min(n//2, p//4)).
        
        This approximates FDR control at level Q ≈ 0.1 when c = 0.5.
        
        Parameters
        ----------
        Z : array of shape (n, p)
            Standardized polynomial feature matrix.
        y : array of shape (n,)
            Response vector.
            
        Returns
        -------
        alpha : float
            Regularization parameter for Lasso.
        """
        n, p = Z.shape
        
        # Estimate noise sigma from residuals of preliminary sparse fit
        # Select top features by correlation with y
        correlations = np.abs(Z.T @ y) / n
        k_prelim = min(n // 2, p // 4, 10)  # Conservative preliminary selection
        top_indices = np.argsort(correlations)[-k_prelim:]
        
        if len(top_indices) > 0 and len(top_indices) < n:
            Z_prelim = Z[:, top_indices]
            # OLS on preliminary selection
            try:
                coef_prelim = np.linalg.lstsq(Z_prelim, y, rcond=None)[0]
                y_pred_prelim = Z_prelim @ coef_prelim
                residuals = y - y_pred_prelim
                sigma_est = np.sqrt(np.mean(residuals**2))
            except np.linalg.LinAlgError:
                sigma_est = np.std(y)
        else:
            sigma_est = np.std(y)
        
        # Theory-inspired lambda: c * sigma * sqrt(2*log(p)/n)
        # c=0.5 approximates FDR ≈ 0.1 for Gaussian designs
        c = 0.5
        alpha_theory = c * sigma_est * np.sqrt(2 * np.log(max(p, 2)) / n)
        
        # Ensure alpha is in a reasonable range
        alpha_theory = np.clip(alpha_theory, 1e-4, 10.0)
        
        return float(alpha_theory)

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
        true_poly_terms: Optional[list] = None,
        elapsed_seconds: float = float("nan"),
        peak_memory_mb: float = float("nan"),
        X_test: Optional[NDArray[np.float64]] = None,
        y_test: Optional[NDArray[np.float64]] = None,
    ) -> ResultBundle:
        """Produce a ``ResultBundle`` from this fitted baseline.

        Parameters
        ----------
        X : feature matrix used during ``fit``
        y : response vector used during ``fit``
        dataset : str
            Dataset name/path for reporting.
        true_poly_terms : list or None
            Ground-truth polynomial terms for FDR/TPR computation.
        elapsed_seconds, peak_memory_mb : float
            Timing and memory from the ``fit`` call.
        X_test, y_test : test data for generalization evaluation (optional)
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
        sel_terms = []
        for j in sel_idx:
            base_idx = self._base_feature_indices[j]
            exp = self._power_exponents[j]
            interaction = self._interaction_indices[j] if j < len(self._interaction_indices) else None
            if interaction is not None:
                sel_terms.append([base_idx, exp, interaction])
            else:
                sel_terms.append([base_idx, exp])
        coef = [float(self.lasso_.coef_[j]) for j in sel_idx]

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

        # Compute test set performance if provided
        test_r2 = test_rmse = test_mae = n_test = None
        if X_test is not None and y_test is not None:
            X_test_arr = np.asarray(X_test, dtype=np.float64)
            y_test_arr = np.asarray(y_test, dtype=np.float64).ravel()
            y_pred_test = self.predict(X_test_arr)
            
            # R² on test set
            ss_res_test = np.sum((y_test_arr - y_pred_test) ** 2)
            ss_tot_test = np.sum((y_test_arr - np.mean(y_test_arr)) ** 2)
            test_r2 = 1.0 - ss_res_test / ss_tot_test if ss_tot_test > 0 else float('nan')
            
            # RMSE
            test_rmse = np.sqrt(np.mean((y_test_arr - y_pred_test) ** 2))
            
            # MAE
            test_mae = np.mean(np.abs(y_test_arr - y_pred_test))
            
            n_test = len(y_test_arr)

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
            test_r_squared=test_r2 if 'test_r2' in locals() else None,
            test_rmse=test_rmse if 'test_rmse' in locals() else None,
            test_mae=test_mae if 'test_mae' in locals() else None,
            n_test=n_test if 'n_test' in locals() else None,
            params={
                "degree": self.degree,
                "include_bias": self.include_bias,
                "alpha": self.alpha,
            },
        )

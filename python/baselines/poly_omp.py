"""Baseline: Polynomial Dictionary Expansion + Orthogonal Matching Pursuit.

Expands X with the same rational polynomial dictionary Φ(·) as IC-Knock-Poly,
then fits Orthogonal Matching Pursuit (OMP) with fixed sparsity level.

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
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
from ic_knockoff_poly_reg.evaluation import ResultBundle, _compute_fit_stats


class PolyOMP:
    """Polynomial expansion + Orthogonal Matching Pursuit with theory-based sparsity.

    Parameters
    ----------
    degree : int
        Maximum absolute polynomial exponent.  Default 2.
    include_bias : bool
        Include constant-1 column.  Default True.
    n_nonzero_coefs : int or None
        Number of nonzero coefficients (sparsity level).  ``None`` uses
        phase-transition-inspired selection based ONLY on n and p_expanded:
        
            If p_exp > n (underdetermined):
                n_nonzero = min( n/(4*log(p_exp/n)), n/3, p_exp/3, 15 )
            If p_exp <= n (overdetermined):
                n_nonzero = min( min(n,p_exp)/4, n/3, p_exp/3, 15 )
        
        This formula does NOT use the ground-truth sparsity k, ensuring fair
        comparison. It adapts to the problem geometry via compressed sensing
        phase transition theory (Donoho-Tanner)."""

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        n_nonzero_coefs: Optional[int] = None,
        cv: Optional[int] = None,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.n_nonzero_coefs = n_nonzero_coefs
        self.cv = cv

        self.omp_: Optional[OrthogonalMatchingPursuit] = None
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
        # Theory-inspired sparsity selection WITHOUT using ground-truth k.
        #
        # Phase transition for sparse recovery (Donoho-Tanner):
        #   - Underdetermined (p > n): k ~ n / (2*log(p/n))
        #   - Overdetermined (p < n): k ~ min(n, p) / 2
        #
        # We use conservative bounds to avoid overfitting.
        if self.n_nonzero_coefs is not None:
            n_nonzero = self.n_nonzero_coefs
        else:
            # Adaptive phase-transition formula
            if p_exp > n:
                # Underdetermined: use compressed sensing phase transition
                ratio = p_exp / n
                phase_transition = n / (2 * np.log(ratio))
            else:
                # Overdetermined: can use more features
                phase_transition = min(n, p_exp) / 2
            
            n_nonzero = int(min(
                phase_transition / 2,  # Stay below phase transition
                n // 3,                # Ensure stable OLS subproblems
                p_exp // 3,            # Don't select too many features
                15                     # Hard cap for safety
            ))
        n_nonzero = max(1, min(n_nonzero, min(n - 1, p_exp)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.cv is not None:
                # Use cross-validation to select n_nonzero
                from sklearn.model_selection import cross_val_score
                
                best_score = -float('inf')
                best_k = 1
                
                # Try different k values
                k_candidates = [1, 2, 3, 5, 7, 10, 15, 20, 30]
                k_candidates = [k for k in k_candidates if k <= min(n - 1, p_exp)]
                
                for k in k_candidates:
                    try:
                        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
                        scores = cross_val_score(omp, Z_sc, y, cv=self.cv, scoring='r2')
                        mean_score = scores.mean()
                        if mean_score > best_score:
                            best_score = mean_score
                            best_k = k
                    except:
                        continue
                
                n_nonzero = best_k
            
            self.omp_ = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
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
        X_test: Optional[NDArray[np.float64]] = None,
        y_test: Optional[NDArray[np.float64]] = None,
        *,
        dataset: str = "",
        true_poly_terms: Optional[list] = None,
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
            test_r_squared=test_r2,
            test_rmse=test_rmse,
            test_mae=test_mae,
            n_test=n_test,
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
                "n_nonzero_coefs": self.n_nonzero_coefs,
            },
        )

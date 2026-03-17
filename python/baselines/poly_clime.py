"""Baseline: CLIME-based Precision Estimation + One-shot Knockoff Filter.

CLIME (Constrained L1-minimisation for Inverse Matrix Estimation) estimates a
sparse precision matrix Θ by solving:

    min  ||Θ||_1   subject to  ||S Θ − I||_∞ ≤ λ

where S is the sample covariance.

Here we approximate CLIME via sklearn's ``GraphicalLasso`` (which minimises the
Gaussian negative log-likelihood with L1 penalty on Θ, converging to the same
solution as λ → 0).  The fitted precision matrix is then used to generate
equicorrelated Gaussian knockoffs for the *entire* polynomial dictionary in a
single (non-iterative) pass, followed by the knockoff+ threshold with a fixed
FDR level q.

This baseline differs from IC-Knock-Poly in two key ways:
  - Distribution is modelled as a single Gaussian (not a GMM).
  - The knockoff filter is applied once (not iteratively).
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import GraphicalLasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
from ic_knockoff_poly_reg.posi_threshold import compute_knockoff_threshold
from ic_knockoff_poly_reg.evaluation import ResultBundle, _compute_fit_stats


class PolyCLIME:
    """CLIME-based Gaussian knockoff filter on polynomial features.

    Parameters
    ----------
    degree : int
        Maximum absolute polynomial exponent.  Default 2.
    include_bias : bool
        Include constant-1 column.  Default True.
    alpha : float or None
        GraphLasso L1 penalty (CLIME proxy).  ``None`` selects by BIC.
    alpha_grid : list of float or None
        Candidate penalties when ``alpha is None``.
    Q : float
        Target FDR level for the knockoff+ filter.  Default 0.10.
    cv : int
        Cross-validation folds for the inner Lasso.  Default 5.
    random_state : int or None
        Seed for knockoff sampling.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        alpha: Optional[float] = None,
        alpha_grid: Optional[list] = None,
        Q: float = 0.10,
        cv: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.alpha = alpha
        self.alpha_grid = alpha_grid or list(np.logspace(-3, 0, 8))
        self.Q = Q
        self.cv = cv
        self.random_state = random_state

        self.precision_: Optional[NDArray] = None
        self.poly_dict_: Optional[PolynomialDictionary] = None
        self._sel_names: list[str] = []
        self._sel_base: list[int] = []
        self._sel_terms: list = []
        self._coef: list[float] = []
        self._intercept: float = 0.0
        self._feature_names: list[str] = []
        self._base_feature_indices: list[int] = []
        self._power_exponents: list[int] = []

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> "PolyCLIME":
        """Fit CLIME-based knockoff filter."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        # --- Step 1: estimate precision matrix via CLIME (GraphLasso proxy) ---
        alpha = self.alpha if self.alpha is not None else self._select_alpha(X)
        self.precision_ = self._fit_graphlasso(X, alpha)

        # --- Step 2: polynomial expansion ---
        self.poly_dict_ = PolynomialDictionary(
            degree=self.degree, include_bias=self.include_bias
        )
        expanded = self.poly_dict_.expand(X)
        Z = expanded.matrix
        self._feature_names = expanded.feature_names
        self._base_feature_indices = expanded.base_feature_indices
        self._power_exponents = expanded.power_exponents
        n_feat = Z.shape[1]

        # --- Step 3: generate Gaussian knockoffs for original X ---
        mu = X.mean(axis=0)
        cov = np.linalg.inv(self.precision_)
        X_tilde = self._sample_knockoffs(X, mu, cov, rng)

        # Expand knockoffs with the same polynomial dictionary
        expanded_tilde = self.poly_dict_.expand(
            X_tilde, base_names=[f"~x{j}" for j in range(p)]
        )
        Z_tilde = expanded_tilde.matrix

        # --- Step 4: Lasso on [Phi(X) | Phi(X_tilde)] ---
        Z_aug = np.hstack([Z, Z_tilde])
        scaler = StandardScaler()
        Z_aug_sc = scaler.fit_transform(Z_aug)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=self.cv, max_iter=5000, random_state=self.random_state)
            lasso.fit(Z_aug_sc, y)

        beta = lasso.coef_
        beta_orig = beta[:n_feat]
        beta_knock = beta[n_feat:]

        # --- Step 5: W-stats and knockoff+ threshold ---
        W = np.abs(beta_orig) - np.abs(beta_knock)
        tau = compute_knockoff_threshold(W, self.Q)
        if np.isinf(tau):
            sel_local = np.array([], dtype=int)
        else:
            sel_local = np.where(W >= tau)[0]

        self._sel_names = [self._feature_names[j] for j in sel_local]
        self._sel_base = sorted(set(
            self._base_feature_indices[j]
            for j in sel_local
            if self._base_feature_indices[j] >= 0
        ))
        self._sel_terms = [
            [self._base_feature_indices[j], self._power_exponents[j]]
            for j in sel_local
        ]
        self._coef = [float(beta_orig[j]) for j in sel_local]
        self._intercept = float(lasso.intercept_)

        # Store for predict
        self._scaler = scaler
        self._lasso = lasso
        self._n_feat = n_feat

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted CLIME knockoff model."""
        X = np.asarray(X, dtype=np.float64)
        expanded = self.poly_dict_.expand(X)
        # Pad zeros for knockoff columns (not available at prediction time)
        Z_aug = np.hstack([expanded.matrix, np.zeros((X.shape[0], self._n_feat))])
        Z_sc = self._scaler.transform(Z_aug)
        return self._lasso.predict(Z_sc)

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
        """Produce a ``ResultBundle`` from this fitted CLIME baseline."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        y_pred = self.predict(X)
        n_params = len(self._coef) + 1
        r2, adj_r2, ss_res, ss_tot, bic, aic = _compute_fit_stats(y, y_pred, n_params)

        fdr = tpr = n_tp = n_fp = n_fn = None
        if true_base_indices is not None:
            true_set = set(true_base_indices)
            sel_set = set(self._sel_base)
            n_tp = len(sel_set & true_set)
            n_fp = len(sel_set - true_set)
            n_fn = len(true_set - sel_set)
            fdr = n_fp / max(1, len(sel_set))
            tpr = n_tp / max(1, len(true_set))

        return ResultBundle(
            method="poly_clime",
            dataset=dataset,
            selected_names=self._sel_names,
            selected_base_indices=self._sel_base,
            selected_terms=self._sel_terms,
            coef=self._coef,
            intercept=self._intercept,
            n_selected=len(self._sel_names),
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
                "Q": self.Q,
                "alpha": float(self.alpha) if self.alpha is not None else None,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_graphlasso(self, X: NDArray, alpha: float) -> NDArray:
        """Fit GraphLasso and return the precision matrix."""
        n, p = X.shape
        S = np.cov(X, rowvar=False) if p > 1 else np.array([[np.var(X)]])
        if S.ndim == 0:
            S = S.reshape(1, 1)
        S += 1e-6 * np.eye(p)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gl = GraphicalLasso(alpha=alpha, max_iter=500, tol=1e-4)
                gl.fit(
                    np.random.default_rng(0).multivariate_normal(
                        np.zeros(p), S, size=max(p * 2, 50)
                    )
                )
            return gl.precision_
        except Exception:
            return np.diag(1.0 / (np.diag(S) + 1e-6))

    def _select_alpha(self, X: NDArray) -> float:
        """Select GraphLasso alpha by BIC."""
        n, p = X.shape
        best_bic = np.inf
        best_alpha = self.alpha_grid[0]
        for alpha in self.alpha_grid:
            try:
                prec = self._fit_graphlasso(X, alpha)
                cov = np.linalg.inv(prec)
                sign, logdet = np.linalg.slogdet(prec)
                if sign <= 0:
                    continue
                diff = X - X.mean(axis=0)
                mahal = np.sum(diff @ prec * diff)
                ll = 0.5 * n * (logdet - p * np.log(2 * np.pi)) - 0.5 * mahal
                n_params = p * (p + 1) // 2
                bic = -2 * ll + n_params * np.log(n)
                if bic < best_bic:
                    best_bic = bic
                    best_alpha = alpha
            except Exception:
                continue
        return best_alpha

    def _sample_knockoffs(
        self,
        X: NDArray,
        mu: NDArray,
        cov: NDArray,
        rng: np.random.Generator,
    ) -> NDArray:
        """Equicorrelated Gaussian knockoffs."""
        n, p = X.shape
        eigvals = np.linalg.eigvalsh(cov)
        lam_min = max(float(eigvals.min()), 1e-10)
        s_val = min(2.0 * lam_min, float(np.min(np.diag(cov))))
        s_val = max(s_val, 1e-10)
        S_mat = np.diag(np.full(p, s_val))
        prec = np.linalg.inv(cov)
        V_tilde = 2 * S_mat - S_mat @ prec @ S_mat
        V_tilde = (V_tilde + V_tilde.T) / 2
        eigv = np.linalg.eigvalsh(V_tilde)
        V_tilde += max(0.0, -eigv.min() + 1e-10) * np.eye(p)
        chol = np.linalg.cholesky(V_tilde)
        A_mat = (cov - S_mat) @ prec
        noise = rng.standard_normal((n, p)) @ chol.T
        return mu + (X - mu) @ (np.eye(p) - A_mat).T + noise

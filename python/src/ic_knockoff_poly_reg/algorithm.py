"""IC-Knock-Poly: Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression.

Main algorithm pipeline implementing all three phases described in the paper:

  Phase 1 – Unsupervised Base Distribution Learning
      Fit a penalised GMM with GraphLasso precision matrices.

  Phase 2 – Initialisation
      Empty active sets, full residuals R_0 = Y.

  Phase 3 – Iterative Expansion and Screening
      For each iteration t:
        (a) Generate conditional knockoffs for unselected base features.
        (b) Expand base features and knockoffs via polynomial dictionary Φ(·).
        (c) Fit cross-validated Lasso on [Φ(X_B), Φ(X̃_B)] predicting R_{t-1}.
        (d) Compute W_j = |β̂_j| - |β̂_j̃| importance statistics.
        (e) Compute PoSI threshold τ_t using alpha-spending budget q_t.
        (f) Add selected polynomial terms; update active sets and residuals.
        (g) Stop if no new features selected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from .gmm_phase import PenalizedGMM
from .knockoffs import ConditionalKnockoffGenerator
from .polynomial import PolynomialDictionary
from .posi_threshold import AlphaSpending, compute_knockoff_threshold


@dataclass
class ICKnockoffPolyResult:
    """Fitted result from IC-Knock-Poly.

    Attributes
    ----------
    selected_poly_indices : set of int
        Indices of selected polynomial features in the expanded dictionary
        (indices relative to the *last iteration's* expanded feature set).
    selected_poly_names : list of str
        Human-readable names of selected polynomial features.
    selected_base_indices : set of int
        Indices of base features appearing in any selected polynomial term.
    coef : ndarray
        Regression coefficients for the final selected polynomial features.
    intercept : float
        Intercept of the final regression.
    n_iterations : int
        Number of IC-Knock-Poly iterations performed.
    iteration_history : list of dict
        Per-iteration diagnostics (n_selected, tau, q_t, ...).
    """

    selected_poly_indices: set
    selected_poly_names: list
    selected_base_indices: set
    selected_terms: list   # list of (base_idx: int, exponent: int) pairs
    coef: NDArray[np.float64]
    intercept: float
    n_iterations: int
    iteration_history: list


class ICKnockoffPolyReg:
    """Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression.

    Parameters
    ----------
    degree : int
        Maximum absolute polynomial exponent.  Default 2.
    n_components : int
        Number of GMM components for Phase 1.  Default 2.
    gmm_alpha : float or None
        GraphLasso penalty λ.  When None, chosen by BIC.
    Q : float
        Global target FDR level Q ∈ (0, 1).  Default 0.10.
    spending_sequence : str
        ``"riemann_zeta"`` (default) or ``"geometric"``.
    gamma : float
        Geometric decay parameter (only used when spending_sequence=
        ``"geometric"``).  Default 0.5.
    max_iter : int
        Maximum number of IC-Knock-Poly outer iterations.  Default 20.
    include_bias : bool
        Whether to include an intercept in the polynomial dictionary.
        Default True.
    random_state : int or None
        Global seed for reproducibility.
    """

    def __init__(
        self,
        degree: int = 2,
        n_components: int = 2,
        gmm_alpha: Optional[float] = None,
        Q: float = 0.10,
        spending_sequence: str = "riemann_zeta",
        gamma: float = 0.5,
        max_iter: int = 20,
        include_bias: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.degree = degree
        self.n_components = n_components
        self.gmm_alpha = gmm_alpha
        self.Q = Q
        self.spending_sequence = spending_sequence
        self.gamma = gamma
        self.max_iter = max_iter
        self.include_bias = include_bias
        self.random_state = random_state

        # State populated during fit
        self.gmm_: Optional[PenalizedGMM] = None
        self.result_: Optional[ICKnockoffPolyResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> "ICKnockoffPolyReg":
        """Fit the IC-Knock-Poly algorithm.

        Parameters
        ----------
        X : (n_samples, p) base feature matrix
        y : (n_samples,) target vector

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        # ----------------------------------------------------------
        # Phase 1: Fit penalised GMM on base features
        # ----------------------------------------------------------
        self.gmm_ = PenalizedGMM(
            n_components=self.n_components,
            alpha=self.gmm_alpha,
            max_iter=200,
            random_state=self.random_state,
        )
        self.gmm_.fit(X)

        # ----------------------------------------------------------
        # Phase 2: Initialisation
        # ----------------------------------------------------------
        alpha_spender = AlphaSpending(
            Q=self.Q,
            sequence=self.spending_sequence,
            gamma=self.gamma,
        )
        poly_dict = PolynomialDictionary(
            degree=self.degree,
            include_bias=self.include_bias,
        )
        knockoff_gen = ConditionalKnockoffGenerator(
            gmm=self.gmm_,
            random_state=self.random_state,
        )

        # Active sets (indices into *base* features and *poly* features)
        active_base: set[int] = set()
        # Poly active set stores (iteration, local_poly_idx) to track origins;
        # for simplicity we store a growing list of selected feature names
        active_poly_names: list[str] = []
        active_poly_coefs: list[float] = []

        # We keep track of (base_j, exponent) pairs for selected poly features
        selected_terms: list[tuple[int, int]] = []  # (base_idx, exponent)
        selected_names: list[str] = []

        residuals = y.copy()
        iteration_history: list[dict] = []

        scaler = StandardScaler()

        # ----------------------------------------------------------
        # Phase 3: Iterative expansion and screening
        # ----------------------------------------------------------
        for t in range(1, self.max_iter + 1):
            q_t = alpha_spender.budget(t)

            # Unselected base features
            all_base = set(range(p))
            B_indices = np.array(sorted(all_base - active_base), dtype=int)
            A_indices = np.array(sorted(active_base), dtype=int)

            if len(B_indices) == 0:
                break  # All base features already selected

            # (a) Conditional knockoffs for unselected base features
            if len(A_indices) == 0:
                # First iteration: unconditional knockoffs (identity conditioning)
                X_B_tilde = self._unconditional_knockoffs(
                    X[:, B_indices], self.random_state
                )
            else:
                X_B_tilde = knockoff_gen.generate(X, A_indices, B_indices)

            # (b) Polynomial expansion of unselected features AND their knockoffs
            X_B = X[:, B_indices]
            base_names_B = [f"x{j}" for j in B_indices]

            exp_original = poly_dict.expand(X_B, base_names=base_names_B)
            exp_knockoff = poly_dict.expand(
                X_B_tilde, base_names=[f"~x{j}" for j in B_indices]
            )

            Phi_B = exp_original.matrix        # (n, 2*|B|*degree + bias)
            Phi_B_tilde = exp_knockoff.matrix  # same shape

            # Concatenate: [Φ(X_B) | Φ(X̃_B)]
            Z = np.hstack([Phi_B, Phi_B_tilde])
            n_orig = Phi_B.shape[1]

            # (c) Cross-validated Lasso predicting residuals
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z_scaled = scaler.fit_transform(Z)
                lasso = LassoCV(cv=5, max_iter=2000, random_state=self.random_state)
                lasso.fit(Z_scaled, residuals)

            beta = lasso.coef_  # length 2*n_orig

            # (d) W_j = |β̂_j| - |β̂_j̃|
            beta_orig = beta[:n_orig]
            beta_knock = beta[n_orig:]
            W = np.abs(beta_orig) - np.abs(beta_knock)

            # Track which poly features are already in the active poly set.
            # We represent the active poly set as indices into the current
            # candidate list (none are active yet within this round).
            active_poly_current: set[int] = set()

            # (e) Knockoff+ threshold
            tau_t = compute_knockoff_threshold(W, q_t, active_poly_current)

            # (f) Select features with W_j >= tau_t
            if np.isinf(tau_t):
                new_local_indices: NDArray[np.intp] = np.array([], dtype=int)
            else:
                new_local_indices = np.where(W >= tau_t)[0]

            n_new = len(new_local_indices)
            iteration_history.append(
                {
                    "iteration": t,
                    "q_t": q_t,
                    "tau_t": tau_t,
                    "n_candidates": n_orig,
                    "n_new_selected": n_new,
                    "residual_norm": float(np.linalg.norm(residuals)),
                }
            )

            if n_new == 0:
                break  # Stopping criterion: no new features selected

            # Record selected polynomial terms
            for local_idx in new_local_indices:
                base_j = B_indices[exp_original.base_feature_indices[local_idx]]
                exponent = exp_original.power_exponents[local_idx]
                feat_name = exp_original.feature_names[local_idx]
                selected_terms.append((int(base_j), exponent))
                selected_names.append(feat_name)
                # Add the base feature to active_base
                if base_j >= 0:
                    active_base.add(int(base_j))

            # Re-fit on all currently selected polynomial features to get residuals
            residuals = self._update_residuals(X, y, selected_terms, poly_dict)

        # ----------------------------------------------------------
        # Final OLS/Lasso fit on selected polynomial features
        # ----------------------------------------------------------
        coef, intercept = self._final_fit(X, y, selected_terms, poly_dict)

        self.result_ = ICKnockoffPolyResult(
            selected_poly_indices=set(range(len(selected_terms))),
            selected_poly_names=selected_names,
            selected_base_indices=set(b for b, _ in selected_terms if b >= 0),
            selected_terms=list(selected_terms),
            coef=coef,
            intercept=intercept,
            n_iterations=t,
            iteration_history=iteration_history,
        )
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted sparse polynomial model.

        Parameters
        ----------
        X : (n_samples, p) base feature matrix

        Returns
        -------
        y_hat : (n_samples,) predictions
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        poly_dict = PolynomialDictionary(
            degree=self.degree, include_bias=self.include_bias
        )
        Z = self._build_term_matrix(X, self.result_.selected_terms, poly_dict)
        if Z.shape[1] == 0:
            return np.full(X.shape[0], self.result_.intercept)
        return Z @ self.result_.coef + self.result_.intercept

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unconditional_knockoffs(
        self,
        X_B: NDArray[np.float64],
        random_state: Optional[int],
    ) -> NDArray[np.float64]:
        """Gaussian knockoffs for unconditional case (A = empty)."""
        rng = np.random.default_rng(random_state)
        n, p = X_B.shape
        mu = X_B.mean(axis=0)
        S = np.cov(X_B, rowvar=False) if p > 1 else np.var(X_B) * np.ones((1, 1))
        if S.ndim == 0:
            S = S.reshape(1, 1)
        reg = 1e-6 * np.eye(p)
        S += reg

        # Equicorrelated knockoffs
        eigvals = np.linalg.eigvalsh(S)
        lam_min = max(float(eigvals.min()), 1e-10)
        s_val = min(2.0 * lam_min, float(np.min(np.diag(S))))
        s_diag = s_val * np.ones(p)

        S_mat = np.diag(s_diag)
        Sigma_inv = np.linalg.inv(S)
        V_tilde = 2 * S_mat - S_mat @ Sigma_inv @ S_mat
        V_tilde = (V_tilde + V_tilde.T) / 2
        eigv = np.linalg.eigvalsh(V_tilde)
        V_tilde += max(0.0, -eigv.min() + 1e-10) * np.eye(p)

        chol = np.linalg.cholesky(V_tilde)
        A_mat = (S - S_mat) @ Sigma_inv
        noise = rng.standard_normal((n, p)) @ chol.T
        X_tilde = mu + (X_B - mu) @ (np.eye(p) - A_mat).T + noise
        return X_tilde

    def _update_residuals(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        selected_terms: list[tuple[int, int]],
        poly_dict: PolynomialDictionary,
    ) -> NDArray[np.float64]:
        """Regress Y on selected polynomial features; return residuals."""
        if not selected_terms:
            return y.copy()
        Z = self._build_term_matrix(X, selected_terms, poly_dict)
        if Z.shape[1] == 0:
            return y.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=5, max_iter=2000, random_state=self.random_state)
            scaler = StandardScaler()
            Z_sc = scaler.fit_transform(Z)
            lasso.fit(Z_sc, y)
        return y - lasso.predict(Z_sc)

    def _final_fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        selected_terms: list[tuple[int, int]],
        poly_dict: PolynomialDictionary,
    ) -> tuple[NDArray[np.float64], float]:
        """Fit final regression on selected polynomial features."""
        if not selected_terms:
            return np.array([]), float(y.mean())
        Z = self._build_term_matrix(X, selected_terms, poly_dict)
        if Z.shape[1] == 0:
            return np.array([]), float(y.mean())
        # Use Lasso with small alpha for final fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=5, max_iter=2000, random_state=self.random_state)
            scaler = StandardScaler()
            Z_sc = scaler.fit_transform(Z)
            lasso.fit(Z_sc, y)
        return lasso.coef_, lasso.intercept_

    def _build_term_matrix(
        self,
        X: NDArray[np.float64],
        selected_terms: list[tuple[int, int]],
        poly_dict: PolynomialDictionary,
    ) -> NDArray[np.float64]:
        """Build feature matrix from (base_idx, exponent) pairs."""
        if not selected_terms:
            return np.zeros((X.shape[0], 0))
        columns = []
        for base_j, exp in selected_terms:
            if base_j < 0:
                columns.append(np.ones(X.shape[0]))
            else:
                xj = X[:, base_j]
                xj_safe = np.where(
                    np.abs(xj) < poly_dict.clip_threshold,
                    np.sign(xj + 1e-300) * poly_dict.clip_threshold,
                    xj,
                )
                columns.append(xj_safe ** exp)
        return np.column_stack(columns)

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError(
                "ICKnockoffPolyReg must be fitted before calling predict(). "
                "Call fit(X, y) first."
            )

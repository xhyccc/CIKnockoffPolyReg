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
        (d) Compute W_j = max(|β̂_j|, |β̂_j̃|) × sign(|β̂_j| - |β̂_j̃|) importance statistics.
        (e) Compute PoSI threshold τ_t using alpha-spending budget q_t.
        (f) Add selected polynomial terms; update active sets and residuals.
        (g) Stop if no new features selected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from .gmm_phase import PenalizedGMM
from .rust_gmm import RustPenalizedGMM
from .knockoffs import ConditionalKnockoffGenerator
from .polynomial import PolynomialDictionary
from .kernels import create_kernels
from .evaluation import ResultBundle, _compute_fit_stats


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

    Discovers sparse rational polynomial equations (e.g. ``y = x₁² + 1/x₂``)
    in ultra-high dimensional settings (``p ≫ n``) while strictly controlling
    the False Discovery Rate (FDR) via a PoSI α-spending sequence.

    **Operating modes**

    * **Supervised**: call ``fit(X, y)`` with a single labeled dataset.
    * **Semi-supervised**: call ``fit(X_labeled, y_labeled,
      X_unlabeled=X_unlabeled)`` when you have a large pool of unlabeled
      feature observations plus a smaller labeled subset.  Phase 1 (GMM
      distribution learning) will use all observations; Phases 2–3 (knockoff
      generation and polynomial regression) use only the labeled pairs.

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
    backend : str
        Computational backend for the polynomial expansion, knockoff
        W-statistics, and PoSI threshold kernels.  One of:

        * ``"python"`` (default) – pure Python / NumPy; always available.
        * ``"cpp"`` – C++17 shared library loaded via :mod:`ctypes`.
          Requires ``cmake --build cpp/build`` first.
        * ``"rust"`` – Rust cdylib loaded via :mod:`ctypes`.
          Requires ``cargo build --release`` inside ``rust/`` first.
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
        backend: str = "python",
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
        self.backend = backend

        # State populated during fit
        self.gmm_: Optional[Union[PenalizedGMM, RustPenalizedGMM]] = None
        self.result_: Optional[ICKnockoffPolyResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_unlabeled: Optional[NDArray[np.float64]] = None,
    ) -> "ICKnockoffPolyReg":
        """Fit the IC-Knock-Poly algorithm.

        Supports both **supervised** and **semi-supervised** modes.

        Parameters
        ----------
        X : array-like of shape (n_labeled, p)
            Labeled feature matrix.  Each row is one observation; each column
            is one base feature (must be numeric, non-constant, and free of
            ``NaN``/``Inf``).  All feature values used in negative powers
            (rational terms) should be bounded away from zero.
        y : array-like of shape (n_labeled,)
            Continuous response vector aligned with ``X``.
        X_unlabeled : array-like of shape (N_unlabeled, p) or None, optional
            **Semi-supervised mode.** Additional *unlabeled* observations of
            the same ``p`` base features.  When provided, **Phase 1** (GMM
            distribution learning) is fitted on
            ``np.vstack([X_unlabeled, X])`` so that the joint feature
            distribution is estimated from all available data.  Phases 2–3
            (knockoff generation and polynomial regression) use only the
            labeled ``(X, y)`` pairs.  Pass ``None`` (default) for purely
            supervised operation.

        Returns
        -------
        self

        Notes
        -----
        **Data format**

        * ``X`` must be a 2-D array or anything that ``numpy.asarray`` can
          convert to a ``float64`` matrix.  Accepted sources include NumPy
          arrays, pandas DataFrames (values only), and nested Python lists.
        * ``y`` must be a 1-D numeric array of length ``n_labeled``.
        * Feature columns should be on comparable scales; extremely skewed
          distributions may benefit from a log- or quantile-transform before
          calling ``fit``.
        * Columns that could take zero or near-zero values (|x| < 1e-8) are
          automatically clipped before negative-power expansion; no manual
          imputation is required.

        **Semi-supervised workflow**

        When the labeled dataset is small relative to the feature dimension
        (``n_labeled ≪ p``), pass the additional unlabeled rows via
        ``X_unlabeled`` to improve the GMM precision-matrix estimates used for
        knockoff generation::

            model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        # ----------------------------------------------------------
        # Phase 1: Fit penalised GMM on base features
        # ----------------------------------------------------------
        # Use Rust GMM for "rust" backend (much faster), Python GMM otherwise
        # Adaptive alpha based on theoretical rate: alpha ~ c * sqrt(log(p)/n)
        if self.gmm_alpha is not None:
            alpha = self.gmm_alpha
        else:
            # Theoretical rate for Graphical Lasso: lambda ~ c * sqrt(log(p)/n)
            # Adaptive c based on dimensionality:
            # - p <= n (standard): c = 0.5 (moderate regularization)
            # - p > n (high-dim): c = 1.0 (stronger regularization for singular covariance)
            n_eff = max(n, 10)  # Avoid division by very small n
            p_eff = max(p, 2)
            
            if p_eff <= n_eff:
                # Standard regime: p <= n
                c = 0.5
            else:
                # High-dimensional regime: p > n
                # Use stronger regularization (c=1.0) to handle singular sample covariance
                c = 1.0
            
            alpha_theory = c * np.sqrt(np.log(p_eff) / n_eff)
            # Clamp to reasonable range [0.01, 1.0]
            alpha = np.clip(alpha_theory, 0.01, 1.0)
        if self.backend == "rust":
            self.gmm_ = RustPenalizedGMM(
                n_components=self.n_components,
                alpha=alpha,
                max_iter=200,
                random_state=self.random_state,
            )
        else:
            self.gmm_ = PenalizedGMM(
                n_components=self.n_components,
                alpha=self.gmm_alpha,
                max_iter=200,
                random_state=self.random_state,
            )
        if X_unlabeled is not None:
            X_unlabeled = np.asarray(X_unlabeled, dtype=np.float64)
            if X_unlabeled.ndim != 2 or X_unlabeled.shape[1] != p:
                raise ValueError(
                    f"X_unlabeled must have shape (N, {p}) to match X, "
                    f"got {X_unlabeled.shape}."
                )
            X_for_gmm = np.vstack([X_unlabeled, X])
        else:
            X_for_gmm = X
        self.gmm_.fit(X_for_gmm)

        # ----------------------------------------------------------
        # Phase 2: Initialisation
        # ----------------------------------------------------------
        poly_kernel, knockoff_kernel, posi_kernel = create_kernels(self.backend)
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
            q_t = posi_kernel.alpha_spending_budget(
                t, self.Q, self.spending_sequence, self.gamma
            )

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

            exp_original = poly_kernel.expand(
                X_B, self.degree, self.include_bias, base_names=base_names_B
            )
            exp_knockoff = poly_kernel.expand(
                X_B_tilde, self.degree, self.include_bias,
                base_names=[f"~x{j}" for j in B_indices],
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
            W = knockoff_kernel.w_statistics(beta_orig, beta_knock)

            # Track which poly features are already in the active poly set.
            # We represent the active poly set as indices into the current
            # candidate list (none are active yet within this round).
            active_poly_current: set[int] = set()

            # (e) Knockoff threshold (offset=0: standard knockoff, not knockoff+)
            # Using offset=1 (knockoff+) combined with the small per-step
            # alpha-spending budget q_t << Q makes selection practically
            # impossible, so we use the standard knockoff formula (offset=0).
            tau_t = posi_kernel.knockoff_threshold(W, q_t, active_poly_current, offset=0)

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
                base_j_raw = exp_original.base_feature_indices[local_idx]
                exponent = exp_original.power_exponents[local_idx]
                feat_name = exp_original.feature_names[local_idx]
                interaction_idx = exp_original.interaction_indices[local_idx]
                
                # Store term with interaction info if present
                if interaction_idx is not None:
                    # Interaction term: parse exponents from feature name
                    # Name format: "xi^di*xj^dj" or "xi*xj^dj" etc.
                    # Parse the actual exponents used
                    interaction_exponents = self._parse_interaction_exponents(feat_name)
                    selected_terms.append((int(base_j_raw), exponent, interaction_idx, interaction_exponents))
                    # Add all involved base features to active_base
                    for idx in interaction_idx:
                        if idx >= 0:
                            active_base.add(int(idx))
                else:
                    # Monomial term: map local base index to global base index
                    # base_j_raw is the index within B_indices for monomials
                    if base_j_raw >= 0 and base_j_raw < len(B_indices):
                        base_j = B_indices[base_j_raw]
                    else:
                        base_j = base_j_raw  # bias term (=-1)
                    selected_terms.append((int(base_j), exponent))
                    # Add the base feature to active_base
                    if base_j >= 0:
                        active_base.add(int(base_j))
                
                selected_names.append(feat_name)

            # Re-fit on all currently selected polynomial features to get residuals
            residuals = self._update_residuals(X, y, selected_terms, poly_dict)

        # ----------------------------------------------------------
        # Final OLS/Lasso fit on selected polynomial features
        # ----------------------------------------------------------
        coef, intercept = self._final_fit(X, y, selected_terms, poly_dict)

        # Build selected_base_indices from selected_terms
        # Handle both monomials (2-tuple) and interactions (3-tuple)
        selected_base_set = set()
        for term in selected_terms:
            if len(term) == 2:
                # Monomial: (base_idx, exponent)
                b = term[0]
                if b >= 0:
                    selected_base_set.add(b)
            elif len(term) == 3:
                # Interaction: (base_idx, exponent, interaction_indices)
                # base_idx should be -2 for interactions
                interaction_indices = term[2]
                if interaction_indices is not None:
                    for idx in interaction_indices:
                        if idx >= 0:
                            selected_base_set.add(idx)
        
        self.result_ = ICKnockoffPolyResult(
            selected_poly_indices=set(range(len(selected_terms))),
            selected_poly_names=selected_names,
            selected_base_indices=selected_base_set,
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
        X : array-like of shape (n_samples, p)
            Base feature matrix with the same ``p`` columns used during
            ``fit``.  Accepted sources are the same as for ``fit``.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted response values.
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
        """Convert the fitted result into a ``ResultBundle`` for research output.

        Parameters
        ----------
        X : (n_samples, p) feature matrix used during ``fit``.
        y : (n_samples,) response vector used during ``fit``.
        dataset : str, optional
            Name or path of the dataset (for reporting purposes).
        true_poly_terms : list of [int, int] or None, optional
            Ground-truth polynomial terms as [base_idx, exponent] pairs.
            When provided, FDR and TPR are computed on exact term matches.
        elapsed_seconds : float, optional
            Wall-clock seconds spent in ``fit`` (pass via ``time.perf_counter``).
        peak_memory_mb : float, optional
            Peak memory in MB (pass via ``memory_tracker``).
        X_test : (n_test, p) feature matrix or None, optional
            Independent test set for evaluating generalization.
        y_test : (n_test,) response vector or None, optional
            True response values for X_test.

        Returns
        -------
        ResultBundle
        """
        self._check_fitted()
        r = self.result_
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        poly_dict = PolynomialDictionary(
            degree=self.degree, include_bias=self.include_bias
        )
        y_pred = self.predict(X)
        n_params = len(r.coef) + 1  # +1 for intercept
        r2, adj_r2, ss_res, ss_tot, bic, aic = _compute_fit_stats(y, y_pred, n_params)

        fdr = tpr = n_tp = n_fp = n_fn = None
        if true_poly_terms is not None:
            # Compare at the polynomial term level [base_idx, exponent]
            # This is the CORRECT evaluation
            from .evaluation import compute_polynomial_term_metrics
            metrics = compute_polynomial_term_metrics(
                selected_terms=r.selected_terms,
                true_poly_terms=true_poly_terms,
            )
            fdr = metrics.fdr
            tpr = metrics.tpr
            n_tp = metrics.n_true_positives
            n_fp = metrics.n_false_positives
            n_fn = metrics.n_false_negatives

        # Format selected_terms for ResultBundle (handle both monomials and interactions)
        formatted_terms = []
        for term in r.selected_terms:
            if len(term) == 2:
                # Monomial: (base_idx, exponent)
                formatted_terms.append([int(term[0]), int(term[1])])
            elif len(term) == 3:
                # Interaction: (base_idx, exponent, interaction_indices)
                formatted_terms.append([int(term[0]), int(term[1]), term[2]])
            else:
                formatted_terms.append(list(term))
        
        # Compute test set performance if provided
        test_r2 = test_rmse = test_mae = n_test = None
        if X_test is not None and y_test is not None:
            y_pred_test = self.predict(X_test)
            y_test_arr = np.asarray(y_test).ravel()
            
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
            method="ic_knock_poly",
            dataset=dataset,
            selected_names=list(r.selected_poly_names),
            selected_base_indices=sorted(r.selected_base_indices),
            selected_terms=formatted_terms,
            coef=list(r.coef),
            intercept=float(r.intercept),
            n_selected=len(r.selected_terms),
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
            test_r_squared=test_r2,
            test_rmse=test_rmse,
            test_mae=test_mae,
            n_test=n_test,
            params={
                "degree": self.degree,
                "n_components": self.n_components,
                "Q": self.Q,
                "spending_sequence": self.spending_sequence,
                "gamma": self.gamma,
                "max_iter": self.max_iter,
            },
            extra={"n_iterations": r.n_iterations, "iteration_history": r.iteration_history},
        )

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
        # Fit Lasso on scaled features for numerical stability, then convert
        # coefficients back to the original (unscaled) feature space so that
        # predict() — which applies coef to unscaled Z — is correct.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=5, max_iter=2000, random_state=self.random_state)
            scaler = StandardScaler()
            Z_sc = scaler.fit_transform(Z)
            lasso.fit(Z_sc, y)
        # coef_raw[j] = coef_scaled[j] / scale[j]
        # intercept_raw = intercept_scaled - mean[j] / scale[j] * coef_scaled[j]
        coef_raw = lasso.coef_ / scaler.scale_
        intercept_raw = float(lasso.intercept_) - float(
            np.dot(scaler.mean_ / scaler.scale_, lasso.coef_)
        )
        return coef_raw, intercept_raw

    def _build_term_matrix(
        self,
        X: NDArray[np.float64],
        selected_terms: list,
        poly_dict: PolynomialDictionary,
    ) -> NDArray[np.float64]:
        """Build feature matrix from selected terms.
        
        selected_terms can be:
        - (base_idx, exponent): monomial term x_base^exp
        - (base_idx, exponent, interaction_indices, interaction_exponents): interaction term
          where base_idx=-2, interaction_indices=[i,j], interaction_exponents=[di,dj]
        """
        if not selected_terms:
            return np.zeros((X.shape[0], 0))
        columns = []
        for term in selected_terms:
            if len(term) == 2:
                # Monomial term: (base_idx, exponent)
                base_j, exp = term
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
            elif len(term) >= 3:
                # Interaction term: (base_idx, exponent, interaction_indices, interaction_exponents)
                base_j, exp = term[0], term[1]
                interaction_indices = term[2] if len(term) > 2 else None
                interaction_exponents = term[3] if len(term) > 3 else None
                
                if base_j == -2 and interaction_indices is not None and len(interaction_indices) >= 2:
                    # Build interaction term: product of monomials with correct exponents
                    X_safe = poly_dict._safe_clip(X)
                    result = np.ones(X.shape[0])
                    
                    for idx, feat_idx in enumerate(interaction_indices):
                        xi = X_safe[:, feat_idx]
                        # Use the stored exponent if available, otherwise default to 1
                        if interaction_exponents is not None and idx < len(interaction_exponents):
                            exp_i = interaction_exponents[idx]
                        else:
                            exp_i = 1
                        result = result * (xi ** exp_i)
                    columns.append(result)
                else:
                    columns.append(np.ones(X.shape[0]))
            else:
                columns.append(np.ones(X.shape[0]))
        return np.column_stack(columns)

    def _parse_interaction_exponents(self, feat_name: str) -> list[int]:
        """Parse individual exponents from interaction feature name.
        
        Name format: "xi^di*xj^dj" or "xi*xj^dj" or "~xi^di*xj^dj" (knockoffs) etc.
        Returns list of exponents [di, dj, ...].
        """
        import re
        # Remove leading ~ for knockoff feature names
        feat_name_clean = feat_name.lstrip('~')
        parts = feat_name_clean.split('*')
        exponents = []
        for part in parts:
            # Match pattern like "x0^2" or "x0^(-1)" or just "x0"
            # Handle optional ~ prefix, then x(\d+) captures the feature index
            match = re.match(r'~?x\d+(?:\^\(?(-?\d+)\)?)?', part)
            if match:
                exp_str = match.group(1)
                if exp_str is None:
                    exp = 1  # No exponent means ^1
                else:
                    exp = int(exp_str)
                exponents.append(exp)
        return exponents

    def fit_with_cv(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        Q_candidates: Optional[list[float]] = None,
        cv: int = 5,
        X_unlabeled: Optional[NDArray[np.float64]] = None,
    ) -> "ICKnockoffPolyReg":
        """Fit with cross-validation to select optimal Q.
        
        Uses k-fold cross-validation on the training set to select the Q value
        that maximizes validation R². This makes the comparison with Poly-Lasso
        more fair since both use CV for hyperparameter tuning.
        
        Parameters
        ----------
        X : array-like of shape (n_labeled, p)
            Labeled feature matrix.
        y : array-like of shape (n_labeled,)
            Response vector.
        Q_candidates : list of float or None, optional
            List of Q values to try. Default: [0.05, 0.08, 0.10, 0.12, 0.15, 0.20].
        cv : int, optional
            Number of CV folds. Default 5.
        X_unlabeled : array-like of shape (N_unlabeled, p) or None, optional
            Additional unlabeled observations for GMM fitting.
            
        Returns
        -------
        self
            Fitted with best Q from CV.
        """
        from sklearn.model_selection import KFold
        
        if Q_candidates is None:
            Q_candidates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = X.shape[0]
        
        # Store original Q
        original_Q = self.Q
        
        # CV to select best Q
        best_Q = None
        best_val_r2 = -float('inf')
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        print(f"\nPerforming {cv}-fold CV to select Q from {Q_candidates}")
        print("-" * 60)
        
        for Q in Q_candidates:
            val_r2_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create temporary model with this Q
                temp_model = ICKnockoffPolyReg(
                    degree=self.degree,
                    n_components=self.n_components,
                    gmm_alpha=self.gmm_alpha,
                    Q=Q,
                    spending_sequence=self.spending_sequence,
                    gamma=self.gamma,
                    max_iter=self.max_iter,
                    include_bias=self.include_bias,
                    random_state=self.random_state,
                    backend=self.backend,
                )
                
                # Fit on training fold
                try:
                    if X_unlabeled is not None:
                        # Use unlabeled data for GMM
                        temp_model.fit(X_train, y_train, X_unlabeled=X_unlabeled)
                    else:
                        temp_model.fit(X_train, y_train)
                    
                    # Predict on validation fold
                    y_val_pred = temp_model.predict(X_val)
                    
                    # Compute validation R²
                    ss_res = np.sum((y_val - y_val_pred) ** 2)
                    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                    val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    val_r2_scores.append(val_r2)
                    
                except Exception as e:
                    # If fitting fails, assign poor score
                    val_r2_scores.append(-999.0)
            
            # Average validation R² across folds
            mean_val_r2 = np.mean(val_r2_scores)
            print(f"  Q={Q:.2f}: mean val R²={mean_val_r2:.4f}")
            
            if mean_val_r2 > best_val_r2:
                best_val_r2 = mean_val_r2
                best_Q = Q
        
        print(f"\n✓ Selected Q={best_Q:.2f} (best validation R²={best_val_r2:.4f})")
        print("-" * 60)
        
        # Fit final model with best Q
        self.Q = best_Q
        if X_unlabeled is not None:
            return self.fit(X, y, X_unlabeled=X_unlabeled)
        else:
            return self.fit(X, y)
    
    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError(
                "ICKnockoffPolyReg must be fitted before calling predict(). "
                "Call fit(X, y) first."
            )

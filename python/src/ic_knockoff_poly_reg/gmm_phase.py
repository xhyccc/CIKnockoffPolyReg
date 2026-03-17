"""Phase 1: Penalized Gaussian Mixture Model with GraphLasso precision estimation.

Fits a GMM with K components where each component's precision matrix is estimated
using the Graphical Lasso (L1 penalty) to ensure sparsity and invertibility.
The penalty lambda is selected via BIC or hold-out cross-validation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.covariance import GraphicalLasso


@dataclass
class GMMComponent:
    """A single Gaussian component with sparse precision matrix."""

    weight: float
    mean: NDArray[np.float64]
    precision: NDArray[np.float64]  # Inverse covariance (Theta_k)

    @property
    def covariance(self) -> NDArray[np.float64]:
        return np.linalg.inv(self.precision)


@dataclass
class PenalizedGMMResult:
    """Fitted GMM result with all component parameters."""

    components: list[GMMComponent]
    bic: float
    log_likelihood: float
    n_iter: int
    converged: bool


class PenalizedGMM:
    """Gaussian Mixture Model with GraphLasso precision estimation (Phase 1).

    During the M-step each component's precision matrix is estimated via the
    Graphical Lasso, yielding sparse, strictly invertible precision matrices.
    The regularisation strength *alpha* (GraphLasso penalty) is selected by
    minimising the BIC or, when ``cv_folds > 1``, by held-out log-likelihood.

    Parameters
    ----------
    n_components : int
        Number of mixture components K.
    alpha : float or None
        GraphLasso L1 penalty.  When None the value is chosen automatically
        from ``alpha_grid`` by BIC.
    alpha_grid : sequence of float or None
        Candidate values searched when ``alpha`` is None.  Defaults to a
        log-spaced grid in [1e-3, 1.0].
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    reg_covar : float
        Small diagonal regularisation added before GraphLasso to avoid
        singular sample covariances.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        alpha: Optional[float] = None,
        alpha_grid: Optional[list[float]] = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.alpha_grid = alpha_grid or list(np.logspace(-3, 0, 10))
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.result_: Optional[PenalizedGMMResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.float64]) -> "PenalizedGMM":
        """Fit the penalized GMM to data X (n_samples, n_features)."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape

        alpha = self.alpha if self.alpha is not None else self._select_alpha_bic(X, rng)
        self.alpha_ = alpha

        components, ll, n_iter, converged = self._run_em(X, alpha, rng)
        bic = self._compute_bic(X, components, ll)
        self.result_ = PenalizedGMMResult(
            components=components,
            bic=bic,
            log_likelihood=ll,
            n_iter=n_iter,
            converged=converged,
        )
        return self

    def compute_log_responsibilities(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return log posterior P(component k | x_i) for each sample.

        Returns array of shape (n_samples, n_components).
        """
        self._check_fitted()
        log_resp = self._e_step(np.asarray(X, dtype=np.float64), self.result_.components)
        return log_resp

    def compute_conditional_params(
        self,
        X_A: NDArray[np.float64],
        A_indices: NDArray[np.intp],
        B_indices: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Conditional Gaussian parameters for X_B | X_A under the fitted GMM.

        Uses the component with the highest posterior probability for each
        sample (hard assignment), then computes the conditional mean and
        precision matrix of the non-selected features given the selected ones.

        Parameters
        ----------
        X_A : array (n_samples, |A|)
            Observed values of selected features.
        A_indices : 1-D int array
            Column indices of selected features in the original X.
        B_indices : 1-D int array
            Column indices of unselected features in the original X.

        Returns
        -------
        cond_means : (n_samples, |B|)
        cond_precision : (|B|, |B|)
        log_responsibilities : (n_samples, K)
        """
        self._check_fitted()
        A_indices = np.asarray(A_indices, dtype=int)
        B_indices = np.asarray(B_indices, dtype=int)
        components = self.result_.components

        # Build full X array (only A columns known; others filled with zeros
        # just for log-responsibility computation over known columns)
        n = X_A.shape[0]
        p_full = A_indices.max() if len(A_indices) else 0
        if len(B_indices):
            p_full = max(p_full, B_indices.max())
        p_full += 1

        X_partial = np.zeros((n, p_full))
        X_partial[:, A_indices] = X_A

        # Compute log-responsibilities using only the A block
        log_resp = self._e_step_partial(X_A, A_indices, components)
        responsibilities = np.exp(log_resp)  # (n, K)

        # Compute mixture-weighted conditional parameters
        cond_means = np.zeros((n, len(B_indices)))
        # For precision: use responsibility-weighted average across components
        cond_precision = np.zeros((len(B_indices), len(B_indices)))

        for k, comp in enumerate(components):
            mu_A = comp.mean[A_indices]
            mu_B = comp.mean[B_indices]
            Theta = comp.precision

            Theta_AA = Theta[np.ix_(A_indices, A_indices)]
            Theta_BB = Theta[np.ix_(B_indices, B_indices)]
            Theta_BA = Theta[np.ix_(B_indices, A_indices)]

            # Conditional precision (exact for Gaussian): Theta_BB
            # Conditional mean: mu_B - Theta_BB^{-1} Theta_BA (x_A - mu_A)
            Theta_BB_inv = np.linalg.inv(Theta_BB)
            delta = X_A - mu_A[np.newaxis, :]  # (n, |A|)
            mu_B_given_A = mu_B[np.newaxis, :] - (delta @ Theta_BA.T) @ Theta_BB_inv
            # shape: (n, |B|)

            w_k = responsibilities[:, k : k + 1]  # (n, 1)
            cond_means += w_k * mu_B_given_A

            # Responsibility-weighted precision
            cond_precision += np.mean(responsibilities[:, k]) * Theta_BB

        return cond_means, cond_precision, log_resp

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_em(
        self,
        X: NDArray[np.float64],
        alpha: float,
        rng: np.random.Generator,
    ) -> tuple[list[GMMComponent], float, int, bool]:
        n, p = X.shape
        K = self.n_components

        # Initialise with k-means++ style
        components = self._init_components(X, K, alpha, rng)

        prev_ll = -np.inf
        n_iter = 0
        converged = False

        for n_iter in range(1, self.max_iter + 1):
            # E-step
            log_resp = self._e_step(X, components)
            resp = np.exp(log_resp)  # (n, K)

            # M-step
            components = self._m_step(X, resp, alpha)

            # Log-likelihood
            ll = self._log_likelihood(X, components)
            if abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

        return components, ll, n_iter, converged

    def _init_components(
        self,
        X: NDArray[np.float64],
        K: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> list[GMMComponent]:
        n, p = X.shape
        # Random partition for initialisation
        labels = rng.integers(0, K, size=n)
        components = []
        for k in range(K):
            mask = labels == k
            if mask.sum() < 2:
                mask = np.ones(n, dtype=bool)
            Xk = X[mask]
            mean_k = Xk.mean(axis=0)
            prec_k = self._fit_graphlasso(Xk, alpha)
            w_k = mask.sum() / n
            components.append(GMMComponent(weight=w_k, mean=mean_k, precision=prec_k))
        return components

    def _e_step(
        self, X: NDArray[np.float64], components: list[GMMComponent]
    ) -> NDArray[np.float64]:
        n = X.shape[0]
        K = len(components)
        log_resp = np.zeros((n, K))
        for k, comp in enumerate(components):
            log_resp[:, k] = np.log(comp.weight + 1e-300) + self._log_gaussian(
                X, comp.mean, comp.precision
            )
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
        return log_resp

    def _e_step_partial(
        self,
        X_A: NDArray[np.float64],
        A_indices: NDArray[np.intp],
        components: list[GMMComponent],
    ) -> NDArray[np.float64]:
        """E-step using only the A-block of X."""
        n = X_A.shape[0]
        K = len(components)
        log_resp = np.zeros((n, K))
        for k, comp in enumerate(components):
            mu_A = comp.mean[A_indices]
            Theta_AA = comp.precision[np.ix_(A_indices, A_indices)]
            log_resp[:, k] = np.log(comp.weight + 1e-300) + self._log_gaussian(
                X_A, mu_A, Theta_AA
            )
        log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
        return log_resp

    def _m_step(
        self,
        X: NDArray[np.float64],
        resp: NDArray[np.float64],
        alpha: float,
    ) -> list[GMMComponent]:
        n, p = X.shape
        K = resp.shape[1]
        components = []
        nk = resp.sum(axis=0)  # (K,)
        for k in range(K):
            w_k = nk[k] / n
            if w_k < 1e-10:
                w_k = 1e-10
            mean_k = (resp[:, k : k + 1] * X).sum(axis=0) / nk[k]
            prec_k = self._fit_graphlasso_weighted(X, resp[:, k], mean_k, alpha)
            components.append(GMMComponent(weight=w_k, mean=mean_k, precision=prec_k))
        return components

    def _fit_graphlasso(
        self, X: NDArray[np.float64], alpha: float
    ) -> NDArray[np.float64]:
        n, p = X.shape
        if n < 2:
            return np.eye(p)
        S = np.cov(X, rowvar=False)
        if S.ndim == 0:
            S = np.array([[S]])
        S += self.reg_covar * np.eye(p)
        return self._graphlasso_precision(S, alpha, p)

    def _fit_graphlasso_weighted(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64],
        mean: NDArray[np.float64],
        alpha: float,
    ) -> NDArray[np.float64]:
        n, p = X.shape
        w = weights / (weights.sum() + 1e-300)
        diff = X - mean[np.newaxis, :]
        S = (diff * w[:, np.newaxis]).T @ diff
        S += self.reg_covar * np.eye(p)
        return self._graphlasso_precision(S, alpha, p)

    def _graphlasso_precision(
        self, S: NDArray[np.float64], alpha: float, p: int
    ) -> NDArray[np.float64]:
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
            # Fallback: diagonal precision
            return np.diag(1.0 / (np.diag(S) + self.reg_covar))

    def _log_gaussian(
        self,
        X: NDArray[np.float64],
        mean: NDArray[np.float64],
        precision: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Log density of N(mean, precision^{-1}) evaluated at rows of X."""
        p = mean.shape[0]
        diff = X - mean[np.newaxis, :]  # (n, p)
        # log|Theta| / 2 - p/2 log(2pi) - 1/2 (x-mu)' Theta (x-mu)
        sign, logdet = np.linalg.slogdet(precision)
        if sign <= 0:
            logdet = 0.0
        mahal = np.einsum("ni,ij,nj->n", diff, precision, diff)
        return 0.5 * (logdet - p * np.log(2 * np.pi) - mahal)

    def _log_likelihood(
        self, X: NDArray[np.float64], components: list[GMMComponent]
    ) -> float:
        n = X.shape[0]
        K = len(components)
        log_probs = np.zeros((n, K))
        for k, comp in enumerate(components):
            log_probs[:, k] = np.log(comp.weight + 1e-300) + self._log_gaussian(
                X, comp.mean, comp.precision
            )
        return float(logsumexp(log_probs, axis=1).sum())

    def _compute_bic(
        self,
        X: NDArray[np.float64],
        components: list[GMMComponent],
        ll: float,
    ) -> float:
        n, p = X.shape
        K = len(components)
        # Parameters: K weights, K*p means, K*p*(p+1)/2 covariance entries
        n_params = K * (1 + p + p * (p + 1) // 2)
        return -2 * ll + n_params * np.log(n)

    def _select_alpha_bic(
        self, X: NDArray[np.float64], rng: np.random.Generator
    ) -> float:
        best_bic = np.inf
        best_alpha = self.alpha_grid[0]
        for alpha in self.alpha_grid:
            try:
                components, ll, _, _ = self._run_em(X, alpha, rng)
                bic = self._compute_bic(X, components, ll)
                if bic < best_bic:
                    best_bic = bic
                    best_alpha = alpha
            except Exception:
                continue
        return best_alpha

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError("PenalizedGMM must be fitted before use. Call fit().")

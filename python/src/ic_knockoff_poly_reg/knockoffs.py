"""Conditional knockoff generation from a fitted PenalizedGMM.

Implements the Gaussian model-X conditional knockoffs (Phase 3, step 1).

Given:
  - The fitted GMM base distribution P(X)
  - The current active base feature set A_base
  - The observed data X

We compute the conditional distribution P(X_B | X_A) for each sample
(using the GMM precision block structure) and then sample knockoff copies
X̃_B that satisfy the pairwise exchangeability condition:

    (X_A, X_B, X̃_B) ≡_d (X_A, X̃_B, X_B)

This is achieved via the equicorrelated Gaussian knockoff construction
applied to the marginal conditional covariance Sigma_{B|A}.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .gmm_phase import PenalizedGMM
from .rust_gmm import RustPenalizedGMM


class ConditionalKnockoffGenerator:
    """Generate conditional knockoffs for unselected base features.

    Parameters
    ----------
    gmm : PenalizedGMM
        A fitted PenalizedGMM instance (Phase 1 output).
    sdp_method : str
        Method to find the knockoff s-values.  ``"equicorrelated"`` uses
        s_j = min(2 * lambda_min(Sigma), 1), which is fast and numerically
        stable.  ``"sdp"`` requires an SDP solver (not bundled).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        gmm: Union[PenalizedGMM, RustPenalizedGMM],
        sdp_method: str = "equicorrelated",
        random_state: Optional[int] = None,
    ) -> None:
        self.gmm = gmm
        self.sdp_method = sdp_method
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        X: NDArray[np.float64],
        A_indices: NDArray[np.intp],
        B_indices: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Sample conditional knockoffs X̃_B for the unselected features.

        Parameters
        ----------
        X : (n_samples, p) full base feature matrix
        A_indices : 1-D int array – indices of selected (active) base features
        B_indices : 1-D int array – indices of unselected base features

        Returns
        -------
        X_tilde_B : (n_samples, |B|) knockoff matrix for unselected features
        """
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        A_indices = np.asarray(A_indices, dtype=int)
        B_indices = np.asarray(B_indices, dtype=int)

        n = X.shape[0]
        X_A = X[:, A_indices]
        X_B = X[:, B_indices]

        # Get conditional parameters from the GMM
        cond_means, cond_precision, _ = self.gmm.compute_conditional_params(
            X_A, A_indices, B_indices
        )
        # cond_precision is (|B|, |B|) — the conditional precision Theta_{B|A}
        cond_cov = np.linalg.inv(cond_precision)

        # Compute knockoff s-values for the conditional covariance
        s_values = self._compute_s_values(cond_cov)

        # Sample knockoffs: for each sample i, draw X̃_B(i) from
        #   X̃_B | X_B ~ N(mean_tilde_i, V_tilde)
        #   where:
        #     V_tilde = 2*S - S * Theta_{B|A} * S
        #     mean_tilde_i = cond_means[i] + (Sigma_{B|A} - S) * Theta_{B|A}
        #                    * (X_B[i] - cond_means[i])
        S_diag = np.diag(s_values)
        Sigma = cond_cov
        Theta = cond_precision

        V_tilde = 2 * S_diag - S_diag @ Theta @ S_diag
        V_tilde = self._nearest_psd(V_tilde)

        A_mat = (Sigma - S_diag) @ Theta  # (|B|, |B|)
        noise_cov_chol = np.linalg.cholesky(V_tilde)

        X_tilde = np.zeros((n, len(B_indices)))
        noise = rng.standard_normal((n, len(B_indices))) @ noise_cov_chol.T
        for i in range(n):
            diff = X_B[i] - cond_means[i]
            X_tilde[i] = cond_means[i] + A_mat @ diff + noise[i]

        return X_tilde

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_s_values(self, Sigma: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute equicorrelated s-values: s_j = min(2*lambda_min, 1)*diag."""
        eigvals = np.linalg.eigvalsh(Sigma)
        lambda_min = max(float(eigvals.min()), 0.0)
        s = min(2.0 * lambda_min, 1.0)
        # Scale by diagonal entries so that S <= 2*Sigma (element-wise for diag)
        diag_sigma = np.diag(Sigma)
        s_values = s * np.ones(Sigma.shape[0])
        # Ensure s_j <= 2 * sigma_{jj}
        s_values = np.minimum(s_values, 2.0 * diag_sigma - 1e-10)
        s_values = np.maximum(s_values, 1e-10)
        return s_values

    @staticmethod
    def _nearest_psd(M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project M onto the nearest positive semi-definite matrix."""
        M = (M + M.T) / 2
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

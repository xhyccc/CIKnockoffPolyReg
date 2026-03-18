"""Python (NumPy) implementations of the IC-Knock-Poly computational kernels.

These classes delegate directly to the existing pure-Python modules, providing
the :class:`~.base.PolynomialKernel`, :class:`~.base.KnockoffKernel`, and
:class:`~.base.PosiKernel` interfaces with no additional overhead.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..polynomial import ExpandedFeatures, PolynomialDictionary
from ..posi_threshold import SpendingSequence, compute_knockoff_threshold
from .base import KnockoffKernel, PosiKernel, PolynomialKernel


class PythonPolynomialKernel(PolynomialKernel):
    """Polynomial expansion backed by :class:`~ic_knockoff_poly_reg.polynomial.PolynomialDictionary`."""

    def n_expanded(self, n_base: int, degree: int, include_bias: bool) -> int:
        return n_base * 2 * degree + (1 if include_bias else 0)

    def expand(
        self,
        X: NDArray[np.float64],
        degree: int,
        include_bias: bool,
        clip_threshold: float = 1e-8,
        base_names: Optional[list[str]] = None,
    ) -> ExpandedFeatures:
        poly = PolynomialDictionary(
            degree=degree,
            include_bias=include_bias,
            clip_threshold=clip_threshold,
        )
        return poly.expand(X, base_names=base_names)


class PythonKnockoffKernel(KnockoffKernel):
    """Knockoff utilities backed by NumPy linear algebra."""

    def w_statistics(
        self,
        beta_original: NDArray[np.float64],
        beta_knockoff: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        beta_original = np.asarray(beta_original, dtype=np.float64)
        beta_knockoff = np.asarray(beta_knockoff, dtype=np.float64)
        if len(beta_original) != len(beta_knockoff):
            raise ValueError("beta_original and beta_knockoff must have the same length")
        return np.abs(beta_original) - np.abs(beta_knockoff)

    def equicorrelated_s_values(
        self,
        cov: NDArray[np.float64],
        reg: float = 1e-10,
    ) -> NDArray[np.float64]:
        cov = np.asarray(cov, dtype=np.float64)
        p = cov.shape[0]
        eigvals = np.linalg.eigvalsh(cov)
        lambda_min = max(float(eigvals.min()), 0.0)
        base_s = 2.0 * lambda_min
        min_diag = float(np.min(np.diag(cov)))
        s = min(base_s, min_diag - reg)
        s = max(s, reg)
        return np.full(p, s)

    def sample_gaussian_knockoffs(
        self,
        X: NDArray[np.float64],
        mu: NDArray[np.float64],
        Sigma: NDArray[np.float64],
        seed: int = 42,
    ) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        Sigma = np.asarray(Sigma, dtype=np.float64)
        n, p = X.shape

        s_vals = self.equicorrelated_s_values(Sigma)
        S_mat = np.diag(s_vals)

        Sigma_reg = Sigma + 1e-8 * np.eye(p)
        Sigma_inv = np.linalg.inv(Sigma_reg)

        A_mat = (Sigma - S_mat) @ Sigma_inv
        V_tilde = 2 * S_mat - S_mat @ Sigma_inv @ S_mat
        V_tilde = (V_tilde + V_tilde.T) / 2
        eigv = np.linalg.eigvalsh(V_tilde)
        V_tilde += max(0.0, -eigv.min() + 1e-8) * np.eye(p)

        chol = np.linalg.cholesky(V_tilde)
        noise = rng.standard_normal((n, p)) @ chol.T
        X_tilde = mu + (X - mu) @ (np.eye(p) - A_mat).T + noise
        return X_tilde


class PythonPosiKernel(PosiKernel):
    """PoSI budget and threshold backed by :mod:`~ic_knockoff_poly_reg.posi_threshold`."""

    def alpha_spending_budget(
        self,
        t: int,
        Q: float,
        sequence: str,
        gamma: float = 0.5,
    ) -> float:
        seq = SpendingSequence(sequence)
        if seq == SpendingSequence.RIEMANN_ZETA:
            return Q * 6.0 / (math.pi ** 2 * t ** 2)
        else:
            return Q * (1.0 - gamma) * (gamma ** (t - 1))

    def knockoff_threshold(
        self,
        W: NDArray[np.float64],
        q_t: float,
        active_poly: Optional[set[int]] = None,
        offset: int = 1,
    ) -> float:
        return compute_knockoff_threshold(W, q_t, active_poly, offset)

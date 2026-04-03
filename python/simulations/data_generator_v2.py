"""Synthetic dataset generation with correlated features for IC-Knock-Poly simulation.

This is an enhanced version with:
- Feature correlation structure (Option 1)
- Higher noise levels (Option 3)

Provides functions for drawing features from a Gaussian Mixture Model (GMM)
with correlated features and constructing a k-sparse rational polynomial response.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import random_correlation

# Make ic_knockoff_poly_reg importable when running from the simulations dir
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from ic_knockoff_poly_reg.polynomial import PolynomialDictionary


# ---------------------------------------------------------------------------
# Data containers (re-export from original)
# ---------------------------------------------------------------------------

@dataclass
class SimulatedDataset:
    """A single synthesised dataset for one simulation run."""
    
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X_test: Optional[NDArray[np.float64]]
    y_test: Optional[NDArray[np.float64]]
    y_test_noisy: Optional[NDArray[np.float64]]
    X_unlabeled: Optional[NDArray[np.float64]]
    true_base_indices: set
    true_poly_terms: list
    true_coef: NDArray[np.float64]
    noise_std: float
    n_labeled: int
    n_test: int
    n_unlabeled: int
    p: int
    k: int
    degree: int
    gmm_n_components: int
    random_state: Optional[int]


# ---------------------------------------------------------------------------
# Enhanced generators with feature correlation
# ---------------------------------------------------------------------------

def generate_correlated_covariance(
    p: int,
    correlation: float = 0.8,
    eigenvalues: Optional[NDArray] = None,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Generate a p×p covariance matrix with specified average correlation.
    
    Parameters
    ----------
    p : int
        Number of features.
    correlation : float
        Target average correlation between features (0-1).
    eigenvalues : ndarray or None
        If provided, use these eigenvalues for the correlation matrix.
        If None, generate eigenvalues that ensure positive definiteness.
    random_state : int or None
        Seed for reproducibility.
    
    Returns
    -------
    cov : ndarray of shape (p, p)
        Positive definite covariance matrix.
    """
    rng = np.random.default_rng(random_state)
    
    if eigenvalues is None:
        # Generate eigenvalues that sum to p (for correlation matrix)
        # Use a decaying pattern: first few large, rest small
        eigenvalues = np.exp(-np.arange(p) * 0.5)
        eigenvalues = eigenvalues / eigenvalues.sum() * p
    
    # Generate random correlation matrix with given eigenvalues
    # Use a simple approach: create a correlation matrix with constant off-diagonal
    corr = np.full((p, p), correlation)
    np.fill_diagonal(corr, 1.0)
    
    # Ensure positive definiteness by adjusting eigenvalues
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.01)  # Ensure positive
    eigvals = eigvals / eigvals.sum() * p  # Normalize
    
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Convert to covariance matrix by adding feature-specific variances
    feature_stds = rng.uniform(0.5, 1.5, size=p)
    cov = corr * np.outer(feature_stds, feature_stds)
    
    return cov


def generate_gmm_features_correlated(
    n: int,
    p: int,
    n_components: int = 2,
    *,
    correlation: float = 0.8,
    center_scale: float = 2.0,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Draw *n* samples from a random p-dimensional GMM with correlated features.
    
    Each component has:
    * A randomly chosen mean drawn from N(0, center_scale² I).
    * A correlated covariance matrix with average correlation ~correlation.
    * Equal mixing weights 1/K.
    
    The features are shifted so that all values are strictly positive
    (minimum value ≥ 0.5) to avoid singularities in negative-exponent terms.
    
    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    n_components : int
        Number of GMM components K.
    correlation : float
        Target average correlation between features (default 0.8).
    center_scale : float
        Scale of component means.
    random_state : int or None
        Seed for reproducibility.
    
    Returns
    -------
    X : ndarray of shape (n, p)
        Feature matrix with all entries ≥ 0.5.
    """
    rng = np.random.default_rng(random_state)
    
    # Random component means
    means = rng.standard_normal((n_components, p)) * center_scale  # (K, p)
    
    # Generate correlated covariance matrices for each component
    covariances = [
        generate_correlated_covariance(p, correlation, random_state=int(rng.integers(0, 2**31)) + i)
        for i in range(n_components)
    ]
    
    weights = np.ones(n_components) / n_components
    
    # Assign each sample to a component
    assignments = rng.choice(n_components, size=n, p=weights)
    
    X = np.empty((n, p), dtype=np.float64)
    for k in range(n_components):
        mask = assignments == k
        nk = mask.sum()
        if nk > 0:
            X[mask] = rng.multivariate_normal(means[k], covariances[k], size=nk)
    
    # Shift so all values are ≥ 0.5 (avoids negative-exponent singularities).
    col_min = X.min(axis=0)
    shift = np.maximum(0.0, 0.5 - col_min + 1e-9)
    X += shift
    
    return X


def generate_simulation_v2(
    n_labeled: int,
    p: int,
    k: int,
    *,
    degree: int = 2,
    n_components: int = 2,
    noise_std: float = 0.5,
    correlation: float = 0.8,
    n_unlabeled: int = 0,
    n_test: int = 200,
    coef_scale: float = 1.0,
    random_state: Optional[int] = None,
) -> SimulatedDataset:
    """Generate a complete synthesised dataset with correlated features.
    
    Enhanced version with:
    - Feature correlation (correlation parameter)
    - Higher noise levels supported (noise_std up to 5.0)
    
    The data-generating process is::
    
        X           ~ GMM(K=n_components, p=p) with correlated features
        Φ(X)        = polynomial dictionary of X (degree d, rational)
        β*          k-sparse vector over Φ(X) columns
        ε           ~ N(0, noise_std²) i.i.d.
        y           = Φ(X) β* + ε
    
    Parameters
    ----------
    n_labeled : int
        Number of labeled (X, y) observations (training set).
    p : int
        Number of base features.
    k : int
        Number of non-zero *polynomial terms* in β*.
    degree : int
        Maximum absolute polynomial exponent.
    n_components : int
        True number of GMM components for X.
    noise_std : float
        Standard deviation of additive Gaussian noise.
        Recommended: 1.0, 2.0, 5.0 for high-noise experiments.
    correlation : float
        Average correlation between features (0-1).
        Default 0.8 for strongly correlated features.
    n_unlabeled : int
        Number of extra unlabeled observations (semi-supervised).
    n_test : int
        Fixed number of test samples to generate (default 200).
    coef_scale : float
        Scale factor for the non-zero regression coefficients.
    random_state : int or None
        Seed for reproducibility.
    
    Returns
    -------
    SimulatedDataset
    """
    n_terms_total = 2 * degree * p
    if k > n_terms_total:
        raise ValueError(
            f"k ({k}) must not exceed the total number of polynomial terms "
            f"({n_terms_total} = 2·degree·p = 2·{degree}·{p})"
        )
    
    rng = np.random.default_rng(random_state)
    
    # ------------------------------------------------------------------
    # Draw labeled (and optionally unlabeled) features from correlated GMM
    # ------------------------------------------------------------------
    n_total = n_labeled + n_unlabeled
    X_all = generate_gmm_features_correlated(
        n_total, p,
        n_components=n_components,
        correlation=correlation,
        random_state=int(rng.integers(0, 2**31)),
    )
    X_labeled = X_all[:n_labeled]
    X_unlabeled = X_all[n_labeled:] if n_unlabeled > 0 else None
    
    # ------------------------------------------------------------------
    # Build polynomial dictionary and select k-sparse β*
    # ------------------------------------------------------------------
    poly = PolynomialDictionary(degree=degree, include_bias=False)
    poly_result = poly.expand(X_labeled)
    Phi = poly_result.matrix
    names = poly_result.feature_names
    base_indices_list = poly_result.base_feature_indices
    exponents = poly_result.power_exponents
    interaction_list = poly_result.interaction_indices
    interaction_exp_list = poly_result.interaction_exponents
    
    # Randomly choose k distinct polynomial term indices
    true_poly_idx = sorted(
        rng.choice(len(base_indices_list), size=k, replace=False)
    )
    
    # Compute true_base_indices
    true_base_indices = set()
    for i in true_poly_idx:
        base_idx = base_indices_list[i]
        interaction = interaction_list[i] if i < len(interaction_list) else None
        if interaction is not None:
            for idx in interaction:
                true_base_indices.add(idx)
        elif base_idx >= 0:
            true_base_indices.add(base_idx)
    
    # Non-zero coefficients drawn uniformly from [-2, -0.5] ∪ [0.5, 2]
    signs = rng.choice([-1, 1], size=k)
    magnitudes = rng.uniform(0.5, 2.0, size=k) * coef_scale
    true_coef = signs * magnitudes
    
    # Assemble full β*
    beta_star = np.zeros(Phi.shape[1])
    for idx, val in zip(true_poly_idx, true_coef):
        beta_star[idx] = val
    
    # ------------------------------------------------------------------
    # Compute response with potentially high noise
    # ------------------------------------------------------------------
    y = Phi @ beta_star + noise_std * rng.standard_normal(n_labeled)
    
    # ------------------------------------------------------------------
    # Generate independent test set
    # ------------------------------------------------------------------
    X_test = None
    y_test_clean = None
    y_test_noisy = None
    if n_test > 0:
        X_test = generate_gmm_features_correlated(
            n_test, p,
            n_components=n_components,
            correlation=correlation,
            random_state=int(rng.integers(0, 2**31)) + 1,
        )
        
        poly_test = PolynomialDictionary(degree=degree, include_bias=False)
        poly_test_result = poly_test.expand(X_test)
        Phi_test = poly_test_result.matrix
        
        y_test_clean = Phi_test @ beta_star
        y_test_noisy = y_test_clean + noise_std * rng.standard_normal(n_test)
    
    # ------------------------------------------------------------------
    # True poly terms for reference
    # ------------------------------------------------------------------
    true_poly_terms = []
    for i in true_poly_idx:
        base_idx = base_indices_list[i]
        exp = exponents[i]
        interaction = interaction_list[i] if i < len(interaction_list) else None
        interaction_exp = interaction_exp_list[i] if i < len(interaction_exp_list) else None
        if interaction is not None:
            true_poly_terms.append([base_idx, exp, interaction, interaction_exp])
        else:
            true_poly_terms.append([base_idx, exp])
    
    return SimulatedDataset(
        X=X_labeled,
        y=y,
        X_test=X_test,
        y_test=y_test_clean,
        y_test_noisy=y_test_noisy,
        X_unlabeled=X_unlabeled,
        true_base_indices=true_base_indices,
        true_poly_terms=true_poly_terms,
        true_coef=true_coef,
        noise_std=noise_std,
        n_labeled=n_labeled,
        n_test=n_test,
        n_unlabeled=n_unlabeled,
        p=p,
        k=k,
        degree=degree,
        gmm_n_components=n_components,
        random_state=random_state,
    )

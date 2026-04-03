"""Extreme dataset generation for ultimate robustness testing.

Combines:
- Heteroscedastic noise (noise depends on feature values)
- Block-diagonal correlation structure
- Heavy-tailed distributions (Student-t)
- High-dimensional settings

This is designed to break standard methods while testing IC-Knock-Poly's robustness.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# Make ic_knockoff_poly_reg importable
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from ic_knockoff_poly_reg.polynomial import PolynomialDictionary


@dataclass
class SimulatedDataset:
    """A synthesised dataset with extreme challenges."""
    
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
    random_state: Optional[int]
    # New: track data characteristics
    has_heavy_tails: bool = True
    has_heteroscedastic: bool = True
    has_block_correlation: bool = True


def generate_block_diagonal_covariance(
    p: int,
    n_blocks: int = 2,
    intra_block_corr: float = 0.95,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Generate block-diagonal covariance matrix.
    
    Features are divided into blocks where features within a block
    are highly correlated, but blocks are independent.
    
    Parameters
    ----------
    p : int
        Number of features.
    n_blocks : int
        Number of blocks (default 2).
    intra_block_corr : float
        Correlation within each block (default 0.95).
    
    Returns
    -------
    cov : ndarray of shape (p, p)
        Block-diagonal covariance matrix.
    """
    # Ensure p is divisible by n_blocks
    p_per_block = p // n_blocks
    actual_p = p_per_block * n_blocks
    
    if actual_p != p:
        warnings.warn(f"Adjusting p from {p} to {actual_p} for equal block sizes")
        p = actual_p
    
    # Create block-diagonal correlation matrix
    corr = np.eye(p)
    
    for b in range(n_blocks):
        start = b * p_per_block
        end = start + p_per_block
        # Fill intra-block correlations
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    corr[i, j] = intra_block_corr
    
    # Convert to covariance with varying scales
    rng = np.random.default_rng(random_state)
    feature_stds = rng.uniform(0.5, 2.0, size=p)  # Heavy variation in scales
    cov = corr * np.outer(feature_stds, feature_stds)
    
    return cov


def generate_heavy_tailed_features(
    n: int,
    p: int,
    cov: NDArray[np.float64],
    df: int = 3,  # Student-t degrees of freedom
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Generate heavy-tailed features using multivariate Student-t.
    
    Uses Student-t distribution with df degrees of freedom.
    Lower df = heavier tails (df=3 has infinite kurtosis).
    
    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    cov : ndarray of shape (p, p)
        Covariance matrix.
    df : int
        Degrees of freedom for Student-t (default 3).
    
    Returns
    -------
    X : ndarray of shape (n, p)
        Heavy-tailed feature matrix.
    """
    rng = np.random.default_rng(random_state)
    
    # Generate from multivariate Student-t
    # X = Z / sqrt(V/df) where Z ~ N(0, Σ), V ~ χ²(df)
    Z = rng.multivariate_normal(np.zeros(p), cov, size=n)
    V = rng.chisquare(df, size=n)
    
    X = Z / np.sqrt(V[:, None] / df)
    
    # Shift to ensure positivity for polynomial features
    # Student-t can produce extreme negative values, so we clip
    col_min = X.min(axis=0)
    shift = np.maximum(0.0, 0.1 - col_min + 1e-9)
    X += shift
    
    return X


def generate_heteroscedastic_noise(
    n: int,
    X: NDArray[np.float64],
    base_noise: float,
    hetero_factor: float = 0.5,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Generate heteroscedastic noise.
    
    Noise variance depends on feature values:
    σ(x) = base_noise * (1 + hetero_factor * |x_1|)
    
    Parameters
    ----------
    n : int
        Number of samples.
    X : ndarray of shape (n, p)
        Feature matrix.
    base_noise : float
        Base noise level.
    hetero_factor : float
        How much noise scales with x (default 0.5).
    
    Returns
    -------
    noise : ndarray of shape (n,)
        Heteroscedastic noise.
    """
    rng = np.random.default_rng(random_state)
    
    # Noise variance depends on first feature
    noise_std = base_noise * (1 + hetero_factor * np.abs(X[:, 0]))
    noise = rng.normal(0, noise_std)
    
    return noise


def add_missing_values(
    X: NDArray[np.float64],
    missing_rate: float = 0.1,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Add missing values (MCAR - Missing Completely At Random).
    
    Parameters
    ----------
    X : ndarray
        Feature matrix.
    missing_rate : float
        Proportion of values to set to NaN (default 0.1).
    
    Returns
    -------
    X_missing : ndarray
        Matrix with missing values (set to 0 for simplicity).
    """
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()
    
    # Randomly mask values
    mask = rng.random(X.shape) < missing_rate
    X_missing[mask] = np.nan
    
    # Fill NaN with column means (simple imputation)
    col_means = np.nanmean(X_missing, axis=0)
    inds = np.where(np.isnan(X_missing))
    X_missing[inds] = np.take(col_means, inds[1])
    
    return X_missing


def add_outliers(
    y: NDArray[np.float64],
    outlier_rate: float = 0.05,
    outlier_scale: float = 10.0,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Add outliers to response variable.
    
    Parameters
    ----------
    y : ndarray
        Response vector.
    outlier_rate : float
        Proportion of samples to corrupt (default 0.05).
    outlier_scale : float
        Scale of outliers relative to y std (default 10.0).
    
    Returns
    -------
    y_outlier : ndarray
        Response with outliers.
    """
    rng = np.random.default_rng(random_state)
    y_outlier = y.copy()
    
    n_outliers = int(len(y) * outlier_rate)
    outlier_indices = rng.choice(len(y), size=n_outliers, replace=False)
    
    # Add extreme values
    y_std = np.std(y)
    y_outlier[outlier_indices] += rng.choice([-1, 1], size=n_outliers) * outlier_scale * y_std
    
    return y_outlier


def generate_extreme_simulation(
    n_labeled: int,
    p: int,
    k: int,
    *,
    degree: int = 2,
    noise_std: float = 5.0,
    n_blocks: int = 2,
    intra_block_corr: float = 0.95,
    hetero_factor: float = 0.5,
    heavy_tail_df: int = 3,
    add_missing: bool = False,
    missing_rate: float = 0.1,
    add_outliers: bool = False,
    outlier_rate: float = 0.05,
    n_unlabeled: int = 0,
    n_test: int = 200,
    coef_scale: float = 1.0,
    random_state: Optional[int] = None,
) -> SimulatedDataset:
    """Generate extreme simulation with all challenges combined.
    
    Challenges:
    1. Block-diagonal correlation (high intra-block correlation)
    2. Heavy-tailed features (Student-t distribution)
    3. Heteroscedastic noise (variance depends on features)
    4. Optional: Missing values
    5. Optional: Outliers
    
    Parameters
    ----------
    n_labeled : int
        Training set size.
    p : int
        Number of features.
    k : int
        Number of non-zero polynomial terms.
    degree : int
        Max polynomial degree.
    noise_std : float
        Base noise level (try 3.0, 5.0, 10.0).
    n_blocks : int
        Number of correlation blocks.
    intra_block_corr : float
        Correlation within blocks.
    hetero_factor : float
        Heteroscedasticity strength.
    heavy_tail_df : int
        Student-t df (lower = heavier tails).
    add_missing : bool
        Whether to add missing values.
    missing_rate : float
        Proportion of missing values.
    add_outliers : bool
        Whether to add outliers.
    outlier_rate : float
        Proportion of outliers.
    n_test : int
        Test set size.
    
    Returns
    -------
    SimulatedDataset
    """
    rng = np.random.default_rng(random_state)
    
    # Adjust p to be divisible by n_blocks
    p_per_block = p // n_blocks
    p = p_per_block * n_blocks
    
    # Generate covariance matrix (block-diagonal)
    cov = generate_block_diagonal_covariance(
        p, n_blocks=n_blocks, intra_block_corr=intra_block_corr,
        random_state=int(rng.integers(0, 2**31))
    )
    
    # Generate heavy-tailed features
    n_total = n_labeled + n_unlabeled
    X_all = generate_heavy_tailed_features(
        n_total, p, cov, df=heavy_tail_df,
        random_state=int(rng.integers(0, 2**31))
    )
    
    # Add missing values if requested
    if add_missing:
        X_all = add_missing_values(
            X_all, missing_rate=missing_rate,
            random_state=int(rng.integers(0, 2**31))
        )
    
    X_labeled = X_all[:n_labeled]
    X_unlabeled = X_all[n_labeled:] if n_unlabeled > 0 else None
    
    # Build polynomial dictionary and select k terms
    poly = PolynomialDictionary(degree=degree, include_bias=False)
    poly_result = poly.expand(X_labeled)
    Phi = poly_result.matrix
    
    n_terms_total = Phi.shape[1]
    if k > n_terms_total:
        raise ValueError(f"k ({k}) exceeds total terms ({n_terms_total})")
    
    # Select k terms
    true_poly_idx = sorted(rng.choice(n_terms_total, size=k, replace=False))
    
    # Get base indices
    base_indices_list = poly_result.base_feature_indices
    true_base_indices = set()
    for idx in true_poly_idx:
        base_idx = base_indices_list[idx]
        if base_idx >= 0:
            true_base_indices.add(base_idx)
    
    # Generate coefficients
    signs = rng.choice([-1, 1], size=k)
    magnitudes = rng.uniform(0.5, 2.0, size=k) * coef_scale
    true_coef = signs * magnitudes
    
    # Assemble beta_star
    beta_star = np.zeros(Phi.shape[1])
    for idx, val in zip(true_poly_idx, true_coef):
        beta_star[idx] = val
    
    # Generate response with heteroscedastic noise
    y_clean = Phi @ beta_star
    noise = generate_heteroscedastic_noise(
        n_labeled, X_labeled, noise_std, hetero_factor,
        random_state=int(rng.integers(0, 2**31))
    )
    y = y_clean + noise
    
    # Add outliers if requested
    if add_outliers:
        y = add_outliers(
            y, outlier_rate=outlier_rate,
            random_state=int(rng.integers(0, 2**31))
        )
    
    # Generate test set
    X_test = None
    y_test_clean = None
    y_test_noisy = None
    if n_test > 0:
        X_test = generate_heavy_tailed_features(
            n_test, p, cov, df=heavy_tail_df,
            random_state=int(rng.integers(0, 2**31)) + 1
        )
        
        if add_missing:
            X_test = add_missing_values(
                X_test, missing_rate=missing_rate,
                random_state=int(rng.integers(0, 2**31)) + 2
            )
        
        poly_test = PolynomialDictionary(degree=degree, include_bias=False)
        poly_test_result = poly_test.expand(X_test)
        Phi_test = poly_test_result.matrix
        
        y_test_clean = Phi_test @ beta_star
        test_noise = generate_heteroscedastic_noise(
            n_test, X_test, noise_std, hetero_factor,
            random_state=int(rng.integers(0, 2**31)) + 3
        )
        y_test_noisy = y_test_clean + test_noise
        
        if add_outliers:
            y_test_noisy = add_outliers(
                y_test_noisy, outlier_rate=outlier_rate,
                random_state=int(rng.integers(0, 2**31)) + 4
            )
    
    # True poly terms
    true_poly_terms = []
    exponents = poly_result.power_exponents
    for idx in true_poly_idx:
        base_idx = base_indices_list[idx]
        exp = exponents[idx]
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
        random_state=random_state,
        has_heavy_tails=True,
        has_heteroscedastic=True,
        has_block_correlation=True,
    )

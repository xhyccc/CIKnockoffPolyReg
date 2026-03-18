"""Synthetic dataset generation for IC-Knock-Poly simulation experiments.

Provides functions for drawing features from a Gaussian Mixture Model (GMM)
and constructing a k-sparse rational polynomial response::

    y = Φ(X)^T β* + ε,   β* is k-sparse,   X ~ GMM(π, μ, Σ)

The polynomial dictionary Φ(·) is the same rational polynomial dictionary
used by the main IC-Knock-Poly algorithm: each base feature x_j contributes
terms x_j^1, x_j^2, …, x_j^d, x_j^{-1}, …, x_j^{-d}.

The ground-truth signal is constructed so that the *base feature indices*
that appear in β* are exactly those in ``true_base_indices``.  This allows
the evaluation code to report FDR and TPR at the base-feature level.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Make ic_knockoff_poly_reg importable when running from the simulations dir
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from ic_knockoff_poly_reg.polynomial import PolynomialDictionary


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SimulatedDataset:
    """A single synthesised dataset for one simulation run.

    Attributes
    ----------
    X : ndarray of shape (n_labeled, p)
        Labeled feature matrix drawn from the GMM.
    y : ndarray of shape (n_labeled,)
        Response vector y = Φ(X) β* + ε.
    X_unlabeled : ndarray of shape (n_unlabeled, p) or None
        Extra unlabeled observations drawn from the same GMM.
        Set to ``None`` in the supervised-only setting.
    true_base_indices : set of int
        Base feature column indices that have a non-zero coefficient in β*.
    true_poly_terms : list of [int, int]
        ``[base_idx, exponent]`` pairs for the non-zero terms in β*.
    true_coef : ndarray
        Ground-truth non-zero polynomial coefficients β*.
    noise_std : float
        Standard deviation of the additive Gaussian noise ε.
    n_labeled : int
        Number of labeled observations.
    n_unlabeled : int
        Number of unlabeled observations (0 in the supervised setting).
    p : int
        Number of base features.
    k : int
        Sparsity (number of non-zero base features in β*).
    degree : int
        Maximum polynomial degree used when constructing the response.
    gmm_n_components : int
        True number of GMM components.
    random_state : Optional[int]
        Seed used to generate this dataset.
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X_unlabeled: Optional[NDArray[np.float64]]
    true_base_indices: set
    true_poly_terms: list
    true_coef: NDArray[np.float64]
    noise_std: float
    n_labeled: int
    n_unlabeled: int
    p: int
    k: int
    degree: int
    gmm_n_components: int
    random_state: Optional[int]


# ---------------------------------------------------------------------------
# Low-level generators
# ---------------------------------------------------------------------------

def generate_gmm_features(
    n: int,
    p: int,
    n_components: int = 2,
    *,
    component_std: float = 1.0,
    center_scale: float = 2.0,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """Draw *n* samples from a random p-dimensional GMM with *n_components* components.

    Each component has:
    * A randomly chosen mean drawn from N(0, center_scale² I).
    * A random diagonal covariance with entries ~Uniform(0.5, 1.5) ×
      component_std².
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
    component_std : float
        Scale of per-component standard deviations.
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

    # Random component means and diagonal covariances
    means = rng.standard_normal((n_components, p)) * center_scale  # (K, p)
    stds = rng.uniform(0.5, 1.5, size=(n_components, p)) * component_std  # (K, p)
    weights = np.ones(n_components) / n_components

    # Assign each sample to a component
    assignments = rng.choice(n_components, size=n, p=weights)

    X = np.empty((n, p), dtype=np.float64)
    for k in range(n_components):
        mask = assignments == k
        nk = mask.sum()
        if nk > 0:
            X[mask] = rng.normal(means[k], stds[k], size=(nk, p))

    # Shift so all values are ≥ 0.5 (avoids negative-exponent singularities).
    # A small epsilon is added to guard against floating-point rounding.
    col_min = X.min(axis=0)
    shift = np.maximum(0.0, 0.5 - col_min + 1e-9)
    X += shift

    return X


def generate_simulation(
    n_labeled: int,
    p: int,
    k: int,
    *,
    degree: int = 2,
    n_components: int = 2,
    noise_std: float = 0.5,
    n_unlabeled: int = 0,
    coef_scale: float = 1.0,
    random_state: Optional[int] = None,
) -> SimulatedDataset:
    """Generate a complete synthesised dataset for one simulation trial.

    The data-generating process is::

        X           ~ GMM(K=n_components, p=p)
        Φ(X)        = polynomial dictionary of X (degree d, rational)
        β*          k-sparse vector over Φ(X) columns, non-zero values ~ U[-2,-0.5]∪[0.5,2]
        ε           ~ N(0, noise_std²) i.i.d.
        y           = Φ(X) β* + ε

    The k non-zero entries of β* are chosen from the *first* k polynomial
    terms of the dictionary (one term per base feature) so that:

    * Each non-zero base feature contributes at least one term.
    * ``true_base_indices`` = {0, 1, …, k-1} (first k features).

    Parameters
    ----------
    n_labeled : int
        Number of labeled (X, y) observations.
    p : int
        Number of base features.
    k : int
        Number of non-zero *base features* in β*.  Must satisfy k ≤ p.
    degree : int
        Maximum absolute polynomial exponent.
    n_components : int
        True number of GMM components for X.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    n_unlabeled : int
        Number of extra unlabeled observations (semi-supervised).  Pass 0
        for the pure supervised setting.
    coef_scale : float
        Scale factor for the non-zero regression coefficients.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    SimulatedDataset
    """
    if k > p:
        raise ValueError(f"k ({k}) must not exceed p ({p})")

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Draw labeled (and optionally unlabeled) features from the GMM
    # ------------------------------------------------------------------
    n_total = n_labeled + n_unlabeled
    X_all = generate_gmm_features(
        n_total, p,
        n_components=n_components,
        random_state=int(rng.integers(0, 2**31)),
    )
    X_labeled = X_all[:n_labeled]
    X_unlabeled = X_all[n_labeled:] if n_unlabeled > 0 else None

    # ------------------------------------------------------------------
    # Build polynomial dictionary and select k-sparse β*
    # ------------------------------------------------------------------
    poly = PolynomialDictionary(degree=degree, include_bias=False)
    poly_result = poly.expand(X_labeled)
    Phi = poly_result.matrix                          # (n_labeled, n_terms)
    names = poly_result.feature_names                 # list of str
    base_indices_list = poly_result.base_feature_indices  # list of int (base feature per col)
    exponents = poly_result.power_exponents           # list of int (exponent per col)

    # Choose one polynomial term per each of the first k base features
    # (specifically the degree-1 term x_j^1) as the support of β*
    true_poly_idx: list[int] = []
    true_base_indices: set[int] = set(range(k))
    for j in range(k):
        # Find the index of x_j^1 in the dictionary
        for col_idx, (base_idx, exp) in enumerate(zip(base_indices_list, exponents)):
            if base_idx == j and exp == 1:
                true_poly_idx.append(col_idx)
                break

    # Non-zero coefficients drawn uniformly from [-2, -0.5] ∪ [0.5, 2]
    signs = rng.choice([-1, 1], size=k)
    magnitudes = rng.uniform(0.5, 2.0, size=k) * coef_scale
    true_coef = signs * magnitudes  # (k,)

    # Assemble full β* (n_terms-dimensional, mostly zeros)
    beta_star = np.zeros(Phi.shape[1])
    for idx, val in zip(true_poly_idx, true_coef):
        beta_star[idx] = val

    # ------------------------------------------------------------------
    # Compute response
    # ------------------------------------------------------------------
    y = Phi @ beta_star + noise_std * rng.standard_normal(n_labeled)

    # ------------------------------------------------------------------
    # True poly terms for reference
    # ------------------------------------------------------------------
    true_poly_terms = [[base_indices_list[i], exponents[i]] for i in true_poly_idx]

    return SimulatedDataset(
        X=X_labeled,
        y=y,
        X_unlabeled=X_unlabeled,
        true_base_indices=true_base_indices,
        true_poly_terms=true_poly_terms,
        true_coef=true_coef,
        noise_std=noise_std,
        n_labeled=n_labeled,
        n_unlabeled=n_unlabeled,
        p=p,
        k=k,
        degree=degree,
        gmm_n_components=n_components,
        random_state=random_state,
    )

"""Baseline sparse polynomial regression methods for comparison with IC-Knock-Poly.

Each baseline accepts the same ``(X, y)`` interface and returns a
``ResultBundle`` from ``ic_knockoff_poly_reg.evaluation``.

Available baselines
-------------------
``PolyLasso``
    Polynomial dictionary expansion + cross-validated Lasso.  No FDR control.
``PolyOMP``
    Polynomial dictionary expansion + Orthogonal Matching Pursuit (CV).
``PolyCLIME``
    CLIME-based precision matrix estimation + one-shot Gaussian knockoff filter
    on the polynomial dictionary.  Provides FDR control like IC-Knock-Poly but
    without the iterative GMM or PoSI α-spending.
``PolyKnockoff``
    Standard (non-iterative) Gaussian knockoff filter with sample covariance +
    Lasso on the polynomial dictionary.
``SparsePolySTLSQ``
    Sequential Thresholded Least Squares (SINDy/STLSQ style): iteratively fits
    OLS and removes features whose coefficients fall below a threshold.

Usage
-----
See ``run_comparison.py`` for a complete comparison workflow.
"""

from .poly_lasso import PolyLasso
from .poly_omp import PolyOMP
from .poly_clime import PolyCLIME
from .poly_knockoff import PolyKnockoff
from .sparse_poly_stlsq import SparsePolySTLSQ
from .data_loader import DataLoader, DataBundle

__all__ = [
    "PolyLasso",
    "PolyOMP",
    "PolyCLIME",
    "PolyKnockoff",
    "SparsePolySTLSQ",
    "DataLoader",
    "DataBundle",
]

"""Evaluation metrics for the IC-Knock-Poly algorithm.

Computes:
  - Empirical FDR: false positives / max(1, total selections)
  - True Positive Rate (Power): true positives / max(1, true positives + false negatives)
  - Peak memory usage via tracemalloc
  - ``ResultBundle``: unified research output format (JSON/CSV serialisable)

Unified output format (``ResultBundle``)
-----------------------------------------
Every method — IC-Knock-Poly as well as all baselines — returns a
``ResultBundle`` when asked.  The bundle is a plain Python dataclass that can
be serialised to JSON (``to_dict()`` / ``to_json()``) or written as a row in a
comparison CSV (``to_csv_row()``).

JSON schema summary::

    {
      "method": "ic_knock_poly",
      "dataset": "experiment.csv",
      "timestamp": "2024-01-01T00:00:00Z",
      "selected_names": ["x0", "x1^(-1)"],
      "selected_base_indices": [0, 1],
      "selected_terms": [[0, 1], [1, -1]],
      "coef": [1.02, 0.98],
      "intercept": 0.05,
      "n_selected": 2,
      "fit": {
        "r_squared": 0.98,
        "adj_r_squared": 0.979,
        "residual_ss": 0.12,
        "total_ss": 6.0,
        "bic": -450.2,
        "aic": -455.1
      },
      "discovery": {
        "fdr": 0.0,
        "tpr": 1.0,
        "n_true_positives": 2,
        "n_false_positives": 0,
        "n_false_negatives": 0
      },
      "compute": {
        "elapsed_seconds": 2.3,
        "peak_memory_mb": 45.2
      },
      "params": {
        "degree": 2,
        "Q": 0.10
      }
    }
"""

from __future__ import annotations

import json
import math
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional, Set

import numpy as np


@dataclass
class DiscoveryMetrics:
    """Container for FDR / TPR metrics."""

    fdr: float
    tpr: float
    n_selected: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    peak_memory_mb: Optional[float] = None


@dataclass
class ResultBundle:
    """Unified research output for any sparse polynomial regression method.

    All fields with ``Optional`` type may be ``None`` when the corresponding
    information is not available (e.g. ``fdr``/``tpr`` when ground truth is
    unknown).

    Attributes
    ----------
    method : str
        Short identifier for the method, e.g. ``"ic_knock_poly"``,
        ``"poly_lasso"``, ``"poly_omp"``.
    dataset : str
        Filename or description of the dataset used.
    timestamp : str
        ISO 8601 UTC timestamp of when the result was produced.
    selected_names : list of str
        Human-readable names of selected polynomial features,
        e.g. ``["x0", "x1^(-1)"]``.
    selected_base_indices : list of int
        Indices of base features that appear in any selected term.
    selected_terms : list of [int, int]
        ``[base_feature_index, exponent]`` pairs for each selected term.
        ``base_feature_index == -1`` denotes the bias/intercept term.
    coef : list of float
        Regression coefficients aligned with ``selected_terms``.
    intercept : float
        Fitted intercept.
    n_selected : int
        Total number of selected polynomial terms.
    r_squared : float
        Coefficient of determination on the training data.
    adj_r_squared : float
        Adjusted R², penalised for model complexity.
    residual_ss : float
        Residual sum of squares (lower is better).
    total_ss : float
        Total sum of squares of the response.
    bic : float
        Bayesian Information Criterion (lower is better).
    aic : float
        Akaike Information Criterion (lower is better).
    elapsed_seconds : float
        Wall-clock time for fitting (seconds).
    peak_memory_mb : float
        Peak resident memory during fitting (MB).
    fdr : float or None
        Empirical false discovery rate (requires ground truth).
    tpr : float or None
        True positive rate / power (requires ground truth).
    n_true_positives : int or None
        Number of true positives (requires ground truth).
    n_false_positives : int or None
        Number of false positives (requires ground truth).
    n_false_negatives : int or None
        Number of false negatives (requires ground truth).
    params : dict
        Method-specific hyper-parameters (e.g. ``degree``, ``Q``, ``alpha``).
    extra : dict
        Any additional method-specific diagnostics.
    """

    # Identity
    method: str
    dataset: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Selections
    selected_names: list = field(default_factory=list)
    selected_base_indices: list = field(default_factory=list)
    selected_terms: list = field(default_factory=list)
    coef: list = field(default_factory=list)
    intercept: float = 0.0
    n_selected: int = 0

    # Goodness-of-fit statistics
    r_squared: float = float("nan")
    adj_r_squared: float = float("nan")
    residual_ss: float = float("nan")
    total_ss: float = float("nan")
    bic: float = float("nan")
    aic: float = float("nan")

    # Compute cost
    elapsed_seconds: float = float("nan")
    peak_memory_mb: float = float("nan")

    # Discovery metrics (None when ground truth unknown)
    fdr: Optional[float] = None
    tpr: Optional[float] = None
    n_true_positives: Optional[int] = None
    n_false_positives: Optional[int] = None
    n_false_negatives: Optional[int] = None

    # Test set performance (None when test data not available)
    test_r_squared: Optional[float] = None
    test_rmse: Optional[float] = None
    test_mae: Optional[float] = None
    n_test: Optional[int] = None

    # Method-specific parameters and extras
    params: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict (nested structure).

        Returns
        -------
        dict
            Nested dictionary matching the ResultBundle JSON schema.
        """
        def _clean(v):
            """Convert numpy scalars and NaN/inf to JSON-safe values."""
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                v = float(v)
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
            return v

        return {
            "method": self.method,
            "dataset": self.dataset,
            "timestamp": self.timestamp,
            "selected_names": self.selected_names,
            "selected_base_indices": self.selected_base_indices,
            "selected_terms": self.selected_terms,
            "coef": [_clean(c) for c in self.coef],
            "intercept": _clean(self.intercept),
            "n_selected": self.n_selected,
            "fit": {
                "r_squared": _clean(self.r_squared),
                "adj_r_squared": _clean(self.adj_r_squared),
                "residual_ss": _clean(self.residual_ss),
                "total_ss": _clean(self.total_ss),
                "bic": _clean(self.bic),
                "aic": _clean(self.aic),
            },
            "discovery": {
                "fdr": _clean(self.fdr) if self.fdr is not None else None,
                "tpr": _clean(self.tpr) if self.tpr is not None else None,
                "n_true_positives": self.n_true_positives,
                "n_false_positives": self.n_false_positives,
                "n_false_negatives": self.n_false_negatives,
            },
            "compute": {
                "elapsed_seconds": _clean(self.elapsed_seconds),
                "peak_memory_mb": _clean(self.peak_memory_mb),
            },
            "params": self.params,
            "extra": self.extra,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string.

        Parameters
        ----------
        indent : int
            Pretty-print indentation level.  Default 2.

        Returns
        -------
        str
            JSON representation of this result bundle.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv_row(self) -> dict:
        """Return a flat dict suitable for one row in a comparison CSV.

        The returned dict has string keys and scalar values only, so it can
        be passed directly to ``csv.DictWriter`` or ``pandas.DataFrame``.

        Returns
        -------
        dict
            Flat key-value mapping of the most important summary statistics.
        """
        def _safe(v):
            if v is None:
                return ""
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return ""
            return v

        return {
            "method": self.method,
            "dataset": self.dataset,
            "timestamp": self.timestamp,
            "n_selected": self.n_selected,
            "r_squared": _safe(self.r_squared),
            "adj_r_squared": _safe(self.adj_r_squared),
            "residual_ss": _safe(self.residual_ss),
            "bic": _safe(self.bic),
            "aic": _safe(self.aic),
            "fdr": _safe(self.fdr),
            "tpr": _safe(self.tpr),
            "n_true_positives": _safe(self.n_true_positives),
            "n_false_positives": _safe(self.n_false_positives),
            "n_false_negatives": _safe(self.n_false_negatives),
            "elapsed_seconds": _safe(self.elapsed_seconds),
            "peak_memory_mb": _safe(self.peak_memory_mb),
            "selected_names": "|".join(self.selected_names),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResultBundle":
        """Reconstruct a ``ResultBundle`` from a dict (as returned by ``to_dict``).

        Parameters
        ----------
        d : dict
            Dict as produced by ``to_dict()`` or parsed from a JSON file.

        Returns
        -------
        ResultBundle
        """
        fit = d.get("fit", {})
        disc = d.get("discovery", {})
        comp = d.get("compute", {})
        return cls(
            method=d.get("method", ""),
            dataset=d.get("dataset", ""),
            timestamp=d.get("timestamp", ""),
            selected_names=d.get("selected_names", []),
            selected_base_indices=d.get("selected_base_indices", []),
            selected_terms=d.get("selected_terms", []),
            coef=d.get("coef", []),
            intercept=d.get("intercept", 0.0) or 0.0,
            n_selected=d.get("n_selected", 0),
            r_squared=fit.get("r_squared") or float("nan"),
            adj_r_squared=fit.get("adj_r_squared") or float("nan"),
            residual_ss=fit.get("residual_ss") or float("nan"),
            total_ss=fit.get("total_ss") or float("nan"),
            bic=fit.get("bic") or float("nan"),
            aic=fit.get("aic") or float("nan"),
            elapsed_seconds=comp.get("elapsed_seconds") or float("nan"),
            peak_memory_mb=comp.get("peak_memory_mb") or float("nan"),
            fdr=disc.get("fdr"),
            tpr=disc.get("tpr"),
            n_true_positives=disc.get("n_true_positives"),
            n_false_positives=disc.get("n_false_positives"),
            n_false_negatives=disc.get("n_false_negatives"),
            params=d.get("params", {}),
            extra=d.get("extra", {}),
        )


def _compute_fit_stats(
    y: np.ndarray,
    y_pred: np.ndarray,
    n_params: int,
) -> tuple[float, float, float, float, float, float]:
    """Compute goodness-of-fit statistics.

    Parameters
    ----------
    y : 1-D array
        Observed response values.
    y_pred : 1-D array
        Predicted response values.
    n_params : int
        Number of estimated parameters (including intercept).

    Returns
    -------
    r_squared, adj_r_squared, residual_ss, total_ss, bic, aic
    """
    n = len(y)
    residuals = y - y_pred
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    denom = n - n_params - 1
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / denom if denom > 0 else float("nan")
    # Log-likelihood under Gaussian noise: -n/2 * log(2*pi*sigma^2) - ss_res/(2*sigma^2)
    sigma2 = ss_res / n if n > 0 else 1.0
    if sigma2 <= 0:
        sigma2 = 1e-300
    log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    bic = -2.0 * log_lik + n_params * math.log(n)
    aic = -2.0 * log_lik + 2.0 * n_params
    return r2, adj_r2, ss_res, ss_tot, bic, aic



def compute_fdr(
    selected: Set[int],
    true_features: Set[int],
) -> float:
    """Empirical False Discovery Rate.

    Parameters
    ----------
    selected : set of int
        Indices of features selected by the algorithm.
    true_features : set of int
        Indices of the ground-truth non-zero features.

    Returns
    -------
    fdr : float in [0, 1]
    """
    if len(selected) == 0:
        return 0.0
    false_positives = len(selected - true_features)
    return false_positives / len(selected)


def compute_tpr(
    selected: Set[int],
    true_features: Set[int],
) -> float:
    """True Positive Rate (Power / Recall).

    Parameters
    ----------
    selected : set of int
        Indices of features selected by the algorithm.
    true_features : set of int
        Indices of the ground-truth non-zero features.

    Returns
    -------
    tpr : float in [0, 1]
    """
    if len(true_features) == 0:
        return 1.0
    true_positives = len(selected & true_features)
    return true_positives / len(true_features)


def compute_metrics(
    selected: Set[int],
    true_features: Set[int],
    peak_memory_mb: Optional[float] = None,
) -> DiscoveryMetrics:
    """Compute all discovery metrics at once.

    Parameters
    ----------
    selected : set of int
        Feature indices selected by the algorithm.
    true_features : set of int
        True non-zero feature indices.
    peak_memory_mb : float or None
        Peak memory usage from ``memory_tracker`` context manager.

    Returns
    -------
    DiscoveryMetrics dataclass
    """
    tp = len(selected & true_features)
    fp = len(selected - true_features)
    fn = len(true_features - selected)
    n_sel = len(selected)
    fdr = fp / max(1, n_sel)
    tpr = tp / max(1, len(true_features))
    return DiscoveryMetrics(
        fdr=fdr,
        tpr=tpr,
        n_selected=n_sel,
        n_true_positives=tp,
        n_false_positives=fp,
        n_false_negatives=fn,
        peak_memory_mb=peak_memory_mb,
    )


# =============================================================================
# Polynomial Term Evaluation (Correct Evaluation)
# =============================================================================

def compute_polynomial_term_metrics(
    selected_terms: list,
    true_poly_terms: list,
    peak_memory_mb: Optional[float] = None,
) -> DiscoveryMetrics:
    """Compute discovery metrics for polynomial term selection.
    
    This is the CORRECT evaluation method that compares exact polynomial terms
    [base_idx, exponent] or [base_idx, exponent, interaction_indices] 
    rather than just base feature indices.
    
    Parameters
    ----------
    selected_terms : list of [int, int] or [int, int, list]
        Selected polynomial terms as [base_feature_index, exponent] pairs
        or [base_feature_index, exponent, interaction_indices] for interaction terms.
    true_poly_terms : list of [int, int] or [int, int, list]
        Ground truth polynomial terms as [base_feature_index, exponent] pairs
        or [base_feature_index, exponent, interaction_indices] for interaction terms.
    peak_memory_mb : float or None
        Peak memory usage.
        
    Returns
    -------
    DiscoveryMetrics dataclass with FDR, TPR, F1 computed on exact term matches.
    """
    # Convert to comparable format for set operations
    # For interaction terms, we need to handle the list in the tuple
    def term_to_tuple(term):
        """Convert term to a hashable tuple."""
        if len(term) == 2:
            return tuple(term)
        elif len(term) >= 3:
            # Handle interaction term: convert interaction_indices list to tuple
            base_idx, exp = term[0], term[1]
            interaction = tuple(term[2]) if term[2] is not None else None
            return (base_idx, exp, interaction)
        return tuple(term)
    
    selected_set = set(term_to_tuple(t) for t in selected_terms)
    true_set = set(term_to_tuple(t) for t in true_poly_terms)
    
    # Compute confusion matrix
    tp_set = selected_set & true_set
    fp_set = selected_set - true_set
    fn_set = true_set - selected_set
    
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    n_sel = len(selected_set)
    
    # Compute metrics
    fdr = fp / max(1, n_sel)
    tpr = tp / max(1, len(true_set))
    precision = tp / max(1, n_sel)
    
    return DiscoveryMetrics(
        fdr=fdr,
        tpr=tpr,
        n_selected=n_sel,
        n_true_positives=tp,
        n_false_positives=fp,
        n_false_negatives=fn,
        peak_memory_mb=peak_memory_mb,
    )


def format_polynomial_terms(terms: list) -> list[str]:
    """Format polynomial terms as human-readable strings.
    
    Parameters
    ----------
    terms : list of [int, int]
        Terms as [base_idx, exponent] pairs.
        
    Returns
    -------
    list of str
        Formatted strings like "x_0^2", "x_1^(-1)".
    """
    formatted = []
    for base_idx, exp in terms:
        if exp > 0:
            formatted.append(f"x_{base_idx}^{exp}")
        else:
            formatted.append(f"x_{base_idx}^({exp})")
    return formatted


@contextmanager
def memory_tracker() -> Iterator[dict]:
    """Context manager that records peak memory usage (in MB).

    Usage::

        with memory_tracker() as mem:
            result = algorithm.fit(X, y)
        print(mem["peak_mb"])
    """
    info: dict = {"peak_mb": None}
    tracemalloc.start()
    try:
        yield info
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        info["peak_mb"] = peak / (1024 ** 2)

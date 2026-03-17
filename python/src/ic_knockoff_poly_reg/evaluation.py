"""Evaluation metrics for the IC-Knock-Poly algorithm.

Computes:
  - Empirical FDR: false positives / max(1, total selections)
  - True Positive Rate (Power): true positives / max(1, true positives + false negatives)
  - Peak memory usage via tracemalloc
"""

from __future__ import annotations

import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
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

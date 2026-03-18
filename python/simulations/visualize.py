"""Visualization utilities for IC-Knock-Poly simulation results.

Provides functions to generate publication-quality plots from the
:class:`~simulations.run_simulation.SimulationResult` objects returned by
:func:`~simulations.run_simulation.run_simulation_suite`.

Four families of plots are supported:

1. **Prediction error** — R² and residual sum-of-squares under varying
   experimental scales (p, n, k, degree, or number of non-zero elements).
2. **Scalability** — wall-clock time and peak memory as problem scale grows.
3. **Selection quality metrics** — FDR, precision, recall, F1 score, and AUC
   under varying scales.
4. **Non-zero identification** — true positives, false positives, false
   negatives, and selected-term counts under varying scales.

Quick start
-----------
::

    from simulations.run_simulation import run_simulation_suite, default_configs
    from simulations.visualize import plot_all

    results = run_simulation_suite(default_configs(n_trials=5))
    plot_all(results, output_dir="figures/")

Each ``plot_*`` function accepts an optional ``ax`` (matplotlib Axes) and
returns the Axes object so that callers can compose custom layouts.

All functions require ``matplotlib``.  Install it with::

    pip install matplotlib
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Lazy matplotlib import (avoid hard dependency at import time)
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        return plt, matplotlib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for visualization.  "
            "Install it with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_X_FIELDS = ("p", "n_labeled", "k", "degree")

_FIELD_LABELS = {
    "p": "Number of base features (p)",
    "n_labeled": "Number of labeled samples (n)",
    "k": "Non-zero elements (k)",
    "degree": "Polynomial degree",
}

_METHOD_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P", "*", "X"]
_METHOD_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _group_results(results, x_field: str):
    """Group SimulationResult objects by (method, fixed_params) mapping
    x_field values to (mean, std) for each metric.

    Returns
    -------
    dict mapping method → dict of x_value → SimulationResult list
    """
    grouped: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in results:
        x_val = getattr(r.config, x_field)
        grouped[r.method][x_val].append(r)
    return grouped


def _aggregate(sim_results_list, attr: str) -> tuple[float, float]:
    """Return mean and std of ``attr`` across a list of SimulationResult."""
    vals = [getattr(r, attr) for r in sim_results_list
            if not math.isnan(getattr(r, attr))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _draw_lines(ax, grouped, x_field, y_attr, y_err_attr=None,
                y_label="", title="", legend=True):
    """Draw one line per method with optional error bars."""
    plt, _ = _require_matplotlib()
    methods = sorted(grouped.keys())
    for idx, method in enumerate(methods):
        xv_map = grouped[method]
        xs = sorted(xv_map.keys())
        ys, yerrs = [], []
        for x in xs:
            m, s = _aggregate(xv_map[x], y_attr)
            ys.append(m)
            if y_err_attr is not None:
                es, _ = _aggregate(xv_map[x], y_err_attr)
                yerrs.append(es)
            else:
                yerrs.append(s)

        color = _METHOD_COLORS[idx % len(_METHOD_COLORS)]
        marker = _METHOD_MARKERS[idx % len(_METHOD_MARKERS)]
        ax.errorbar(
            xs, ys, yerr=yerrs,
            label=method,
            color=color, marker=marker,
            linestyle="-", capsize=3,
        )

    ax.set_xlabel(_FIELD_LABELS.get(x_field, x_field))
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if legend:
        ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_prediction_error(
    results,
    x_field: str = "n_labeled",
    *,
    metric: str = "r_squared_mean",
    ax=None,
    title: Optional[str] = None,
    legend: bool = True,
):
    """Plot prediction error (R²) as ``x_field`` varies.

    Parameters
    ----------
    results : list of SimulationResult
        As returned by :func:`~simulations.run_simulation.run_simulation_suite`.
    x_field : str
        Dimension to put on the x-axis.  One of ``"p"``, ``"n_labeled"``,
        ``"k"``, ``"degree"``.  Default ``"n_labeled"``.
    metric : str
        Result attribute to plot.  Default ``"r_squared_mean"`` (R²).
        Use ``"r_squared_mean"`` for R² or any other numeric attribute on
        :class:`~simulations.run_simulation.SimulationResult`.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure is created when ``None``.
    title : str or None
        Plot title.  Auto-generated when ``None``.
    legend : bool
        Whether to draw a legend.  Default ``True``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt, _ = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    grouped = _group_results(results, x_field)
    y_label = "R² (mean ± std)" if "r_squared" in metric else metric
    _title = title or f"Prediction Error vs {_FIELD_LABELS.get(x_field, x_field)}"
    _draw_lines(ax, grouped, x_field, metric, title=_title,
                y_label=y_label, legend=legend)
    return ax


def plot_scalability(
    results,
    x_field: str = "n_labeled",
    *,
    ax_time=None,
    ax_mem=None,
    title_prefix: Optional[str] = None,
    legend: bool = True,
):
    """Plot scalability: wall-clock time and peak memory vs ``x_field``.

    Parameters
    ----------
    results : list of SimulationResult
        As returned by :func:`~simulations.run_simulation.run_simulation_suite`.
    x_field : str
        Dimension for the x-axis.  Default ``"n_labeled"``.
    ax_time : matplotlib.axes.Axes or None
        Axes for the time plot.  Created when ``None``.
    ax_mem : matplotlib.axes.Axes or None
        Axes for the memory plot.  Created when ``None``.
    title_prefix : str or None
        Prefix for plot titles.
    legend : bool
        Whether to draw legends.  Default ``True``.

    Returns
    -------
    tuple of (ax_time, ax_mem)
    """
    plt, _ = _require_matplotlib()
    if ax_time is None or ax_mem is None:
        fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(12, 4))
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    grouped = _group_results(results, x_field)
    xlab = _FIELD_LABELS.get(x_field, x_field)
    prefix = f"{title_prefix}: " if title_prefix else ""

    _draw_lines(ax_time, grouped, x_field, "elapsed_mean",
                y_label="Wall-clock time (s)",
                title=f"{prefix}Runtime vs {xlab}", legend=legend)
    _draw_lines(ax_mem, grouped, x_field, "peak_memory_mean",
                y_label="Peak memory (MB)",
                title=f"{prefix}Memory vs {xlab}", legend=legend)
    return ax_time, ax_mem


def plot_selection_metrics(
    results,
    x_field: str = "n_labeled",
    *,
    metrics: Optional[Sequence[str]] = None,
    axes=None,
    title_prefix: Optional[str] = None,
    legend: bool = True,
):
    """Plot precision, recall, F1, AUC, FDR, and TPR vs ``x_field``.

    Parameters
    ----------
    results : list of SimulationResult
        As returned by :func:`~simulations.run_simulation.run_simulation_suite`.
    x_field : str
        Dimension for the x-axis.  Default ``"n_labeled"``.
    metrics : list of str or None
        Which metric attributes to plot.  Defaults to
        ``["fdr_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]``.
    axes : sequence of matplotlib.axes.Axes or None
        Axes to draw on (one per metric).  Created when ``None``.
    title_prefix : str or None
        Prefix for plot titles.
    legend : bool
        Whether to draw legends.  Default ``True``.

    Returns
    -------
    list of matplotlib.axes.Axes
    """
    plt, _ = _require_matplotlib()
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    if metrics is None:
        metrics = ["fdr_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]

    if axes is None:
        n = len(metrics)
        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)
        _, axes_arr = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes_arr).flatten().tolist()

    grouped = _group_results(results, x_field)
    xlab = _FIELD_LABELS.get(x_field, x_field)
    prefix = f"{title_prefix}: " if title_prefix else ""

    _METRIC_NAMES = {
        "fdr_mean": "FDR (mean)",
        "tpr_mean": "TPR / Recall (mean)",
        "precision_mean": "Precision (mean)",
        "recall_mean": "Recall (mean)",
        "f1_mean": "F1 Score (mean)",
        "auc_mean": "AUC (mean)",
    }

    result_axes = []
    for ax, metric in zip(axes, metrics):
        y_label = _METRIC_NAMES.get(metric, metric)
        _draw_lines(ax, grouped, x_field, metric,
                    y_label=y_label,
                    title=f"{prefix}{y_label} vs {xlab}",
                    legend=legend)
        result_axes.append(ax)

    return result_axes


def plot_nonzero_identification(
    results,
    x_field: str = "k",
    *,
    ax=None,
    title: Optional[str] = None,
    legend: bool = True,
):
    """Plot non-zero element identification: TP, FP, FN, and n_selected.

    Parameters
    ----------
    results : list of SimulationResult
        As returned by :func:`~simulations.run_simulation.run_simulation_suite`.
    x_field : str
        Dimension for the x-axis.  Default ``"k"`` (true sparsity).
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure is created when ``None``.
    title : str or None
        Plot title.  Auto-generated when ``None``.
    legend : bool
        Whether to draw a legend.  Default ``True``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt, _ = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    grouped = _group_results(results, x_field)
    xlab = _FIELD_LABELS.get(x_field, x_field)
    _title = title or f"Non-Zero Identification vs {xlab}"
    methods = sorted(grouped.keys())

    for midx, method in enumerate(methods):
        xv_map = grouped[method]
        xs = sorted(xv_map.keys())
        color = _METHOD_COLORS[midx % len(_METHOD_COLORS)]

        tp_vals, fp_vals, fn_vals, sel_vals = [], [], [], []
        for x in xs:
            tp_m, _ = _aggregate(xv_map[x], "n_true_positives_mean")
            fp_m, _ = _aggregate(xv_map[x], "n_false_positives_mean")
            fn_m, _ = _aggregate(xv_map[x], "n_false_negatives_mean")
            sel_m, _ = _aggregate(xv_map[x], "n_selected_mean")
            tp_vals.append(tp_m)
            fp_vals.append(fp_m)
            fn_vals.append(fn_m)
            sel_vals.append(sel_m)

        prefix = f"{method}: " if len(methods) > 1 else ""
        ax.plot(xs, tp_vals, marker="o", color=color, linestyle="-",
                label=f"{prefix}True Positives")
        ax.plot(xs, fp_vals, marker="s", color=color, linestyle="--",
                label=f"{prefix}False Positives")
        ax.plot(xs, fn_vals, marker="^", color=color, linestyle=":",
                label=f"{prefix}False Negatives")
        ax.plot(xs, sel_vals, marker="D", color=color, linestyle="-.",
                label=f"{prefix}Selected")

    ax.set_xlabel(xlab)
    ax.set_ylabel("Mean count")
    ax.set_title(_title)
    if legend:
        ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Composite "plot all" function
# ---------------------------------------------------------------------------

def plot_all(
    results,
    *,
    output_dir: Optional[str] = None,
    x_fields: Optional[Sequence[str]] = None,
    dpi: int = 150,
    show: bool = False,
) -> dict[str, object]:
    """Generate all standard plots for a simulation sweep.

    Creates four families of figures for each ``x_field``:

    1. ``prediction_error`` — R² vs varying scale.
    2. ``scalability`` — runtime and memory vs varying scale.
    3. ``selection_metrics`` — FDR, precision, recall, F1, AUC vs varying scale.
    4. ``nonzero_identification`` — TP, FP, FN, n_selected vs varying scale.

    Parameters
    ----------
    results : list of SimulationResult
        As returned by :func:`~simulations.run_simulation.run_simulation_suite`.
    output_dir : str or None
        Directory to save PNG files.  Created if it does not exist.
        When ``None`` the figures are returned but not saved.
    x_fields : sequence of str or None
        Which x-axis dimensions to produce plots for.  Defaults to all
        dimensions that have more than one unique value in ``results``.
    dpi : int
        Figure resolution for saved images (default 150).
    show : bool
        Call ``plt.show()`` after generating all figures.  Default ``False``.

    Returns
    -------
    dict mapping figure-name → matplotlib.figure.Figure
    """
    plt, _ = _require_matplotlib()

    if not results:
        return {}

    # Auto-detect which x_fields have more than one unique value
    if x_fields is None:
        x_fields = []
        for field in _VALID_X_FIELDS:
            vals = {getattr(r.config, field) for r in results}
            if len(vals) > 1:
                x_fields.append(field)
        if not x_fields:
            # Fall back to all valid fields
            x_fields = list(_VALID_X_FIELDS)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    figures: dict[str, object] = {}

    for x_field in x_fields:
        xlab = x_field

        # 1. Prediction error
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_prediction_error(results, x_field=x_field, ax=ax)
        fig.tight_layout()
        name = f"prediction_error_vs_{xlab}"
        figures[name] = fig
        if output_dir:
            fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=dpi)

        # 2. Scalability (time + memory)
        fig2, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4))
        plot_scalability(results, x_field=x_field, ax_time=ax_t, ax_mem=ax_m)
        fig2.tight_layout()
        name2 = f"scalability_vs_{xlab}"
        figures[name2] = fig2
        if output_dir:
            fig2.savefig(os.path.join(output_dir, f"{name2}.png"), dpi=dpi)

        # 3. Selection metrics (FDR, precision, recall, F1, AUC)
        sel_metrics = [
            "fdr_mean", "precision_mean", "recall_mean",
            "f1_mean", "auc_mean",
        ]
        ncols = min(len(sel_metrics), 3)
        nrows = math.ceil(len(sel_metrics) / ncols)
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        ax_list = np.array(axes3).flatten().tolist()
        # Hide any spare axes
        for spare_ax in ax_list[len(sel_metrics):]:
            spare_ax.set_visible(False)
        plot_selection_metrics(
            results, x_field=x_field,
            metrics=sel_metrics, axes=ax_list[:len(sel_metrics)],
        )
        fig3.tight_layout()
        name3 = f"selection_metrics_vs_{xlab}"
        figures[name3] = fig3
        if output_dir:
            fig3.savefig(os.path.join(output_dir, f"{name3}.png"), dpi=dpi)

        # 4. Non-zero identification
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        plot_nonzero_identification(results, x_field=x_field, ax=ax4)
        fig4.tight_layout()
        name4 = f"nonzero_identification_vs_{xlab}"
        figures[name4] = fig4
        if output_dir:
            fig4.savefig(os.path.join(output_dir, f"{name4}.png"), dpi=dpi)

    if show:
        plt.show()

    return figures

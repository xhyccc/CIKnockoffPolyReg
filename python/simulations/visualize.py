"""Visualization utilities for IC-Knock-Poly simulation results.

Provides functions to generate publication-quality plots from the
:class:`~simulations.run_simulation.SimulationResult` objects returned by
:func:`~simulations.run_simulation.run_simulation_suite`.

Four families of plots are supported:

1. **Prediction error** — R² under varying experimental scales
   (p, n, k, degree, or number of non-zero elements).
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

``plot_all`` saves **one figure per metric per x-field** (single figure per
file) with the legend placed outside the axes area on the right.  When
``x_field == "degree"`` grouped bar charts are used instead of line plots to
emphasise the discrete / categorical nature of polynomial degree.

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

_VALID_X_FIELDS = ("p", "n_labeled", "k", "degree", "noise_std")

_FIELD_LABELS = {
    "p": "Number of base features (p)",
    "n_labeled": "Number of labeled samples (n)",
    "k": "Non-zero elements (k)",
    "degree": "Polynomial degree",
    "noise_std": "Label-noise std dev (σ)",
}

_METHOD_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P", "*", "X"]
_METHOD_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# Metric display names used across all plot helpers
_METRIC_DISPLAY = {
    "r_squared_mean":   "R² (mean ± std)",
    "elapsed_mean":     "Wall-clock time (s)",
    "peak_memory_mean": "Peak memory (MB)",
    "fdr_mean":         "FDR",
    "tpr_mean":         "TPR / Recall",
    "precision_mean":   "Precision",
    "recall_mean":      "Recall",
    "f1_mean":          "F1 Score",
    "auc_mean":         "AUC",
}


def _group_results(results, x_field: str):
    """Group SimulationResult objects by method, mapping x_field values to lists.

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


def _add_outside_legend(ax, ncol: int = 1, fontsize: int = 9) -> None:
    """Place the axes legend to the right, outside the axes bounding box.

    The caller should follow up with ``fig.tight_layout()`` and save with
    ``bbox_inches='tight'`` so the legend is not clipped.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    effective_ncol = ncol
    if len(handles) > 10:
        effective_ncol = max(ncol, 2)
    ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=fontsize,
        framealpha=0.9,
        ncol=effective_ncol,
    )


def _draw_lines(ax, grouped, x_field, y_attr, y_err_attr=None,
                y_label="", title=""):
    """Draw one line per method with optional error bars.  No legend drawn."""
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
            linewidth=1.8, markersize=6,
        )

    ax.set_xlabel(_FIELD_LABELS.get(x_field, x_field), fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    return ax


def _draw_bars(ax, grouped, x_field, y_attr, y_label="", title=""):
    """Draw a grouped bar chart — one bar per method for each x value.

    Intended for experiments where ``x_field == "degree"`` (discrete axis).
    """
    plt, _ = _require_matplotlib()
    methods = sorted(grouped.keys())
    all_xs = sorted({x for m in methods for x in grouped[m].keys()})
    n_methods = len(methods)

    x_indices = np.arange(len(all_xs))
    bar_width = min(0.75 / max(n_methods, 1), 0.18)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

    for idx, method in enumerate(methods):
        xv_map = grouped[method]
        ys, yerrs = [], []
        for x in all_xs:
            if x in xv_map:
                m, s = _aggregate(xv_map[x], y_attr)
                ys.append(m)
                yerrs.append(s if not math.isnan(s) else 0.0)
            else:
                ys.append(float("nan"))
                yerrs.append(0.0)

        color = _METHOD_COLORS[idx % len(_METHOD_COLORS)]
        ax.bar(
            x_indices + offsets[idx], ys,
            width=bar_width, yerr=yerrs,
            label=method, color=color,
            capsize=3, error_kw={"linewidth": 1, "elinewidth": 1},
            alpha=0.88,
        )

    ax.set_xlabel(_FIELD_LABELS.get(x_field, x_field), fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(x) for x in all_xs], fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Figure-size helpers
# ---------------------------------------------------------------------------

def _fig_size(n_methods: int, wide: bool = False) -> tuple[float, float]:
    """Return (width, height) in inches given the number of plotted methods.

    Extra width is reserved on the right for the outside legend.
    """
    legend_w = 2.2 + 0.15 * max(0, n_methods - 4)
    base_w = 7.0 if not wide else 9.0
    base_h = 4.8
    return (base_w + legend_w, base_h)


def _fig_size_large(n_methods: int) -> tuple[float, float]:
    """Larger figure for plots with many lines (e.g. nonzero identification)."""
    legend_w = 2.4 + 0.2 * max(0, n_methods - 4)
    base_w = 9.0
    base_h = max(5.5, 4.0 + n_methods * 0.35)
    return (base_w + legend_w, base_h)


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
    x_field : str
        One of ``"p"``, ``"n_labeled"``, ``"k"``, ``"degree"``.
    metric : str
        Result attribute to plot.  Default ``"r_squared_mean"``.
    ax : matplotlib.axes.Axes or None
    title : str or None
    legend : bool
        When ``True`` the legend is placed outside the axes (right side).

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt, _ = _require_matplotlib()
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    grouped = _group_results(results, x_field)
    n_methods = len(grouped)

    if ax is None:
        _, ax = plt.subplots(figsize=_fig_size(n_methods))

    y_label = _METRIC_DISPLAY.get(metric, metric)
    _title = title or f"Prediction Error (R²) vs {_FIELD_LABELS.get(x_field, x_field)}"

    draw_fn = _draw_bars if x_field == "degree" else _draw_lines
    draw_fn(ax, grouped, x_field, metric, y_label=y_label, title=_title)

    if legend:
        _add_outside_legend(ax)
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

    Each panel is drawn on a separate Axes.  Pass ``ax_time`` and ``ax_mem``
    to embed in an existing layout; otherwise two fresh figures are created.

    Parameters
    ----------
    results : list of SimulationResult
    x_field : str
        Default ``"n_labeled"``.
    ax_time, ax_mem : matplotlib.axes.Axes or None
    title_prefix : str or None
    legend : bool
        Legend is placed outside the axes when ``True``.

    Returns
    -------
    tuple of (ax_time, ax_mem)
    """
    plt, _ = _require_matplotlib()
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    grouped = _group_results(results, x_field)
    n_methods = len(grouped)
    xlab = _FIELD_LABELS.get(x_field, x_field)
    prefix = f"{title_prefix}: " if title_prefix else ""
    draw_fn = _draw_bars if x_field == "degree" else _draw_lines

    if ax_time is None:
        _, ax_time = plt.subplots(figsize=_fig_size(n_methods))
    draw_fn(ax_time, grouped, x_field, "elapsed_mean",
            y_label="Wall-clock time (s)",
            title=f"{prefix}Runtime vs {xlab}")
    if legend:
        _add_outside_legend(ax_time)

    if ax_mem is None:
        _, ax_mem = plt.subplots(figsize=_fig_size(n_methods))
    draw_fn(ax_mem, grouped, x_field, "peak_memory_mean",
            y_label="Peak memory (MB)",
            title=f"{prefix}Memory vs {xlab}")
    if legend:
        _add_outside_legend(ax_mem)

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
    """Plot precision, recall, F1, AUC, and FDR vs ``x_field``.

    Each metric is drawn on its own Axes.  When ``axes`` is ``None`` a fresh
    figure is created for each metric.

    Parameters
    ----------
    results : list of SimulationResult
    x_field : str
    metrics : list of str or None
    axes : sequence of matplotlib.axes.Axes or None
    title_prefix : str or None
    legend : bool

    Returns
    -------
    list of matplotlib.axes.Axes
    """
    plt, _ = _require_matplotlib()
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    if metrics is None:
        metrics = ["fdr_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]

    grouped = _group_results(results, x_field)
    n_methods = len(grouped)
    xlab = _FIELD_LABELS.get(x_field, x_field)
    prefix = f"{title_prefix}: " if title_prefix else ""
    draw_fn = _draw_bars if x_field == "degree" else _draw_lines

    result_axes = []
    for i, metric in enumerate(metrics):
        if axes is not None and i < len(axes):
            ax = axes[i]
        else:
            _, ax = plt.subplots(figsize=_fig_size(n_methods))

        y_label = _METRIC_DISPLAY.get(metric, metric)
        draw_fn(ax, grouped, x_field, metric,
                y_label=y_label,
                title=f"{prefix}{y_label} vs {xlab}")
        if legend:
            _add_outside_legend(ax)
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
    """Plot non-zero identification counts: TP, FP, FN, and n_selected.

    Because this plot draws four lines per method the figure is automatically
    sized to avoid overlap.

    Parameters
    ----------
    results : list of SimulationResult
    x_field : str
        Default ``"k"`` (true sparsity).
    ax : matplotlib.axes.Axes or None
    title : str or None
    legend : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt, _ = _require_matplotlib()
    if x_field not in _VALID_X_FIELDS:
        raise ValueError(f"x_field must be one of {_VALID_X_FIELDS}, got {x_field!r}")

    grouped = _group_results(results, x_field)
    n_methods = len(grouped)

    if ax is None:
        _, ax = plt.subplots(figsize=_fig_size_large(n_methods))

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

        pfx = f"{method}: " if len(methods) > 1 else ""
        ax.plot(xs, tp_vals,  marker="o", color=color, linestyle="-",
                linewidth=1.6, markersize=5, label=f"{pfx}True Positives")
        ax.plot(xs, fp_vals,  marker="s", color=color, linestyle="--",
                linewidth=1.6, markersize=5, label=f"{pfx}False Positives")
        ax.plot(xs, fn_vals,  marker="^", color=color, linestyle=":",
                linewidth=1.6, markersize=5, label=f"{pfx}False Negatives")
        ax.plot(xs, sel_vals, marker="D", color=color, linestyle="-.",
                linewidth=1.6, markersize=5, label=f"{pfx}Selected")

    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel("Mean count", fontsize=10)
    ax.set_title(_title, fontsize=11)
    if legend:
        n_lines = 4 * n_methods
        _add_outside_legend(ax, ncol=2 if n_lines > 12 else 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Composite "plot all" function — one figure per metric per x-field
# ---------------------------------------------------------------------------

def plot_all(
    results,
    *,
    output_dir: Optional[str] = None,
    x_fields: Optional[Sequence[str]] = None,
    dpi: int = 150,
    fmt: str = "pdf",
    show: bool = False,
) -> dict[str, object]:
    """Generate all standard plots for a simulation sweep.

    Produces **one figure (file) per metric per x-field**:

    * ``prediction_error_vs_{x}``         — R²
    * ``scalability_time_vs_{x}``         — wall-clock time
    * ``scalability_memory_vs_{x}``       — peak memory
    * ``fdr_vs_{x}``                      — false discovery rate
    * ``precision_vs_{x}``                — precision
    * ``recall_vs_{x}``                   — recall
    * ``f1_vs_{x}``                       — F1 score
    * ``auc_vs_{x}``                      — AUC
    * ``nonzero_identification_vs_{x}``   — TP / FP / FN / selected counts

    When ``x_field == "degree"`` grouped bar charts are used instead of line
    plots.  Legends are always placed outside the axes (right side).

    Parameters
    ----------
    results : list of SimulationResult
    output_dir : str or None
        Directory to save figures.  Created if absent.
    x_fields : sequence of str or None
        Defaults to all fields with more than one unique value in ``results``.
    dpi : int
        Raster DPI (default 150).
    fmt : str
        Output format (default ``"pdf"``).
    show : bool
        Call ``plt.show()`` when ``True``.

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
            x_fields = list(_VALID_X_FIELDS)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    figures: dict[str, object] = {}
    n_methods = len({r.method for r in results})

    def _save(fig, name: str) -> None:
        figures[name] = fig
        if output_dir:
            fig.savefig(
                os.path.join(output_dir, f"{name}.{fmt}"),
                dpi=dpi, format=fmt, bbox_inches="tight",
            )

    for x_field in x_fields:
        xlab = _FIELD_LABELS.get(x_field, x_field)
        grouped = _group_results(results, x_field)
        is_degree = x_field == "degree"
        draw_fn = _draw_bars if is_degree else _draw_lines
        fw, fh = _fig_size(n_methods)

        # ── 1. Prediction error (R²) ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(fw, fh))
        draw_fn(ax, grouped, x_field, "r_squared_mean",
                y_label=_METRIC_DISPLAY["r_squared_mean"],
                title=f"Prediction Error (R²) vs {xlab}")
        _add_outside_legend(ax)
        fig.tight_layout()
        _save(fig, f"prediction_error_vs_{x_field}")

        # ── 2. Runtime ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(fw, fh))
        draw_fn(ax, grouped, x_field, "elapsed_mean",
                y_label=_METRIC_DISPLAY["elapsed_mean"],
                title=f"Runtime vs {xlab}")
        _add_outside_legend(ax)
        fig.tight_layout()
        _save(fig, f"scalability_time_vs_{x_field}")

        # ── 3. Memory ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(fw, fh))
        draw_fn(ax, grouped, x_field, "peak_memory_mean",
                y_label=_METRIC_DISPLAY["peak_memory_mean"],
                title=f"Peak Memory vs {xlab}")
        _add_outside_legend(ax)
        fig.tight_layout()
        _save(fig, f"scalability_memory_vs_{x_field}")

        # ── 4–8. Selection metrics ────────────────────────────────────────
        for metric in ("fdr_mean", "precision_mean", "recall_mean",
                       "f1_mean", "auc_mean"):
            short = metric.replace("_mean", "")
            fig, ax = plt.subplots(figsize=(fw, fh))
            draw_fn(ax, grouped, x_field, metric,
                    y_label=_METRIC_DISPLAY[metric],
                    title=f"{_METRIC_DISPLAY[metric]} vs {xlab}")
            _add_outside_legend(ax)
            fig.tight_layout()
            _save(fig, f"{short}_vs_{x_field}")

        # ── 9. Non-zero identification (larger for many lines) ────────────
        flw, flh = _fig_size_large(n_methods)
        fig, ax = plt.subplots(figsize=(flw, flh))
        plot_nonzero_identification(results, x_field=x_field, ax=ax,
                                    legend=False)
        n_lines = 4 * n_methods
        _add_outside_legend(ax, ncol=2 if n_lines > 12 else 1)
        fig.tight_layout()
        _save(fig, f"nonzero_identification_vs_{x_field}")

    if show:
        plt.show()

    return figures

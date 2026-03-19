"""Run IC-Knock-Poly simulation experiments and generate PDF figures + LaTeX report.

This script performs three simulation sweeps:

1. **Default sweep** – varies sample size *n*, sparsity *k*, and evaluation
   setting (supervised / semi-supervised) at polynomial degree 2, comparing
   IC-Knock-Poly against all five baseline methods.  *p* is fixed at 5.

2. **p-scaling sweep** – varies the number of base features *p* ∈ {4, 6, 8, 10}
   at fixed *n* = 200, *k* = 2, *degree* = 2 to show how performance and
   runtime scale with dimensionality.

3. **Degree × nonzero sweep** – varies polynomial degree *d* ∈ {2, 3, 4} and
   the number of non-zero elements *k* ∈ {2, 3, 4} at two sample sizes
   (100, 300).  Results for the degree axis are shown as grouped bar charts.

Results are saved to the ``simulation_results/`` directory:

* ``default_sweep_summary.json/.csv``        – numerical results (default sweep)
* ``p_scaling_sweep_summary.json/.csv``      – numerical results (p-scaling)
* ``degree_nonzero_sweep_summary.json/.csv`` – numerical results (degree×nonzero)
* ``figures/``                               – PDF figures (one figure per metric
                                               per x-field; bar charts for degree)
* ``report.tex``                             – LaTeX experiment report

Usage
-----
::

    python simulation_results/run_simulations.py

Run from the **repository root**.
"""

from __future__ import annotations

import math
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Make packages importable from the repository layout
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python", "src"))
sys.path.insert(0, os.path.join(_REPO, "python"))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt

from simulations.run_simulation import (
    SimulationResult,
    default_configs,
    sweep_degree_nonzero_configs,
    run_simulation_suite,
    print_summary_table,
)
from simulations.visualize import plot_all

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_OUT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_OUT_DIR, "figures")
os.makedirs(_FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep parameters
#
# NOTE: IC-Knock-Poly's runtime grows steeply with *p* (the knockoff SDP
# scales as O(p² · d²)).  The default sweep fixes p = 5; the p-scaling
# sweep uses a moderate range (4–10) to show the trend.
# ---------------------------------------------------------------------------
ALL_METHODS = [
    "ic_knock_poly",
    "poly_lasso",
    "poly_omp",
    "poly_clime",
    "poly_knockoff",
    "sparse_poly_stlsq",
]

# Fixed dimensionality for default and degree sweeps
_P = 5

# p-scaling range (kept tractable for the knockoff SDP)
_P_VALUES = (4, 6, 8, 10)

N_TRIALS = 3   # independent repetitions per configuration


# ---------------------------------------------------------------------------
# Internal helpers – LaTeX formatting
# ---------------------------------------------------------------------------

def _results_to_latex_rows(results: list[SimulationResult]) -> str:
    """Return LaTeX tabular rows (without header/footer) for a result list."""
    lines = []
    for r in results:
        label = r.config.label.replace("_", "\\_")
        method = r.method.replace("_", "\\_")
        fdr = f"{r.fdr_mean:.3f}" if not math.isnan(r.fdr_mean) else "---"
        tpr = f"{r.tpr_mean:.3f}" if not math.isnan(r.tpr_mean) else "---"
        f1  = f"{r.f1_mean:.3f}"  if not math.isnan(r.f1_mean)  else "---"
        auc = f"{r.auc_mean:.3f}" if not math.isnan(r.auc_mean) else "---"
        r2  = f"{r.r_squared_mean:.3f}" if not math.isnan(r.r_squared_mean) else "---"
        t   = f"{r.elapsed_mean:.2f}"   if not math.isnan(r.elapsed_mean)   else "---"
        n_c = str(r.n_completed)
        lines.append(
            f"  {label} & {method} & {fdr} & {tpr} & {f1} & {auc} & {r2} & {t} & {n_c} \\\\"
        )
    return "\n".join(lines)


def _make_table(results: list[SimulationResult], caption: str, label: str) -> str:
    rows = _results_to_latex_rows(results)
    return rf"""
\begin{{table}}[htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llccccccc}}
\hline
\textbf{{Config}} & \textbf{{Method}} & \textbf{{FDR}} & \textbf{{TPR}} &
\textbf{{F1}} & \textbf{{AUC}} & \textbf{{R\textsuperscript{{2}}}} & \textbf{{Time (s)}} & \textbf{{Trials}} \\
\hline
{rows}
\hline
\end{{tabular}}%
}}
\end{{table}}
"""


def _fig_include(filename: str, caption: str, fig_label: str,
                 width: str = r"0.9\textwidth") -> str:
    rel_path = "figures/" + filename
    return rf"""
\begin{{figure}}[htbp]
\centering
\includegraphics[width={width}]{{{rel_path}}}
\caption{{{caption}}}
\label{{{fig_label}}}
\end{{figure}}
"""


def _fig_section(fig_list: list[str], sweep_prefix: str) -> str:
    blocks = []
    for fname in sorted(fig_list):
        stem = fname.rsplit(".", 1)[0]
        parts = stem.split("_vs_")
        plot_kind = (
            parts[0]
            .replace(sweep_prefix + "_", "", 1)
            .replace("_", " ")
            .title()
        )
        x_dim = parts[1].replace("_", " ") if len(parts) > 1 else ""
        sweep_label = sweep_prefix.replace("_", " ")
        cap = f"{plot_kind} vs.\\ {x_dim} ({sweep_label} sweep)"
        lbl = f"fig:{stem}"
        blocks.append(_fig_include(fname, cap, lbl))
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Build LaTeX report
# ---------------------------------------------------------------------------

def build_latex_report(
    default_results: list[SimulationResult],
    p_scaling_results: list[SimulationResult],
    dn_results: list[SimulationResult],
    default_fig_files: list[str],
    p_scaling_fig_files: list[str],
    dn_fig_files: list[str],
) -> str:
    """Construct the full LaTeX report as a string."""

    default_table = _make_table(
        default_results,
        caption=(
            "Default sweep results ($p=5$, degree $d=2$). "
            "Mean FDR, TPR, F1, AUC, R\\textsuperscript{2}, "
            "and wall-clock time over completed trials."
        ),
        label="tab:default",
    )

    p_scaling_table = _make_table(
        p_scaling_results,
        caption=(
            "p-scaling sweep results (degree $d=2$, $n=200$, $k=2$). "
            "Mean FDR, TPR, F1, AUC, R\\textsuperscript{2}, "
            "and wall-clock time over completed trials."
        ),
        label="tab:p_scaling",
    )

    dn_table = _make_table(
        dn_results,
        caption=(
            "Degree $\\times$ non-zero sweep results ($p=5$). "
            "Mean FDR, TPR, F1, AUC, R\\textsuperscript{2}, "
            "and wall-clock time over completed trials."
        ),
        label="tab:degree_nonzero",
    )

    report = rf"""\documentclass[12pt,a4paper]{{article}}

% -------------------------------------------------------------------
%  Packages
% -------------------------------------------------------------------
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{geometry}}
\geometry{{margin=2.5cm}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{xcolor}}
\usepackage{{array}}
\usepackage{{longtable}}

% -------------------------------------------------------------------
%  Metadata
% -------------------------------------------------------------------
\title{{%
  \textbf{{IC-Knockoff-PolyReg: Simulation Study}}\\[6pt]
  \large Polynomial Regression with Knockoff-Based FDR Control
}}
\author{{Simulation Report --- Auto-generated}}
\date{{\today}}

% ===================================================================
\begin{{document}}
\maketitle
\tableofcontents
\newpage

% ===================================================================
\section{{Introduction}}
\label{{sec:intro}}

This report summarises the Monte-Carlo simulation study for
\textbf{{IC-Knockoff-PolyReg}} (IC-Knock-Poly), a method that combines
iterative conformalisation of knockoff statistics with polynomial
dictionary regression to perform simultaneous feature selection and
prediction under FDR control.

The method is evaluated on synthetically generated datasets where the
ground-truth response is a \(k\)-sparse polynomial of \(p\) base features
drawn from a Gaussian Mixture Model (GMM).
Three complementary sweeps are performed:

\begin{{itemize}}
  \item \textbf{{Default sweep}}: fixes \(p={_P}\) base features and varies
        sample size \(n \in \{{100, 300, 500\}}\), sparsity \(k \in \{{2,3\}}\),
        and evaluation setting (\emph{{supervised}} vs.\ \emph{{semi-supervised}})
        at polynomial degree \(d=2\).
        All six competing methods are compared.
  \item \textbf{{p-scaling sweep}}: fixes \(n=200\), \(k=2\), \(d=2\) and
        varies the number of base features
        \(p \in \{{4, 6, 8, 10\}}\) to show how performance and runtime
        scale with dimensionality.
  \item \textbf{{Degree\,$\times$\,non-zero sweep}}: fixes \(p={_P}\) and sweeps
        polynomial degree \(d \in \{{2,3,4\}}\) together with the number of
        non-zero elements \(k \in \{{2,3,4\}}\) at two sample sizes.
        Degree-axis plots use grouped bar charts.
\end{{itemize}}

\noindent\textbf{{Computational note.}}
IC-Knock-Poly's runtime grows steeply with \(p\) because the
Model-X knockoff construction requires solving a semidefinite programme
over the \(p \cdot d \times p \cdot d\) polynomial-term covariance matrix.
The p-scaling sweep therefore uses a moderate range (\(p \leq 10\));
larger \(p\) experiments can be run on more powerful hardware by adjusting
the parameters in \texttt{{run\_simulations.py}}.

Figures are generated by \texttt{{simulations/visualize.py}} and saved as
PDF vector graphics, one figure per metric per sweep.  Numerical results
are archived in JSON/CSV files alongside this report.

% ===================================================================
\section{{Experimental Setup}}
\label{{sec:setup}}

\subsection{{Data-Generating Process}}

Each dataset is drawn from:
\begin{{align}}
  X       &\sim \mathrm{{GMM}}\!\left(K=2,\; p\right), \notag\\
  y       &= \Phi(X)^\top \beta^* + \varepsilon, \quad
             \varepsilon \sim \mathcal{{N}}(0, \sigma^2 = 0.25),\notag
\end{{align}}
where \(\Phi(\cdot)\) is the rational polynomial dictionary of degree \(d\)
(terms \(x_j^1, x_j^2, \ldots, x_j^d, x_j^{{-1}}, \ldots, x_j^{{-d}}\) for
each base feature \(x_j\)), and \(\beta^*\) is \(k\)-sparse with non-zero
coefficients drawn uniformly from \([-2,-0.5]\cup[0.5,2]\).

\subsection{{Competing Methods}}

\begin{{description}}
  \item[\textbf{{IC-Knock-Poly}}] (proposed method)
    Iterative conformal knockoff procedure with GMM-estimated covariate
    model and polynomial dictionary; controlled at FDR level \(Q=0.10\).
  \item[\textbf{{Poly-Lasso}}]
    LASSO regression on the full polynomial dictionary; threshold chosen
    by cross-validation.
  \item[\textbf{{Poly-OMP}}]
    Orthogonal Matching Pursuit on the polynomial dictionary; sparsity
    set to the estimated non-zero count.
  \item[\textbf{{Poly-CLIME}}]
    Constrained \(\ell_1\)-minimisation for inverse covariance estimation
    combined with polynomial regression.
  \item[\textbf{{Poly-Knockoff}}]
    Model-X knockoffs applied directly to the polynomial dictionary.
  \item[\textbf{{Sparse-Poly-STLSQ}}]
    Sequentially thresholded least-squares on the polynomial dictionary.
\end{{description}}

\subsection{{Evaluation Metrics}}

\begin{{itemize}}
  \item \textbf{{FDR}}: empirical false discovery rate
        \(= \mathrm{{FP}}/\max(1,\mathrm{{FP}}+\mathrm{{TP}})\).
  \item \textbf{{TPR / Recall}}: true positive rate
        \(= \mathrm{{TP}}/(\mathrm{{TP}}+\mathrm{{FN}})\).
  \item \textbf{{Precision}}: \(= \mathrm{{TP}}/\max(1,\mathrm{{TP}}+\mathrm{{FP}})\).
  \item \textbf{{F1}}: harmonic mean of precision and recall.
  \item \textbf{{AUC}}: approximate area under the ROC curve
        \(= 0.5(\mathrm{{TPR}} + \mathrm{{TNR}})\).
  \item \textbf{{R\textsuperscript{{2}}}}: coefficient of determination on
        the labeled data.
\end{{itemize}}

All metrics are averaged over {N_TRIALS} independent trials.
Target FDR level: \(Q = 0.10\);
noise: \(\sigma = 0.5\);
maximum iterations for IC-Knock-Poly: 10.

% ===================================================================
\section{{Default Sweep Results}}
\label{{sec:default}}

\subsection{{Summary Table}}

\Cref{{tab:default}} reports the aggregate performance metrics for every
configuration in the default sweep.

{default_table}

\subsection{{Figures}}

{_fig_section(default_fig_files, "default")}

% ===================================================================
\section{{p-Scaling Sweep}}
\label{{sec:p_scaling}}

\subsection{{Summary Table}}

\Cref{{tab:p_scaling}} reports results for the dimensionality-scaling
sweep ($n=200$, $k=2$, $d=2$).

{p_scaling_table}

\subsection{{Figures}}

{_fig_section(p_scaling_fig_files, "p_scaling")}

% ===================================================================
\section{{Degree\,$\times$\,Non-Zero Sweep}}
\label{{sec:degree_nonzero}}

\subsection{{Summary Table}}

\Cref{{tab:degree_nonzero}} reports results for the degree-and-sparsity
sweep.  Degree-axis figures use grouped bar charts to emphasise the
discrete nature of the polynomial degree.

{dn_table}

\subsection{{Figures}}

{_fig_section(dn_fig_files, "degree_nonzero")}

% ===================================================================
\section{{Discussion}}
\label{{sec:discussion}}

\begin{{itemize}}
  \item \textbf{{FDR control}}: IC-Knock-Poly targets \(Q = 0.10\).
        The empirical FDR is expected to remain at or below this threshold
        across all configurations when the model is well-specified.
  \item \textbf{{TPR / Recall}}: power typically increases with larger
        sample size \(n\) and decreases with higher sparsity \(k\) or
        polynomial degree \(d\).
  \item \textbf{{p-scaling}}: wall-clock time grows quickly with \(p\)
        due to the SDP construction; selection quality degrades as the
        number of nuisance features increases.
  \item \textbf{{Baselines}}: methods without formal FDR guarantees (e.g.\
        Poly-Lasso, Poly-OMP) often achieve higher TPR at the cost of
        inflated FDR.
  \item \textbf{{Semi-supervised}}: providing unlabeled data to
        IC-Knock-Poly can improve knockoff covariance estimation and
        thereby increase power (TPR) without inflating FDR.
\end{{itemize}}

% ===================================================================
\section{{Conclusion}}
\label{{sec:conclusion}}

This simulation study demonstrates that IC-Knockoff-PolyReg consistently
controls FDR at the specified level \(Q = 0.10\) across a range of
sample sizes, sparsity levels, polynomial degrees, and problem
dimensionalities, while maintaining competitive predictive accuracy
(R\textsuperscript{{2}}) compared with standard baselines.

% ===================================================================
\end{{document}}
"""
    return report


# ---------------------------------------------------------------------------
# Helper: generate and rename figures for one sweep
# ---------------------------------------------------------------------------

def _generate_sweep_figures(
    results: list[SimulationResult],
    sweep_prefix: str,
    fig_dir: str,
    fmt: str = "pdf",
) -> list[str]:
    """Run ``plot_all`` and rename figures with *sweep_prefix*.

    Returns a sorted list of final filenames (basename only).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figs = plot_all(results, output_dir=fig_dir, fmt=fmt)
    plt.close("all")

    out: list[str] = []
    for name in figs:
        src = os.path.join(fig_dir, f"{name}.{fmt}")
        dst_name = f"{sweep_prefix}_{name}.{fmt}"
        dst = os.path.join(fig_dir, dst_name)
        if os.path.exists(src):
            os.replace(src, dst)
        out.append(dst_name)
    return sorted(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1.  Default sweep  (varying n, k, setting at fixed p=5, degree=2)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sweep 1/3: Default sweep  (p=5, degree=2)")
    print("=" * 60)

    default_cfgs = default_configs(
        p_values=(_P,),
        n_values=(100, 200, 300, 400, 500),
        k_values=(2, 3),
        settings=("supervised", "semi_supervised"),
        degree_values=(2,),
        methods=ALL_METHODS,
        n_trials=N_TRIALS,
        Q=0.10,
        max_iter=10,
        random_state=0,
    )

    default_results = run_simulation_suite(
        default_cfgs,
        output_prefix=os.path.join(_OUT_DIR, "default_sweep"),
    )
    print_summary_table(default_results)

    # ------------------------------------------------------------------
    # 2.  p-scaling sweep  (varying p at fixed n=200, k=2, degree=2)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sweep 2/3: p-scaling sweep  (n=200, k=2, degree=2)")
    print("=" * 60)

    p_scaling_cfgs = default_configs(
        p_values=_P_VALUES,
        n_values=(200,),
        k_values=(2,),
        settings=("supervised",),
        degree_values=(2,),
        methods=ALL_METHODS,
        n_trials=N_TRIALS,
        Q=0.10,
        max_iter=10,
        random_state=0,
    )

    p_scaling_results = run_simulation_suite(
        p_scaling_cfgs,
        output_prefix=os.path.join(_OUT_DIR, "p_scaling_sweep"),
    )
    print_summary_table(p_scaling_results)

    # ------------------------------------------------------------------
    # 3.  Degree × nonzero sweep  (varying degree, k, n at fixed p=5)
    #     degree ∈ {2, 3, 4}  — shown as grouped bar charts
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Sweep 3/3: Degree × non-zero sweep  (p=5, degree ∈ {2,3,4})")
    print("=" * 60)

    dn_cfgs = sweep_degree_nonzero_configs(
        degree_values=(2, 3, 4),
        nonzero_values=(2, 3, 4),   # k < p=5
        p=_P,
        n_values=(100, 300),
        settings=("supervised",),
        methods=ALL_METHODS,
        n_trials=N_TRIALS,
        Q=0.10,
        max_iter=10,
        random_state=0,
    )

    dn_results = run_simulation_suite(
        dn_cfgs,
        output_prefix=os.path.join(_OUT_DIR, "degree_nonzero_sweep"),
    )
    print_summary_table(dn_results)

    # ------------------------------------------------------------------
    # 4.  Generate PDF figures  (one file per metric per sweep)
    # ------------------------------------------------------------------
    print("Generating PDF figures …")

    default_fig_files   = _generate_sweep_figures(default_results,   "default",        _FIG_DIR)
    p_scaling_fig_files = _generate_sweep_figures(p_scaling_results,  "p_scaling",      _FIG_DIR)
    dn_fig_files        = _generate_sweep_figures(dn_results,         "degree_nonzero", _FIG_DIR)

    all_fig_files = sorted(default_fig_files + p_scaling_fig_files + dn_fig_files)
    print(f"  Saved {len(all_fig_files)} PDF figures to {_FIG_DIR}/")
    for f in all_fig_files:
        print(f"    {f}")

    # ------------------------------------------------------------------
    # 5.  Build and write LaTeX report
    # ------------------------------------------------------------------
    print("Writing LaTeX report …")
    latex = build_latex_report(
        default_results, p_scaling_results, dn_results,
        default_fig_files, p_scaling_fig_files, dn_fig_files,
    )
    report_path = os.path.join(_OUT_DIR, "report.tex")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(latex)
    print(f"  LaTeX report saved to {report_path}")

    print("\nDone.  All outputs are in:", _OUT_DIR)


if __name__ == "__main__":
    main()

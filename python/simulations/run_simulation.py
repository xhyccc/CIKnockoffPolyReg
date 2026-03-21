"""Simulation sweep for IC-Knock-Poly.

Runs IC-Knock-Poly (and optionally all baselines) on synthesised datasets
that vary in:

* **Number of base features** *p* (dimensions).
* **Number of labeled samples** *n*.
* **Evaluation setting**: *supervised* (no unlabeled data) or
  *semi-supervised* (unlabeled pool of size ``n_unlabeled_ratio × n``).
* **Degree of polynomials**: maximum polynomial exponent (e.g. 2 or 3).
* **Number of non-zero polynomial terms** *k*: sparsity of the true model.

Each configuration is repeated ``n_trials`` times with different random seeds
so that empirical FDR, TPR, precision, recall, F1, and AUC can be averaged.

Command-line usage
------------------
::

    # Run the default sweep and print a summary table
    python -m simulations.run_simulation

    # Customise sweep parameters
    python -m simulations.run_simulation \\
        --p 5 10 20 \\
        --n 100 300 500 \\
        --k 2 3 \\
        --degree 2 3 \\
        --n-trials 5 \\
        --output results/sweep \\
        --methods ic_knock_poly poly_lasso

    # Run the degree × nonzero sweep (degree=[2,3], k=[5,10,15,20])
    python -m simulations.run_simulation --sweep degree_nonzero \\
        --output results/degree_nonzero

Python API
----------
::

    from simulations.run_simulation import (
        run_simulation_suite, default_configs, sweep_degree_nonzero_configs,
    )
    results = run_simulation_suite(default_configs())
    for r in results:
        print(r.config, r.fdr_mean, r.tpr_mean, r.f1_mean)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Make packages importable when run directly
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ic_knockoff_poly_reg import ICKnockoffPolyReg
from ic_knockoff_poly_reg.evaluation import ResultBundle

from baselines.data_loader import DataLoader
from baselines.run_comparison import run_comparison

from .data_generator import generate_simulation, SimulatedDataset


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Parameters for one simulation scenario.

    Attributes
    ----------
    p : int
        Number of base features (dimensionality).
    n_labeled : int
        Number of labeled training samples.
    k : int
        Sparsity — number of non-zero polynomial terms in β*.
    setting : str
        ``"supervised"`` or ``"semi_supervised"``.
    n_unlabeled_ratio : float
        Multiplier for the number of unlabeled samples in the
        semi-supervised setting: ``n_unlabeled = n_unlabeled_ratio × n_labeled``.
        Ignored when ``setting == "supervised"``.
    degree : int
        Maximum polynomial exponent for IC-Knock-Poly and baselines.
    n_components : int
        True number of GMM components used to generate X.
    Q : float
        Target FDR level.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    n_trials : int
        Number of independent simulation trials.
    methods : list of str
        Method names to evaluate.  Any subset of the names accepted by
        ``run_comparison``.
    max_iter : int
        Maximum IC-Knock-Poly iterations.
    random_state : int or None
        Base random seed; trial *t* uses seed ``random_state + t``.
    label : str
        Human-readable label for this configuration.
    """

    p: int = 10
    n_labeled: int = 200
    k: int = 2
    setting: str = "supervised"
    n_unlabeled_ratio: float = 4.0
    degree: int = 2
    n_components: int = 2
    Q: float = 0.10
    noise_std: float = 0.5
    n_trials: int = 10
    methods: list = field(default_factory=lambda: ["ic_knock_poly"])
    max_iter: int = 10
    random_state: Optional[int] = 0
    label: str = ""
    backend: str = "rust"

    def __post_init__(self):
        if not self.label:
            noise_tag = f"_noise{self.noise_std:.2g}" if self.noise_std != 0.5 else ""
            self.label = (
                f"p{self.p}_n{self.n_labeled}_k{self.k}"
                f"_d{self.degree}_{self.setting}{noise_tag}"
            )


# ---------------------------------------------------------------------------
# Per-trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Outcome of a single simulation trial.

    Attributes
    ----------
    trial : int
        Trial index (0-based).
    method : str
        Method name.
    fdr : float or None
        Empirical FDR for this trial.
    tpr : float or None
        True positive rate for this trial.
    precision : float or None
        Precision (TP / (TP + FP)) for this trial.
    recall : float or None
        Recall — alias for TPR (TP / (TP + FN)).
    f1 : float or None
        F1 score (harmonic mean of precision and recall).
    auc : float or None
        Approximate AUC: 0.5 * (TPR + specificity).
    n_selected : int
        Number of selected polynomial terms.
    n_true_positives : int or None
        Number of true positive selections.
    n_false_positives : int or None
        Number of false positive selections.
    n_false_negatives : int or None
        Number of false negative (missed) true features.
    r_squared : float
        Coefficient of determination.
    elapsed_seconds : float
        Wall-clock fitting time.
    peak_memory_mb : float
        Peak memory usage.
    """

    trial: int
    method: str
    fdr: Optional[float]
    tpr: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    auc: Optional[float]
    n_selected: int
    n_true_positives: Optional[int]
    n_false_positives: Optional[int]
    n_false_negatives: Optional[int]
    r_squared: float
    elapsed_seconds: float
    peak_memory_mb: float

    def to_dict(self) -> dict:
        return {
            "trial": self.trial,
            "method": self.method,
            "fdr": self.fdr,
            "tpr": self.tpr,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "n_selected": self.n_selected,
            "n_true_positives": self.n_true_positives,
            "n_false_positives": self.n_false_positives,
            "n_false_negatives": self.n_false_negatives,
            "r_squared": self.r_squared,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }


# ---------------------------------------------------------------------------
# Aggregated simulation result (across all trials)
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Aggregated outcomes for one simulation scenario across all trials.

    Attributes
    ----------
    config : SimulationConfig
        The configuration that produced this result.
    method : str
        Method name.
    fdr_mean : float
        Mean empirical FDR over completed trials.
    fdr_std : float
        Standard deviation of empirical FDR.
    tpr_mean : float
        Mean TPR over completed trials.
    tpr_std : float
        Standard deviation of TPR.
    precision_mean : float
        Mean precision (TP / (TP + FP)) over completed trials.
    precision_std : float
        Standard deviation of precision.
    recall_mean : float
        Mean recall (alias for TPR) over completed trials.
    recall_std : float
        Standard deviation of recall.
    f1_mean : float
        Mean F1 score over completed trials.
    f1_std : float
        Standard deviation of F1 score.
    auc_mean : float
        Mean approximate AUC over completed trials.
    auc_std : float
        Standard deviation of approximate AUC.
    n_selected_mean : float
        Mean number of selected terms.
    n_true_positives_mean : float
        Mean number of true positive selections.
    n_false_positives_mean : float
        Mean number of false positive selections.
    n_false_negatives_mean : float
        Mean number of missed true features.
    r_squared_mean : float
        Mean R².
    elapsed_mean : float
        Mean wall-clock time per trial (seconds).
    peak_memory_mean : float
        Mean peak memory usage (MB).
    n_completed : int
        Number of trials that completed without error.
    trial_results : list of TrialResult
        Raw per-trial results.
    """

    config: SimulationConfig
    method: str
    fdr_mean: float
    fdr_std: float
    tpr_mean: float
    tpr_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    auc_mean: float
    auc_std: float
    n_selected_mean: float
    n_true_positives_mean: float
    n_false_positives_mean: float
    n_false_negatives_mean: float
    r_squared_mean: float
    elapsed_mean: float
    peak_memory_mean: float
    n_completed: int
    trial_results: list = field(default_factory=list)

    def to_dict(self) -> dict:
        cfg = self.config
        return {
            "label": cfg.label,
            "p": cfg.p,
            "n_labeled": cfg.n_labeled,
            "k": cfg.k,
            "degree": cfg.degree,
            "noise_std": cfg.noise_std,
            "setting": cfg.setting,
            "method": self.method,
            "fdr_mean": self.fdr_mean,
            "fdr_std": self.fdr_std,
            "tpr_mean": self.tpr_mean,
            "tpr_std": self.tpr_std,
            "precision_mean": self.precision_mean,
            "precision_std": self.precision_std,
            "recall_mean": self.recall_mean,
            "recall_std": self.recall_std,
            "f1_mean": self.f1_mean,
            "f1_std": self.f1_std,
            "auc_mean": self.auc_mean,
            "auc_std": self.auc_std,
            "n_selected_mean": self.n_selected_mean,
            "n_true_positives_mean": self.n_true_positives_mean,
            "n_false_positives_mean": self.n_false_positives_mean,
            "n_false_negatives_mean": self.n_false_negatives_mean,
            "r_squared_mean": self.r_squared_mean,
            "elapsed_mean": self.elapsed_mean,
            "peak_memory_mean": self.peak_memory_mean,
            "n_completed": self.n_completed,
        }

    def to_csv_row(self) -> dict:
        d = self.to_dict()
        d.pop("label", None)
        return d


# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------

def run_simulation(config: SimulationConfig) -> list[SimulationResult]:
    """Run all trials for one ``SimulationConfig``.

    Parameters
    ----------
    config : SimulationConfig
        Defines the scenario (p, n, k, setting, methods, …).

    Returns
    -------
    list of SimulationResult
        One entry per method in ``config.methods``.
    """
    # Accumulate per-method, per-trial raw data
    method_trials: dict[str, list[TrialResult]] = {m: [] for m in config.methods}

    n_unlabeled = (
        int(config.n_labeled * config.n_unlabeled_ratio)
        if config.setting == "semi_supervised"
        else 0
    )

    for trial in range(config.n_trials):
        seed = (
            config.random_state + trial
            if config.random_state is not None
            else None
        )

        # Generate synthetic dataset
        dataset = generate_simulation(
            n_labeled=config.n_labeled,
            p=config.p,
            k=config.k,
            degree=config.degree,
            n_components=config.n_components,
            noise_std=config.noise_std,
            n_unlabeled=n_unlabeled,
            random_state=seed,
        )

        # Wrap in DataBundle for run_comparison
        bundle = DataLoader.from_arrays(
            dataset.X,
            dataset.y,
            X_unlabeled=dataset.X_unlabeled,
            source=config.label,
        )

        # Run the selected methods
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = run_comparison(
                    bundle,
                    true_base_indices=dataset.true_base_indices,
                    output_prefix=None,
                    degree=config.degree,
                    n_components=config.n_components,
                    Q=config.Q,
                    max_iter=config.max_iter,
                    random_state=seed,
                    methods=config.methods,
                    backend=config.backend,
                )
        except Exception as exc:  # noqa: BLE001 – KeyboardInterrupt/SystemExit are BaseException, not caught here
            print(
                f"  [WARN] Trial {trial} failed for config '{config.label}': {exc}"
            )
            continue

        for rb in results:
            # Compute derived discovery metrics
            tp = rb.n_true_positives if rb.n_true_positives is not None else None
            fp = rb.n_false_positives if rb.n_false_positives is not None else None
            fn = rb.n_false_negatives if rb.n_false_negatives is not None else None

            if tp is not None and fp is not None:
                precision = tp / max(1, tp + fp)
            else:
                precision = None

            recall = rb.tpr  # recall == TPR

            if precision is not None and recall is not None:
                denom = precision + recall
                f1 = 2 * precision * recall / denom if denom > 0 else 0.0
            else:
                f1 = None

            # Approximate AUC: 0.5 * (TPR + specificity)
            # specificity = TN / (TN + FP)
            # At the base-feature level: TN = p - n_true_base - FP,
            # where n_true_base = TP + FN (the true support size).
            if (rb.tpr is not None and tp is not None
                    and fp is not None and fn is not None):
                n_true_base = tp + fn
                tn = max(0, config.p - n_true_base - fp)
                specificity = tn / max(1, tn + fp)
                auc = 0.5 * (rb.tpr + specificity)
            else:
                auc = None

            method_trials[rb.method].append(
                TrialResult(
                    trial=trial,
                    method=rb.method,
                    fdr=rb.fdr,
                    tpr=rb.tpr,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    auc=auc,
                    n_selected=rb.n_selected,
                    n_true_positives=tp,
                    n_false_positives=fp,
                    n_false_negatives=fn,
                    r_squared=rb.r_squared if rb.r_squared is not None else float("nan"),
                    elapsed_seconds=rb.elapsed_seconds
                    if rb.elapsed_seconds is not None
                    else float("nan"),
                    peak_memory_mb=rb.peak_memory_mb
                    if rb.peak_memory_mb is not None
                    else float("nan"),
                )
            )

    # Aggregate per method
    sim_results: list[SimulationResult] = []
    for method, trials in method_trials.items():
        if not trials:
            continue
        fdrs = [t.fdr for t in trials if t.fdr is not None]
        tprs = [t.tpr for t in trials if t.tpr is not None]
        precisions = [t.precision for t in trials if t.precision is not None]
        recalls = [t.recall for t in trials if t.recall is not None]
        f1s = [t.f1 for t in trials if t.f1 is not None]
        aucs = [t.auc for t in trials if t.auc is not None]
        r2s = [t.r_squared for t in trials]
        n_sels = [t.n_selected for t in trials]
        n_tps = [t.n_true_positives for t in trials if t.n_true_positives is not None]
        n_fps = [t.n_false_positives for t in trials if t.n_false_positives is not None]
        n_fns = [t.n_false_negatives for t in trials if t.n_false_negatives is not None]
        elaps = [t.elapsed_seconds for t in trials]
        mems = [t.peak_memory_mb for t in trials]

        sim_results.append(
            SimulationResult(
                config=config,
                method=method,
                fdr_mean=float(np.mean(fdrs)) if fdrs else float("nan"),
                fdr_std=float(np.std(fdrs)) if len(fdrs) > 1 else float("nan"),
                tpr_mean=float(np.mean(tprs)) if tprs else float("nan"),
                tpr_std=float(np.std(tprs)) if len(tprs) > 1 else float("nan"),
                precision_mean=float(np.mean(precisions)) if precisions else float("nan"),
                precision_std=float(np.std(precisions)) if len(precisions) > 1 else float("nan"),
                recall_mean=float(np.mean(recalls)) if recalls else float("nan"),
                recall_std=float(np.std(recalls)) if len(recalls) > 1 else float("nan"),
                f1_mean=float(np.mean(f1s)) if f1s else float("nan"),
                f1_std=float(np.std(f1s)) if len(f1s) > 1 else float("nan"),
                auc_mean=float(np.mean(aucs)) if aucs else float("nan"),
                auc_std=float(np.std(aucs)) if len(aucs) > 1 else float("nan"),
                n_selected_mean=float(np.mean(n_sels)) if n_sels else float("nan"),
                n_true_positives_mean=float(np.mean(n_tps)) if n_tps else float("nan"),
                n_false_positives_mean=float(np.mean(n_fps)) if n_fps else float("nan"),
                n_false_negatives_mean=float(np.mean(n_fns)) if n_fns else float("nan"),
                r_squared_mean=float(np.nanmean(r2s)) if r2s else float("nan"),
                elapsed_mean=float(np.nanmean(elaps)) if elaps else float("nan"),
                peak_memory_mean=float(np.nanmean(mems)) if mems else float("nan"),
                n_completed=len(trials),
                trial_results=trials,
            )
        )

    return sim_results


def run_simulation_suite(
    configs: list[SimulationConfig],
    *,
    output_prefix: Optional[str] = None,
    verbose: bool = True,
) -> list[SimulationResult]:
    """Run a list of simulation configurations and optionally save results.

    Parameters
    ----------
    configs : list of SimulationConfig
        Configurations to sweep.
    output_prefix : str or None
        If provided, results are written to:

        * ``<output_prefix>_summary.json``
        * ``<output_prefix>_summary.csv``
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list of SimulationResult
        All results, one entry per (config, method) pair.
    """
    all_results: list[SimulationResult] = []
    for i, cfg in enumerate(configs):
        if verbose:
            print(
                f"[{i + 1}/{len(configs)}] {cfg.label}  "
                f"(p={cfg.p}, n={cfg.n_labeled}, k={cfg.k}, "
                f"degree={cfg.degree}, setting={cfg.setting}, trials={cfg.n_trials})",
                flush=True,
            )
        results = run_simulation(cfg)
        all_results.extend(results)
        if verbose:
            for r in results:
                fdr_s = f"{r.fdr_mean:.3f}±{r.fdr_std:.3f}" if not np.isnan(r.fdr_mean) else "n/a"
                tpr_s = f"{r.tpr_mean:.3f}±{r.tpr_std:.3f}" if not np.isnan(r.tpr_mean) else "n/a"
                f1_s = f"{r.f1_mean:.3f}±{r.f1_std:.3f}" if not np.isnan(r.f1_mean) else "n/a"
                print(
                    f"   {r.method:<25} FDR={fdr_s}  TPR={tpr_s}  F1={f1_s}  "
                    f"n_sel={r.n_selected_mean:.1f}  t={r.elapsed_mean:.1f}s "
                    f"({r.n_completed}/{cfg.n_trials} trials)"
                )

    if output_prefix is not None:
        _save_results(all_results, output_prefix)
        if verbose:
            print(f"\nResults saved to {output_prefix}_summary.json/.csv")

    return all_results


def default_configs(
    p_values: tuple[int, ...] = (5, 10, 20),
    n_values: tuple[int, ...] = (100, 300, 500),
    k_values: tuple[int, ...] = (2, 3),
    settings: tuple[str, ...] = ("supervised", "semi_supervised"),
    degree_values: tuple[int, ...] = (2,),
    methods: Optional[list[str]] = None,
    n_trials: int = 5,
    degree: int = 2,
    Q: float = 0.10,
    max_iter: int = 10,
    random_state: int = 0,
    backend: str = "rust",
) -> list[SimulationConfig]:
    """Build the default simulation sweep configurations.

    Creates one :class:`SimulationConfig` for every combination of
    ``p × n × k × setting × degree``.  Configurations where ``k >= 2·degree·p``
    (i.e. k equals or exceeds the total number of polynomial terms) are
    automatically skipped.

    Parameters
    ----------
    p_values : tuple of int
        Dimensions to sweep (default ``(5, 10, 20)``).
    n_values : tuple of int
        Sample sizes to sweep (default ``(100, 300, 500)``).
    k_values : tuple of int
        Sparsity levels — number of non-zero polynomial terms — to sweep
        (default ``(2, 3)``).
    settings : tuple of str
        Evaluation settings: ``"supervised"`` and/or
        ``"semi_supervised"`` (default: both).
    degree_values : tuple of int
        Polynomial degrees to sweep (default ``(2,)``).  When a single
        value is given, behaves identically to the ``degree`` parameter.
    methods : list of str or None
        Methods to evaluate.  Default: ``["ic_knock_poly"]``.
    n_trials : int
        Trials per configuration (default 5).
    degree : int
        Polynomial degree (default 2).  Ignored when ``degree_values``
        contains more than one value.
    Q : float
        Target FDR level (default 0.10).
    max_iter : int
        Maximum IC-Knock-Poly iterations (default 10).
    random_state : int
        Base random seed (default 0).
    backend : str
        Computational kernel for all methods (default ``"rust"``).

    Returns
    -------
    list of SimulationConfig
    """
    if methods is None:
        methods = ["ic_knock_poly"]

    # degree_values takes precedence; fall back to single degree
    if len(degree_values) == 1 and degree_values[0] == 2 and degree != 2:
        effective_degrees: tuple[int, ...] = (degree,)
    else:
        effective_degrees = degree_values

    configs: list[SimulationConfig] = []
    for deg in effective_degrees:
        for p in p_values:
            for n in n_values:
                for k in k_values:
                    if k >= 2 * deg * p:
                        continue
                    for setting in settings:
                        configs.append(
                            SimulationConfig(
                                p=p,
                                n_labeled=n,
                                k=k,
                                setting=setting,
                                degree=deg,
                                Q=Q,
                                n_trials=n_trials,
                                methods=list(methods),
                                max_iter=max_iter,
                                random_state=random_state,
                                backend=backend,
                            )
                        )
    return configs


def sweep_degree_nonzero_configs(
    degree_values: tuple[int, ...] = (2, 3),
    nonzero_values: tuple[int, ...] = (5, 10, 15, 20),
    p: int = 25,
    n_values: tuple[int, ...] = (100, 300, 500),
    settings: tuple[str, ...] = ("supervised",),
    methods: Optional[list[str]] = None,
    n_trials: int = 5,
    Q: float = 0.10,
    max_iter: int = 10,
    random_state: int = 0,
    backend: str = "rust",
) -> list[SimulationConfig]:
    """Build sweep configurations for polynomial degree and non-zero elements.

    Creates one :class:`SimulationConfig` for every combination of
    ``degree × k × n × setting``.  Intended for requirement 1 of the
    simulation study: varying the degree of polynomials (2, 3) and the
    number of non-zero polynomial terms (5, 10, 15, 20) while keeping *p*
    and other hyper-parameters fixed.

    Parameters
    ----------
    degree_values : tuple of int
        Polynomial degrees to sweep (default ``(2, 3)``).
    nonzero_values : tuple of int
        Sparsity levels *k* (non-zero polynomial terms) to sweep
        (default ``(5, 10, 15, 20)``).
    p : int
        Number of base features.
        Default 25.
    n_values : tuple of int
        Sample sizes to sweep (default ``(100, 300, 500)``).
    settings : tuple of str
        Evaluation settings (default ``("supervised",)``).
    methods : list of str or None
        Methods to evaluate.  Default: ``["ic_knock_poly"]``.
    n_trials : int
        Trials per configuration (default 5).
    Q : float
        Target FDR level (default 0.10).
    max_iter : int
        Maximum IC-Knock-Poly iterations (default 10).
    random_state : int
        Base random seed (default 0).
    backend : str
        Computational kernel for all methods (default ``"rust"``).

    Returns
    -------
    list of SimulationConfig
        One entry per ``degree × k × n × setting`` combination with
        ``k < 2·degree·p``.
    """
    if methods is None:
        methods = ["ic_knock_poly"]

    configs: list[SimulationConfig] = []
    for deg in degree_values:
        for k in nonzero_values:
            if k >= 2 * deg * p:
                continue
            for n in n_values:
                for setting in settings:
                    configs.append(
                        SimulationConfig(
                            p=p,
                            n_labeled=n,
                            k=k,
                            setting=setting,
                            degree=deg,
                            Q=Q,
                            n_trials=n_trials,
                            methods=list(methods),
                            max_iter=max_iter,
                            random_state=random_state,
                            backend=backend,
                        )
                    )
    return configs


def sweep_noise_configs(
    noise_values: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0),
    p: int = 5,
    n_labeled: int = 300,
    k: int = 2,
    degree: int = 2,
    settings: tuple[str, ...] = ("supervised",),
    methods: Optional[list[str]] = None,
    n_trials: int = 5,
    Q: float = 0.10,
    max_iter: int = 10,
    random_state: int = 0,
    backend: str = "rust",
) -> list[SimulationConfig]:
    """Build sweep configurations varying the label-noise standard deviation.

    Creates one :class:`SimulationConfig` for every combination of
    ``noise_std × setting`` at fixed ``p``, ``n_labeled``, ``k``, and
    ``degree``.  Intended for studying how prediction accuracy (R²),
    FDR, and recall degrade as the noise level increases.

    Parameters
    ----------
    noise_values : tuple of float
        Noise standard-deviation levels to sweep.
        Default ``(0.1, 0.25, 0.5, 1.0, 2.0)``.
    p : int
        Number of base features.  Default 5.
    n_labeled : int
        Number of labeled training samples.  Default 300.
    k : int
        Sparsity — non-zero polynomial terms in β*.  Default 2.
    degree : int
        Maximum polynomial exponent.  Default 2.
    settings : tuple of str
        Evaluation settings (default ``("supervised",)``).
    methods : list of str or None
        Methods to evaluate.  Default: ``["ic_knock_poly"]``.
    n_trials : int
        Trials per configuration (default 5).
    Q : float
        Target FDR level (default 0.10).
    max_iter : int
        Maximum IC-Knock-Poly iterations (default 10).
    random_state : int
        Base random seed (default 0).
    backend : str
        Computational kernel for all methods (default ``"rust"``).

    Returns
    -------
    list of SimulationConfig
        One entry per ``noise_std × setting`` combination.
    """
    if methods is None:
        methods = ["ic_knock_poly"]

    configs: list[SimulationConfig] = []
    for noise_std in noise_values:
        for setting in settings:
            configs.append(
                SimulationConfig(
                    p=p,
                    n_labeled=n_labeled,
                    k=k,
                    setting=setting,
                    degree=degree,
                    noise_std=noise_std,
                    Q=Q,
                    n_trials=n_trials,
                    methods=list(methods),
                    max_iter=max_iter,
                    random_state=random_state,
                    backend=backend,
                )
            )
    return configs


def print_summary_table(results: list[SimulationResult]) -> None:
    """Print a formatted summary table of simulation results.

    Parameters
    ----------
    results : list of SimulationResult
        As returned by ``run_simulation_suite``.
    """
    cols = ["label", "method", "fdr_mean", "tpr_mean", "precision_mean",
            "recall_mean", "f1_mean", "auc_mean",
            "n_selected_mean", "r_squared_mean", "elapsed_mean", "n_completed"]
    widths = {c: max(len(c), 10) for c in cols}
    widths["label"] = max((len(r.config.label) for r in results), default=10) + 2
    widths["method"] = max((len(r.method) for r in results), default=10) + 2

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        d = r.to_dict()
        def _fmt(k):
            v = d.get(k, "")
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        line = "  ".join(_fmt(c).ljust(widths[c]) for c in cols)
        print(line)
    print()


def _save_results(results: list[SimulationResult], prefix: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)

    # JSON (all per-trial detail)
    json_path = prefix + "_summary.json"
    all_dicts = []
    for r in results:
        d = r.to_dict()
        d["trials"] = [t.to_dict() for t in r.trial_results]
        all_dicts.append(d)
    with open(json_path, "w") as f:
        json.dump(all_dicts, f, indent=2)

    # CSV (summary only)
    csv_path = prefix + "_summary.csv"
    if results:
        rows = [r.to_csv_row() for r in results]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run IC-Knock-Poly simulation sweep with synthesised GMM data."
    )
    parser.add_argument(
        "--sweep",
        choices=["default", "degree_nonzero"],
        default="default",
        help=(
            "Sweep preset: 'default' sweeps p×n×k×setting×degree; "
            "'degree_nonzero' sweeps degree=[2,3] × k=[5,10,15,20] "
            "at fixed p=25. (default: default)"
        ),
    )
    parser.add_argument(
        "--p", nargs="+", type=int, default=[5, 10, 20], metavar="DIM",
        help="Feature dimensions to sweep (default: 5 10 20)",
    )
    parser.add_argument(
        "--n", nargs="+", type=int, default=[100, 300, 500], metavar="N",
        help="Sample sizes to sweep (default: 100 300 500)",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[2, 3], metavar="K",
        help="Sparsity levels to sweep (default: 2 3)",
    )
    parser.add_argument(
        "--settings", nargs="+", default=["supervised", "semi_supervised"],
        choices=["supervised", "semi_supervised"],
        help="Evaluation settings (default: supervised semi_supervised)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to run (default: ic_knock_poly)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=5, metavar="T",
        help="Trials per configuration (default: 5)",
    )
    parser.add_argument(
        "--degree", nargs="+", type=int, default=[2],
        help="Polynomial degree(s) to sweep (default: 2)",
    )
    parser.add_argument(
        "--nonzero", nargs="+", type=int, default=[5, 10, 15, 20], metavar="NZ",
        help=(
            "Non-zero element counts to sweep in 'degree_nonzero' preset "
            "(default: 5 10 15 20)"
        ),
    )
    parser.add_argument(
        "--Q", type=float, default=0.10,
        help="Target FDR level (default: 0.10)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=10,
        help="Max IC-Knock-Poly iterations (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)",
    )
    parser.add_argument(
        "--output", metavar="PREFIX", default=None,
        help="Output prefix for JSON/CSV files (default: no file output)",
    )
    parser.add_argument(
        "--backend",
        default="rust",
        choices=["python", "cpp", "rust"],
        help="Kernel backend (default: rust)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help=(
            "Generate visualisation plots after the sweep.  Requires "
            "matplotlib.  Saves figures to <output>_plots/ when --output "
            "is given, otherwise displays interactively."
        ),
    )
    args = parser.parse_args()

    if args.sweep == "degree_nonzero":
        configs = sweep_degree_nonzero_configs(
            degree_values=tuple(args.degree) if len(args.degree) > 1
            else (2, 3),
            nonzero_values=tuple(args.nonzero),
            n_values=tuple(args.n),
            settings=tuple(args.settings),
            methods=args.methods,
            n_trials=args.n_trials,
            Q=args.Q,
            max_iter=args.max_iter,
            random_state=args.seed,
            backend=args.backend,
        )
    else:
        configs = default_configs(
            p_values=tuple(args.p),
            n_values=tuple(args.n),
            k_values=tuple(args.k),
            settings=tuple(args.settings),
            degree_values=tuple(args.degree),
            methods=args.methods,
            n_trials=args.n_trials,
            Q=args.Q,
            max_iter=args.max_iter,
            random_state=args.seed,
            backend=args.backend,
        )

    print(f"Running {len(configs)} simulation configuration(s) …\n")
    results = run_simulation_suite(configs, output_prefix=args.output)
    print_summary_table(results)

    if args.plot:
        from .visualize import plot_all
        plot_dir = (args.output + "_plots") if args.output else None
        plot_all(results, output_dir=plot_dir)
        if plot_dir:
            print(f"\nPlots saved to {plot_dir}/")
        else:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    _cli()

"""Simulation sweep for IC-Knock-Poly.

Runs IC-Knock-Poly (and optionally all baselines) on synthesised datasets
that vary in:

* **Number of base features** *p* (dimensions).
* **Number of labeled samples** *n*.
* **Evaluation setting**: *supervised* (no unlabeled data) or
  *semi-supervised* (unlabeled pool of size ``n_unlabeled_ratio × n``).

Each configuration is repeated ``n_trials`` times with different random seeds
so that empirical FDR and TPR can be averaged.

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
        --n-trials 5 \\
        --output results/sweep \\
        --methods ic_knock_poly poly_lasso

Python API
----------
::

    from simulations.run_simulation import run_simulation_suite, default_configs
    results = run_simulation_suite(default_configs())
    for r in results:
        print(r.config, r.fdr_mean, r.tpr_mean)
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
        Sparsity — number of non-zero base features in β*.
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

    def __post_init__(self):
        if not self.label:
            self.label = (
                f"p{self.p}_n{self.n_labeled}_k{self.k}_{self.setting}"
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
    n_selected : int
        Number of selected polynomial terms.
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
    n_selected: int
    r_squared: float
    elapsed_seconds: float
    peak_memory_mb: float

    def to_dict(self) -> dict:
        return {
            "trial": self.trial,
            "method": self.method,
            "fdr": self.fdr,
            "tpr": self.tpr,
            "n_selected": self.n_selected,
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
    n_selected_mean : float
        Mean number of selected terms.
    r_squared_mean : float
        Mean R².
    elapsed_mean : float
        Mean wall-clock time per trial (seconds).
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
    n_selected_mean: float
    r_squared_mean: float
    elapsed_mean: float
    n_completed: int
    trial_results: list = field(default_factory=list)

    def to_dict(self) -> dict:
        cfg = self.config
        return {
            "label": cfg.label,
            "p": cfg.p,
            "n_labeled": cfg.n_labeled,
            "k": cfg.k,
            "setting": cfg.setting,
            "method": self.method,
            "fdr_mean": self.fdr_mean,
            "fdr_std": self.fdr_std,
            "tpr_mean": self.tpr_mean,
            "tpr_std": self.tpr_std,
            "n_selected_mean": self.n_selected_mean,
            "r_squared_mean": self.r_squared_mean,
            "elapsed_mean": self.elapsed_mean,
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
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [WARN] Trial {trial} failed for config '{config.label}': {exc}"
            )
            continue

        for rb in results:
            method_trials[rb.method].append(
                TrialResult(
                    trial=trial,
                    method=rb.method,
                    fdr=rb.fdr,
                    tpr=rb.tpr,
                    n_selected=rb.n_selected,
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
        r2s = [t.r_squared for t in trials]
        n_sels = [t.n_selected for t in trials]
        elaps = [t.elapsed_seconds for t in trials]

        sim_results.append(
            SimulationResult(
                config=config,
                method=method,
                fdr_mean=float(np.mean(fdrs)) if fdrs else float("nan"),
                fdr_std=float(np.std(fdrs)) if len(fdrs) > 1 else float("nan"),
                tpr_mean=float(np.mean(tprs)) if tprs else float("nan"),
                tpr_std=float(np.std(tprs)) if len(tprs) > 1 else float("nan"),
                n_selected_mean=float(np.mean(n_sels)) if n_sels else float("nan"),
                r_squared_mean=float(np.nanmean(r2s)) if r2s else float("nan"),
                elapsed_mean=float(np.nanmean(elaps)) if elaps else float("nan"),
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
                f"setting={cfg.setting}, trials={cfg.n_trials})",
                flush=True,
            )
        results = run_simulation(cfg)
        all_results.extend(results)
        if verbose:
            for r in results:
                fdr_s = f"{r.fdr_mean:.3f}±{r.fdr_std:.3f}" if not np.isnan(r.fdr_mean) else "n/a"
                tpr_s = f"{r.tpr_mean:.3f}±{r.tpr_std:.3f}" if not np.isnan(r.tpr_mean) else "n/a"
                print(
                    f"   {r.method:<25} FDR={fdr_s}  TPR={tpr_s}  "
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
    methods: Optional[list[str]] = None,
    n_trials: int = 5,
    degree: int = 2,
    Q: float = 0.10,
    max_iter: int = 10,
    random_state: int = 0,
) -> list[SimulationConfig]:
    """Build the default simulation sweep configurations.

    Creates one :class:`SimulationConfig` for every combination of
    ``p × n × k × setting``.  Configurations where ``k >= p`` are
    automatically skipped.

    Parameters
    ----------
    p_values : tuple of int
        Dimensions to sweep (default ``(5, 10, 20)``).
    n_values : tuple of int
        Sample sizes to sweep (default ``(100, 300, 500)``).
    k_values : tuple of int
        Sparsity levels to sweep (default ``(2, 3)``).
    settings : tuple of str
        Evaluation settings: ``"supervised"`` and/or
        ``"semi_supervised"`` (default: both).
    methods : list of str or None
        Methods to evaluate.  Default: ``["ic_knock_poly"]``.
    n_trials : int
        Trials per configuration (default 5).
    degree : int
        Polynomial degree (default 2).
    Q : float
        Target FDR level (default 0.10).
    max_iter : int
        Maximum IC-Knock-Poly iterations (default 10).
    random_state : int
        Base random seed (default 0).

    Returns
    -------
    list of SimulationConfig
    """
    if methods is None:
        methods = ["ic_knock_poly"]

    configs: list[SimulationConfig] = []
    for p in p_values:
        for n in n_values:
            for k in k_values:
                if k >= p:
                    continue
                for setting in settings:
                    configs.append(
                        SimulationConfig(
                            p=p,
                            n_labeled=n,
                            k=k,
                            setting=setting,
                            degree=degree,
                            Q=Q,
                            n_trials=n_trials,
                            methods=list(methods),
                            max_iter=max_iter,
                            random_state=random_state,
                        )
                    )
    return configs


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary_table(results: list[SimulationResult]) -> None:
    """Print a formatted summary table of simulation results.

    Parameters
    ----------
    results : list of SimulationResult
        As returned by ``run_simulation_suite``.
    """
    cols = ["label", "method", "fdr_mean", "fdr_std", "tpr_mean", "tpr_std",
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
        "--degree", type=int, default=2,
        help="Polynomial degree (default: 2)",
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
    args = parser.parse_args()

    configs = default_configs(
        p_values=tuple(args.p),
        n_values=tuple(args.n),
        k_values=tuple(args.k),
        settings=tuple(args.settings),
        methods=args.methods,
        n_trials=args.n_trials,
        degree=args.degree,
        Q=args.Q,
        max_iter=args.max_iter,
        random_state=args.seed,
    )

    print(f"Running {len(configs)} simulation configuration(s) …\n")
    results = run_simulation_suite(configs, output_prefix=args.output)
    print_summary_table(results)


if __name__ == "__main__":
    _cli()

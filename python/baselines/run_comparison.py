"""Comparison runner for IC-Knock-Poly and all baseline methods.

Loads a dataset (CSV / NPZ / NPY / in-memory) and runs all six methods on the
same data.  Each method produces a ``ResultBundle``; the results are written to:

  * ``<output_prefix>_results.json``  — one JSON object per method
  * ``<output_prefix>_results.csv``   — one row per method (flat summary)

Usage (command line)::

    python run_comparison.py --data experiment.csv --output results/exp1

Usage (Python API)::

    from baselines.run_comparison import run_comparison
    from baselines.data_loader import DataLoader

    bundle = DataLoader.from_csv("experiment.csv")
    results = run_comparison(
        bundle,
        true_base_indices={0, 2},   # optional ground truth
        output_prefix="results/exp1",
        degree=2,
        Q=0.10,
        random_state=42,
    )
    for r in results:
        print(r.method, r.n_selected, r.r_squared)

The script always writes full JSON and a summary table.  Pass
``output_prefix=None`` to suppress file output (results are returned only).
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
from typing import Optional

import numpy as np

# Make the package importable whether run directly or as a module
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from ic_knockoff_poly_reg import ICKnockoffPolyReg
from ic_knockoff_poly_reg.evaluation import ResultBundle

from .data_loader import DataBundle, DataLoader
from .poly_lasso import PolyLasso
from .poly_omp import PolyOMP
from .poly_clime import PolyCLIME
from .poly_knockoff import PolyKnockoff
from .sparse_poly_stlsq import SparsePolySTLSQ


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------

def run_comparison(
    data: DataBundle,
    *,
    true_base_indices: Optional[set] = None,
    output_prefix: Optional[str] = None,
    degree: int = 2,
    n_components: int = 2,
    Q: float = 0.10,
    spending_sequence: str = "riemann_zeta",
    max_iter: int = 20,
    random_state: Optional[int] = 42,
    methods: Optional[list[str]] = None,
) -> list[ResultBundle]:
    """Run IC-Knock-Poly and all baselines on the same dataset.

    Parameters
    ----------
    data : DataBundle
        Dataset loaded via ``DataLoader`` (or ``DataLoader.from_arrays``).
    true_base_indices : set of int or None
        Ground-truth non-zero base feature indices.  When provided,
        FDR and TPR are computed for every method.
    output_prefix : str or None
        File path prefix for output files.  The runner writes:

        * ``<output_prefix>_results.json``
        * ``<output_prefix>_results.csv``

        Pass ``None`` to suppress file output.
    degree : int
        Polynomial degree (shared by all methods).  Default 2.
    n_components : int
        Number of GMM components for IC-Knock-Poly.  Default 2.
    Q : float
        Target FDR level (shared by FDR-controlling methods).  Default 0.10.
    spending_sequence : str
        PoSI spending sequence for IC-Knock-Poly.  Default ``"riemann_zeta"``.
    max_iter : int
        Maximum iterations for IC-Knock-Poly.  Default 20.
    random_state : int or None
        Shared random seed.  Default 42.
    methods : list of str or None
        Subset of method names to run.  Available:
        ``["ic_knock_poly", "poly_lasso", "poly_omp",
           "poly_clime", "poly_knockoff", "sparse_poly_stlsq"]``.
        Default ``None`` runs all.

    Returns
    -------
    list of ResultBundle
        One bundle per method, in the order they were run.
    """
    all_methods = [
        "ic_knock_poly",
        "poly_lasso",
        "poly_omp",
        "poly_clime",
        "poly_knockoff",
        "sparse_poly_stlsq",
    ]
    if methods is None:
        methods = all_methods
    else:
        unknown = set(methods) - set(all_methods)
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}. Available: {all_methods}")

    X, y = data.X, data.y
    X_unlabeled = data.X_unlabeled
    dataset_name = data.source

    results: list[ResultBundle] = []

    # ------------------------------------------------------------------
    # IC-Knock-Poly (main method)
    # ------------------------------------------------------------------
    total = len(methods)
    step = 0

    if "ic_knock_poly" in methods:
        step += 1
        print(f"[{step}/{total}] IC-Knock-Poly ...", flush=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ICKnockoffPolyReg(
                degree=degree,
                n_components=n_components,
                Q=Q,
                spending_sequence=spending_sequence,
                max_iter=max_iter,
                random_state=random_state,
            )
            model.fit(X, y, X_unlabeled=X_unlabeled)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 ** 2)
        rb = model.to_result_bundle(
            X, y,
            dataset=dataset_name,
            true_base_indices=true_base_indices,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak_mb,
        )
        results.append(rb)
        print(f"       selected={rb.n_selected}, R²={rb.r_squared:.3f}, t={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Poly + Lasso
    # ------------------------------------------------------------------
    if "poly_lasso" in methods:
        step += 1
        print(f"[{step}/{total}] PolyLasso ...", flush=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl = PolyLasso(degree=degree, random_state=random_state)
            bl.fit(X, y)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rb = bl.to_result_bundle(
            X, y,
            dataset=dataset_name,
            true_base_indices=true_base_indices,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak / (1024 ** 2),
        )
        results.append(rb)
        print(f"       selected={rb.n_selected}, R²={rb.r_squared:.3f}, t={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Poly + OMP
    # ------------------------------------------------------------------
    if "poly_omp" in methods:
        step += 1
        print(f"[{step}/{total}] PolyOMP ...", flush=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl = PolyOMP(degree=degree)
            bl.fit(X, y)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rb = bl.to_result_bundle(
            X, y,
            dataset=dataset_name,
            true_base_indices=true_base_indices,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak / (1024 ** 2),
        )
        results.append(rb)
        print(f"       selected={rb.n_selected}, R²={rb.r_squared:.3f}, t={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Poly + CLIME knockoff filter
    # ------------------------------------------------------------------
    if "poly_clime" in methods:
        step += 1
        print(f"[{step}/{total}] PolyCLIME ...", flush=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl = PolyCLIME(degree=degree, Q=Q, random_state=random_state)
            bl.fit(X, y)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rb = bl.to_result_bundle(
            X, y,
            dataset=dataset_name,
            true_base_indices=true_base_indices,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak / (1024 ** 2),
        )
        results.append(rb)
        print(f"       selected={rb.n_selected}, R²={rb.r_squared:.3f}, t={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Poly + standard knockoff filter
    # ------------------------------------------------------------------
    if "poly_knockoff" in methods:
        step += 1
        print(f"[{step}/{total}] PolyKnockoff ...", flush=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl = PolyKnockoff(degree=degree, Q=Q, random_state=random_state)
            bl.fit(X, y)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rb = bl.to_result_bundle(
            X, y,
            dataset=dataset_name,
            true_base_indices=true_base_indices,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak / (1024 ** 2),
        )
        results.append(rb)
        print(f"       selected={rb.n_selected}, R²={rb.r_squared:.3f}, t={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Sparse poly STLSQ
    # ------------------------------------------------------------------
    if "sparse_poly_stlsq" in methods:
        step += 1
        print(f"[{step}/{total}] SparsePolySTLSQ ...", flush=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl = SparsePolySTLSQ(degree=degree)
            bl.fit(X, y)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rb = bl.to_result_bundle(
            X, y,
            dataset=dataset_name,
            true_base_indices=true_base_indices,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak / (1024 ** 2),
        )
        results.append(rb)
        print(f"       selected={rb.n_selected}, R²={rb.r_squared:.3f}, t={elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    if output_prefix is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_prefix)) or ".", exist_ok=True)
        _write_json(results, output_prefix + "_results.json")
        _write_csv(results, output_prefix + "_results.csv")
        print(f"\nResults written to {output_prefix}_results.json/.csv")

    return results


def print_table(results: list[ResultBundle]) -> None:
    """Print a summary table to stdout.

    Parameters
    ----------
    results : list of ResultBundle
        As returned by ``run_comparison``.
    """
    cols = ["method", "n_selected", "r_squared", "bic", "fdr", "tpr",
            "elapsed_seconds"]
    widths = {c: max(len(c), 14) for c in cols}
    widths["method"] = max(len(r.method) for r in results) + 2

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = r.to_csv_row()
        line = "  ".join(
            str(row.get(c, "")).ljust(widths[c]) for c in cols
        )
        print(line)
    print()


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write_json(results: list[ResultBundle], path: str) -> None:
    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_csv(results: list[ResultBundle], path: str) -> None:
    if not results:
        return
    rows = [r.to_csv_row() for r in results]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run IC-Knock-Poly and baseline methods on a dataset."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", metavar="FILE", help="CSV file (last column = response)")
    src.add_argument("--npz", metavar="FILE", help="NPZ file with keys X, y, [X_unlabeled]")
    src.add_argument(
        "--npy",
        nargs=2,
        metavar=("X_FILE", "Y_FILE"),
        help="Two NPY files for X and y",
    )
    parser.add_argument("--unlabeled-csv", metavar="FILE",
                        help="CSV of unlabeled X features (semi-supervised)")
    parser.add_argument("--output", metavar="PREFIX", default="comparison",
                        help="Output prefix (default: comparison)")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--Q", type=float, default=0.10)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run (default: all)",
    )
    args = parser.parse_args()

    if args.csv:
        bundle = DataLoader.from_csv(args.csv, unlabeled_path=args.unlabeled_csv)
    elif args.npz:
        bundle = DataLoader.from_npz(args.npz)
    else:
        bundle = DataLoader.from_npy(args.npy[0], args.npy[1])

    results = run_comparison(
        bundle,
        output_prefix=args.output,
        degree=args.degree,
        n_components=args.n_components,
        Q=args.Q,
        max_iter=args.max_iter,
        random_state=args.seed,
        methods=args.methods,
    )
    print_table(results)


if __name__ == "__main__":
    _cli()

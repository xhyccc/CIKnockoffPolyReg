# IC-Knock-Poly Python Package

Python implementation of the **IC-Knock-Poly** algorithm: *Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression*.

## Overview

IC-Knock-Poly discovers sparse rational polynomial equations (e.g. `y = x₁² + 1/x₂`) in ultra-high dimensional settings (`p ≫ n`) while strictly controlling the **False Discovery Rate (FDR)** via a PoSI α-spending sequence.

### Algorithm Phases

| Phase | Description |
|-------|-------------|
| **Phase 1** | Fit a penalised GMM with GraphLasso precision matrices to learn `P(X)` |
| **Phase 2** | Initialise empty active sets; residuals `R₀ = Y` |
| **Phase 3** | Iteratively generate conditional knockoffs, expand features via `Φ(·)`, compute `W_j` statistics, apply PoSI threshold `τ_t`, update active sets |

### FDR Control (PoSI α-Spending)

Two spending sequences are supported:

- **Riemann Zeta** (default): `q_t = Q · 6 / (π² t²)` — general-purpose, mathematically sound.
- **Geometric**: `q_t = Q · (1−γ) · γ^(t−1)` — better power for very sparse models.

Both satisfy `Σ q_t ≤ Q`, guaranteeing global FDR ≤ Q.

## Installation

```bash
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Data Interface

### Input format

| Argument | Type | Shape | Notes |
|---|---|---|---|
| `X` | `np.ndarray` | `(n_labeled, p)` | Labeled feature matrix, `float64`. Each row is one observation; each column one feature. |
| `y` | `np.ndarray` | `(n_labeled,)` | Continuous response vector, `float64`. |
| `X_unlabeled` *(optional)* | `np.ndarray` | `(N_unlabeled, p)` | Additional unlabeled feature observations for semi-supervised mode. Must have the same `p` columns as `X`. |

**Accepted sources for `X` and `y`:** NumPy arrays, pandas DataFrames/Series
(call `.to_numpy()` first or pass directly — `np.asarray` is applied
internally), and nested Python lists.

### Feature requirements

- **All numeric** — no categorical columns (encode them before passing).
- **No `NaN` or `Inf`** — impute or drop missing values before fitting.
- **Non-constant** — constant columns produce degenerate knockoffs; drop them.
- **Bounded away from zero** for columns that may appear in rational (inverse)
  terms: values `|x| < 1e-8` are automatically clipped, but very large
  fractions of near-zero values indicate the feature should be transformed
  (e.g. `log(x)`) before being passed.

### Loading data from CSV

```python
import numpy as np
import pandas as pd

# Load from CSV (features in columns 0..p-1, response in last column)
df = pd.read_csv("my_data.csv")
X = df.iloc[:, :-1].to_numpy(dtype=float)  # (n, p)
y = df.iloc[:, -1].to_numpy(dtype=float)   # (n,)
```

## Quick Start

### Supervised mode

```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg

# Generate synthetic data: y = x0 + 1/x1 + noise
rng = np.random.default_rng(42)
n, p = 200, 20
X = rng.uniform(0.5, 3.0, size=(n, p))
y = X[:, 0] + 1.0 / X[:, 1] + 0.1 * rng.standard_normal(n)

# Fit IC-Knock-Poly with FDR target Q=0.10
model = ICKnockoffPolyReg(
    degree=2,
    n_components=2,
    Q=0.10,
    spending_sequence="riemann_zeta",
    random_state=42,
)
model.fit(X, y)

print("Selected features:", model.result_.selected_poly_names)
print("Base features used:", model.result_.selected_base_indices)
print("Iterations run:", model.result_.n_iterations)
```

### Semi-supervised mode

Use this when you have **many unlabeled observations** of the covariates but
only a **small labeled subset** with response values.  Phase 1 learns the
joint feature distribution `P(X)` from all data; Phases 2–3 run only on the
labeled pairs.

```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg

rng = np.random.default_rng(0)
p = 15

# Unlabeled pool: 2000 observations, no y
X_unlabeled = rng.uniform(0.5, 3.0, size=(2000, p))

# Labeled subset: only 80 observations
n_labeled = 80
X_labeled = rng.uniform(0.5, 3.0, size=(n_labeled, p))
y_labeled = X_labeled[:, 0] + 1.0 / X_labeled[:, 2] + 0.1 * rng.standard_normal(n_labeled)

model = ICKnockoffPolyReg(
    degree=2,
    n_components=3,   # more components → richer distribution model
    Q=0.10,
    spending_sequence="riemann_zeta",
    random_state=0,
)

# Phase 1 uses X_unlabeled + X_labeled (2080 rows)
# Phases 2–3 use only (X_labeled, y_labeled) (80 rows)
model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)

print("Selected features:", model.result_.selected_poly_names)
print("Iterations run:", model.result_.n_iterations)

# Predict on new data
X_test = rng.uniform(0.5, 3.0, size=(50, p))
y_pred = model.predict(X_test)
```

### Reading results

After calling `fit`, inspect `model.result_`:

```python
r = model.result_

# Human-readable names of the selected polynomial features
print(r.selected_poly_names)   # e.g. ['x0', 'x2^(-1)']

# Indices of base features that appear in any selected term
print(r.selected_base_indices) # e.g. {0, 2}

# (base_feature_index, exponent) for each selected term
print(r.selected_terms)        # e.g. [(0, 1), (2, -1)]

# Regression coefficients for the selected polynomial features
print(r.coef)

# Number of outer iterations performed
print(r.n_iterations)

# Per-iteration diagnostics
for step in r.iteration_history:
    print(step)  # keys: iteration, q_t, tau_t, n_candidates, n_new_selected, residual_norm
```

## Module Structure

| Module | Description |
|--------|-------------|
| `gmm_phase` | `PenalizedGMM`: EM + GraphLasso precision estimation |
| `knockoffs` | `ConditionalKnockoffGenerator`: equicorrelated Gaussian knockoffs |
| `polynomial` | `PolynomialDictionary`: rational polynomial dictionary `Φ(·)` |
| `posi_threshold` | `AlphaSpending`, `compute_knockoff_threshold`: PoSI FDR control |
| `algorithm` | `ICKnockoffPolyReg`: main iterative pipeline |
| `evaluation` | `compute_fdr`, `compute_tpr`, `memory_tracker`: metrics |

## Output Format

After calling `fit`, all methods return a ``ResultBundle`` — a unified
dataclass that captures selections, statistical properties, and compute cost.
It is serialisable to JSON and CSV for cross-language comparison.

### ResultBundle fields

| Field | Type | Description |
|---|---|---|
| `method` | str | Method identifier, e.g. `"ic_knock_poly"` |
| `selected_names` | list[str] | Human-readable polynomial feature names |
| `selected_base_indices` | list[int] | Base feature indices in selected terms |
| `selected_terms` | list[[int, int]] | `[base_idx, exponent]` pairs |
| `coef` | list[float] | Regression coefficients |
| `intercept` | float | Fitted intercept |
| `n_selected` | int | Number of selected terms |
| `r_squared` | float | R² on training data |
| `adj_r_squared` | float | Adjusted R² |
| `residual_ss` | float | Residual sum of squares |
| `bic` | float | Bayesian Information Criterion |
| `aic` | float | Akaike Information Criterion |
| `elapsed_seconds` | float | Wall-clock fit time |
| `peak_memory_mb` | float | Peak memory usage |
| `fdr` | float or None | Empirical FDR (needs ground truth) |
| `tpr` | float or None | True positive rate (needs ground truth) |
| `params` | dict | Method-specific hyper-parameters |
| `extra` | dict | Method-specific diagnostics |

### Generating a ResultBundle from IC-Knock-Poly

```python
import time, tracemalloc, json
from ic_knockoff_poly_reg import ICKnockoffPolyReg

model = ICKnockoffPolyReg(degree=2, Q=0.10, random_state=42)

tracemalloc.start()
t0 = time.perf_counter()
model.fit(X, y)
elapsed = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

rb = model.to_result_bundle(
    X, y,
    dataset="experiment.csv",
    true_base_indices={0, 2},         # optional ground truth
    elapsed_seconds=elapsed,
    peak_memory_mb=peak / 1024**2,
)

# JSON output
print(rb.to_json())

# Flat CSV row
print(rb.to_csv_row())

# Round-trip from JSON
from ic_knockoff_poly_reg import ResultBundle
rb2 = ResultBundle.from_dict(json.loads(rb.to_json()))
```

### JSON schema

```json
{
  "method": "ic_knock_poly",
  "dataset": "experiment.csv",
  "timestamp": "2024-01-01T00:00:00Z",
  "selected_names": ["x0", "x2^(-1)"],
  "selected_base_indices": [0, 2],
  "selected_terms": [[0, 1], [2, -1]],
  "coef": [1.02, 0.98],
  "intercept": 0.05,
  "n_selected": 2,
  "fit": {
    "r_squared": 0.982,
    "adj_r_squared": 0.981,
    "residual_ss": 0.12,
    "total_ss": 6.67,
    "bic": -452.1,
    "aic": -457.3
  },
  "discovery": {
    "fdr": 0.0,
    "tpr": 1.0,
    "n_true_positives": 2,
    "n_false_positives": 0,
    "n_false_negatives": 0
  },
  "compute": {
    "elapsed_seconds": 3.2,
    "peak_memory_mb": 48.1
  },
  "params": { "degree": 2, "Q": 0.10, "n_components": 2 },
  "extra": { "n_iterations": 3 }
}
```

## Baseline Methods

`python/baselines/` contains five comparison methods that use the **same
polynomial dictionary** Φ(·) as IC-Knock-Poly:

| Class | Method | FDR control |
|---|---|---|
| `PolyLasso` | Lasso + CV alpha selection | ✗ |
| `PolyOMP` | Orthogonal Matching Pursuit + CV | ✗ |
| `PolyCLIME` | CLIME precision matrix + one-shot knockoff filter | ✓ |
| `PolyKnockoff` | Standard Gaussian knockoffs + Lasso | ✓ |
| `SparsePolySTLSQ` | Sequential Thresholded Least Squares (SINDy-style) | ✗ |

Each baseline exposes `fit(X, y)`, `predict(X)`, and `to_result_bundle(...)`.

### Running all methods on one dataset

```bash
# From CSV (last column = response)
python -m baselines.run_comparison --csv data/experiment.csv \
       --output results/exp1 --degree 2 --Q 0.10 --seed 42

# Specify a subset of methods
python -m baselines.run_comparison --csv data/experiment.csv \
       --methods ic_knock_poly poly_lasso poly_omp

# From NPZ archive
python -m baselines.run_comparison --npz data/experiment.npz \
       --output results/exp1
```

Or from the Python API:

```python
from baselines.data_loader import DataLoader
from baselines.run_comparison import run_comparison, print_table

bundle = DataLoader.from_csv("data/experiment.csv")

# Optionally supply unlabeled data for IC-Knock-Poly's semi-supervised mode
bundle_semi = DataLoader.from_csv(
    "data/experiment.csv",
    unlabeled_path="data/unlabeled.csv",
)

results = run_comparison(
    bundle,
    true_base_indices={0, 2},       # optional ground truth
    output_prefix="results/exp1",   # write JSON + CSV
    degree=2,
    Q=0.10,
    random_state=42,
)
print_table(results)
```

The runner writes:
- `results/exp1_results.json` — list of JSON objects, one per method
- `results/exp1_results.csv`  — flat table with one row per method

### Using individual baselines

```python
from baselines import PolyLasso, PolyOMP, SparsePolySTLSQ
from baselines.data_loader import DataLoader

bundle = DataLoader.from_csv("data/experiment.csv")
X, y = bundle.X, bundle.y

for Cls in [PolyLasso, PolyOMP, SparsePolySTLSQ]:
    model = Cls(degree=2)
    model.fit(X, y)
    rb = model.to_result_bundle(X, y, dataset="experiment.csv")
    print(rb.method, rb.n_selected, rb.r_squared)
```



```bash
cd python
pytest tests/ -v
```

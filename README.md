# CIKnockoffPolyReg

**IC-Knock-Poly**: Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression.

Identifies sparse rational polynomial equations (e.g. `y = x₁² + 1/x₂`) in ultra-high dimensional settings (`p ≫ n`) while strictly controlling the **False Discovery Rate (FDR)** via a PoSI α-spending sequence.

## Repository Structure

| Directory | Language | Description |
|-----------|----------|-------------|
| `python/` | Python 3.10+ | Full IC-Knock-Poly algorithm (GMM + knockoffs + iterative pipeline) |
| `cpp/`    | C++17    | Core computational routines (matrix ops, polynomial expansion, FDR threshold) |
| `rust/`   | Rust 2021 | Statistical utilities (polynomial expansion, knockoff W-stats, PoSI threshold) |

## Kernel Backends

IC-Knock-Poly exposes a **unified kernel interface** so callers can opt into the
compiled C++ or Rust implementations for the three hot sub-tasks:

| Sub-task | Python symbol | Description |
|---|---|---|
| Polynomial expansion | `PolynomialKernel.expand` | Rational dictionary Φ(X) |
| Knockoff W-statistics | `KnockoffKernel.w_statistics` | W_j = |β_j| − |β̃_j| |
| PoSI threshold | `PosiKernel.alpha_spending_budget` / `knockoff_threshold` | FDR budget q_t, threshold τ_t |

Three backend implementations are available:

| Backend | Availability | Notes |
|---------|-------------|-------|
| `"python"` | Always (default) | Pure Python / NumPy; no compilation required |
| `"cpp"` | After building the C++ shared library | C++17 loaded via `ctypes` |
| `"rust"` | After building the Rust cdylib | Rust 2021 loaded via `ctypes` |

### Building the native backends

**C++ backend** (produces `cpp/build/libic_knockoff_py.so`):

```bash
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target ic_knockoff_py
```

**Rust backend** (produces `rust/target/release/libic_knockoff_poly_reg.so`):

```bash
cd rust && cargo build --release
```

Override default library search paths via environment variables:

```bash
export LIBIC_KNOCKOFF_CPP_PATH=/path/to/libic_knockoff_py.so
export LIBIC_KNOCKOFF_RUST_PATH=/path/to/libic_knockoff_poly_reg.so
```

### Discovering available backends

```python
from ic_knockoff_poly_reg import AVAILABLE_BACKENDS

print(AVAILABLE_BACKENDS)  # frozenset({'python', 'cpp', 'rust'})
```

`AVAILABLE_BACKENDS` is a `frozenset` probed at import time; it contains only
the backends whose shared libraries can actually be loaded.

### Using kernels directly

```python
import numpy as np
from ic_knockoff_poly_reg import create_kernels

# Select a backend
poly, knock, posi = create_kernels("cpp")   # or "python" / "rust"

# Polynomial expansion
X = np.random.uniform(0.5, 3.0, size=(100, 5))
ef = poly.expand(X, degree=2, include_bias=True)

# Knockoff W-statistics
beta_orig  = np.array([1.0, 0.0, -0.5, 0.0, 0.3])
beta_knock = np.array([0.1, 0.0, -0.1, 0.0, 0.3])
W = knock.w_statistics(beta_orig, beta_knock)

# PoSI alpha-spending budget and threshold
qt  = posi.alpha_spending_budget(t=1, Q=0.1, sequence="riemann_zeta")
tau = posi.knockoff_threshold(W, q_t=qt)
```

### Selecting a backend for the main estimator

Pass `backend=` to `ICKnockoffPolyReg`; the polynomial expansion, W-statistics,
and PoSI threshold in Phase 3 are all routed through the chosen kernel:

```python
from ic_knockoff_poly_reg import ICKnockoffPolyReg

model = ICKnockoffPolyReg(degree=2, Q=0.10, backend="rust")
model.fit(X, y)
```

## Data Interface

### What your dataset must look like

IC-Knock-Poly expects **numeric, tabular** data.

| Variable | Shape | Description |
|---|---|---|
| `X` | `(n, p)` float matrix | Base feature matrix. Rows = observations; columns = features. |
| `y` | `(n,)` float vector | Continuous response variable, aligned row-by-row with `X`. |
| `X_unlabeled` *(optional)* | `(N, p)` float matrix | Extra unlabeled feature observations for semi-supervised mode (same `p` columns as `X`). |

**Requirements for `X`:**
- All entries must be **finite** (no `NaN` or `Inf`).
- Columns should be **non-constant** (constant columns produce degenerate knockoffs).
- Columns that may appear in rational terms (denominator) should be **bounded away from zero**; values `|x| < 1e-8` are automatically clipped.
- No prior scaling is required, but extreme skew may warrant a log-transform.

### How to supply data

```python
import numpy as np
import pandas as pd

# --- From a CSV file ---
df = pd.read_csv("experiment.csv")
X = df.drop(columns=["response"]).to_numpy(dtype=float)  # (n, p)
y = df["response"].to_numpy(dtype=float)                  # (n,)

# --- From NumPy arrays directly ---
X = np.load("features.npy")   # (n, p)
y = np.load("labels.npy")     # (n,)

# --- From nested Python lists ---
X = np.array([[1.2, 0.8], [2.1, 1.3], [0.9, 2.2]])
y = np.array([3.4, 4.7, 2.8])
```

## Algorithm Overview

### Phase 1 — Base Distribution Learning
Fit a penalised GMM with GraphLasso (L1 penalty λ) precision matrices to learn P(X).

> **Semi-supervised**: when `X_unlabeled` is provided this phase uses
> `np.vstack([X_unlabeled, X])` — the full pool of observations — so that
> the joint distribution is estimated from all available data.

### Phase 2 — Initialisation
Empty active sets A_base = A_poly = ∅; residuals R₀ = Y.

### Phase 3 — Iterative Expansion and Screening
For each iteration t:
1. **Conditional knockoffs** for unselected base features (using GMM precision blocks).
2. **Polynomial expansion** via Φ(·) = (·, 1/·, 1)^d on unselected features + knockoffs.
3. **Cross-validated Lasso** on [Φ(X_B), Φ(X̃_B)] predicting residuals R_{t-1}.
4. **W-statistics**: W_j = |β̂_j| - |β̂_j̃|.
5. **PoSI threshold** τ_t using α-spending budget q_t.
6. Update active sets; refit residuals. **Stop** if no new features selected.

### FDR Control (PoSI α-Spending)
| Sequence | Formula | Use case |
|----------|---------|----------|
| Riemann Zeta | q_t = Q · 6/(π²t²) | General (unknown iterations) |
| Geometric | q_t = Q · (1−γ) · γ^(t−1) | Very sparse models |

Both satisfy Σ q_t ≤ Q, guaranteeing global FDR ≤ Q.

## Output Format

Every method — IC-Knock-Poly as well as all baselines — produces a
**`ResultBundle`** that is serialisable to JSON and CSV.

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
  "params": {
    "degree": 2,
    "Q": 0.10
  }
}
```

The `discovery` block is populated only when ground-truth feature indices are
supplied (`true_base_indices=`).  C++ and Rust methods omit `discovery` and
`timestamp` (they output the same `fit` / `compute` / `params` blocks).

### Accessing results in Python

```python
# JSON string
print(model.to_result_bundle(X, y, dataset="exp.csv").to_json())

# Flat dict for CSV
row = model.to_result_bundle(X, y).to_csv_row()

# Round-trip from JSON
from ic_knockoff_poly_reg import ResultBundle
rb = ResultBundle.from_dict(json.loads(json_str))
```

## Baseline Methods

Each language folder contains a `baselines/` directory with the following
implementations that use the **same polynomial dictionary** as IC-Knock-Poly
but without the iterative GMM + PoSI α-spending machinery:

| Method | Description |
|---|---|
| **PolyLasso** | Polynomial dictionary + cross-validated Lasso (no FDR control) |
| **PolyOMP** | Polynomial dictionary + Orthogonal Matching Pursuit |
| **PolyCLIME** | CLIME-based sparse precision matrix + one-shot knockoff filter (Python only) |
| **PolyKnockoff** | Standard Gaussian knockoff filter on polynomial features (Python only) |
| **SparsePolySTLSQ** | Sequential Thresholded Least Squares (SINDy-style) |

All baselines expose `fit(X, y)` / `predict(X)` and `to_result_bundle(...)`.

## Running Comparisons

### Python — all methods on one dataset

```bash
# From CSV (last column = response)
python -m baselines.run_comparison --csv data/experiment.csv --output results/exp1 \
       --degree 2 --Q 0.10 --seed 42

# From NPZ (keys: X, y, [X_unlabeled])
python -m baselines.run_comparison --npz data/experiment.npz --output results/exp1
```

Or via the Python API:

```python
from baselines.data_loader import DataLoader
from baselines.run_comparison import run_comparison, print_table

bundle = DataLoader.from_csv("data/experiment.csv")
results = run_comparison(bundle, true_base_indices={0, 2}, output_prefix="results/exp1")
print_table(results)
```

### C++ — PolyLasso, PolyOMP, PolySTLSQ

```bash
cd cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./baseline_runner ../../data/experiment.csv 2 ../../results/exp1_cpp
```

### Rust — PolyLasso, PolyOMP, PolySTLSQ

```bash
cd rust
cargo run --bin baseline_runner -- ../data/experiment.csv 2 ../results/exp1_rust
```

All three runners write `_results.json` and `_results.csv` using the same
field names, so results can be concatenated and compared directly.

## Usage Examples

### Python — basic (default Python backend)

```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg

rng = np.random.default_rng(42)
n, p = 200, 20
X = rng.uniform(0.5, 3.0, size=(n, p))
y = X[:, 0] + 1.0 / X[:, 1] + 0.1 * rng.standard_normal(n)

model = ICKnockoffPolyReg(degree=2, n_components=2, Q=0.10, random_state=42)
model.fit(X, y)
print("Selected:", model.result_.selected_poly_names)
```

### Python — using a compiled kernel backend

After building the native library (see [Building the native backends](#building-the-native-backends)):

```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg, AVAILABLE_BACKENDS

print(AVAILABLE_BACKENDS)  # e.g. frozenset({'python', 'cpp', 'rust'})

rng = np.random.default_rng(42)
n, p = 200, 20
X = rng.uniform(0.5, 3.0, size=(n, p))
y = X[:, 0] + 1.0 / X[:, 1] + 0.1 * rng.standard_normal(n)

# Route Phase 3 kernels through the Rust shared library
model = ICKnockoffPolyReg(degree=2, n_components=2, Q=0.10, random_state=42, backend="rust")
model.fit(X, y)
print("Selected:", model.result_.selected_poly_names)
```

### Python — semi-supervised (large X pool, few labeled pairs)
```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg

rng = np.random.default_rng(0)
p = 15

# 2000 unlabeled observations — used only for distribution learning
X_unlabeled = rng.uniform(0.5, 3.0, size=(2000, p))

# 80 labeled observations — used for regression
X_labeled = rng.uniform(0.5, 3.0, size=(80, p))
y_labeled  = X_labeled[:, 0] + 1.0 / X_labeled[:, 2] + 0.1 * rng.standard_normal(80)

model = ICKnockoffPolyReg(degree=2, n_components=3, Q=0.10, random_state=0)
# Phase 1 sees all 2080 rows; Phases 2-3 use only the 80 labeled rows
model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)
print("Selected:", model.result_.selected_poly_names)
```

### Python — install & test
```bash
cd python
pip install -e ".[dev]"
pytest tests/ -v
```

### C++
```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest --output-on-failure
```

### Rust
```bash
cd rust
cargo test
cargo run   # runs the demo
```

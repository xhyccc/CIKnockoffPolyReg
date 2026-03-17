# CIKnockoffPolyReg

**IC-Knock-Poly**: Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression.

Identifies sparse rational polynomial equations (e.g. `y = x₁² + 1/x₂`) in ultra-high dimensional settings (`p ≫ n`) while strictly controlling the **False Discovery Rate (FDR)** via a PoSI α-spending sequence.

## Repository Structure

| Directory | Language | Description |
|-----------|----------|-------------|
| `python/` | Python 3.10+ | Full IC-Knock-Poly algorithm (GMM + knockoffs + iterative pipeline) |
| `cpp/`    | C++17    | Core computational routines (matrix ops, polynomial expansion, FDR threshold) |
| `rust/`   | Rust 2021 | Statistical utilities (polynomial expansion, knockoff W-stats, PoSI threshold) |

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

## Quick Start

### Python — supervised
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

### Python — semi-supervised (large X pool, few labeled pairs)
```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg

rng = np.random.default_rng(0)
p = 15

# 2 000 unlabeled observations — used only for distribution learning
X_unlabeled = rng.uniform(0.5, 3.0, size=(2000, p))

# 80 labeled observations — used for regression
X_labeled = rng.uniform(0.5, 3.0, size=(80, p))
y_labeled  = X_labeled[:, 0] + 1.0 / X_labeled[:, 2] + 0.1 * rng.standard_normal(80)

model = ICKnockoffPolyReg(degree=2, n_components=3, Q=0.10, random_state=0)
# Phase 1 sees all 2 080 rows; Phases 2-3 use only the 80 labeled rows
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

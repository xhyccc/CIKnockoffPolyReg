# CIKnockoffPolyReg

**IC-Knock-Poly**: Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression.

Discovers sparse rational polynomial equations (e.g. `y = x₁² + 1/x₂`) while controlling the False Discovery Rate (FDR).

## Installation

```bash
cd python
pip install -e ".[dev]"
```

## Running Experiments

IC-Knock-Poly supports three computational **kernel backends** (`"python"`, `"cpp"`, `"rust"`) and two **learning settings** (supervised and semi-supervised).

### Step 1 — Build a native backend (optional)

Skip this step to use the default pure-Python backend.

**C++ backend:**
```bash
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target ic_knockoff_py
```

**Rust backend:**
```bash
cd rust && cargo build --release
```

### Step 2a — Supervised learning (labeled data only)

Pass `--backend` to choose the kernel (`python` / `cpp` / `rust`).

```bash
cd python

# default Python kernel
python -m baselines.run_comparison --csv data/experiment.csv \
       --output results/supervised_py --backend python

# C++ kernel
python -m baselines.run_comparison --csv data/experiment.csv \
       --output results/supervised_cpp --backend cpp

# Rust kernel
python -m baselines.run_comparison --csv data/experiment.csv \
       --output results/supervised_rust --backend rust
```

### Step 2b — Semi-supervised learning (unlabeled pool + labeled subset)

Provide an extra CSV of unlabeled feature observations via `--unlabeled-csv`.
IC-Knock-Poly uses these for distribution learning (Phase 1); all baselines
ignore them and run in supervised mode.

```bash
cd python

# default Python kernel, semi-supervised
python -m baselines.run_comparison \
       --csv data/labeled.csv \
       --unlabeled-csv data/unlabeled.csv \
       --output results/semisup_py --backend python

# C++ kernel, semi-supervised
python -m baselines.run_comparison \
       --csv data/labeled.csv \
       --unlabeled-csv data/unlabeled.csv \
       --output results/semisup_cpp --backend cpp

# Rust kernel, semi-supervised
python -m baselines.run_comparison \
       --csv data/labeled.csv \
       --unlabeled-csv data/unlabeled.csv \
       --output results/semisup_rust --backend rust
```

Each run writes `<output>_results.json` and `<output>_results.csv`.

### Step 2c — Baseline methods only (supervised, no kernel switching)

Baseline methods (`poly_lasso`, `poly_omp`, `poly_clime`, `poly_knockoff`,
`sparse_poly_stlsq`) always run in supervised mode and always use the Python
kernel regardless of `--backend`.

```bash
cd python

python -m baselines.run_comparison --csv data/experiment.csv \
       --output results/baselines \
       --methods poly_lasso poly_omp poly_clime poly_knockoff sparse_poly_stlsq
```

## Python API

### Supervised with switchable kernel

```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg

rng = np.random.default_rng(42)
n, p = 200, 20
X = rng.uniform(0.5, 3.0, size=(n, p))
y = X[:, 0] + 1.0 / X[:, 1] + 0.1 * rng.standard_normal(n)

# choose backend: "python" (default), "cpp", or "rust"
model = ICKnockoffPolyReg(degree=2, Q=0.10, backend="rust", random_state=42)
model.fit(X, y)
print("Selected:", model.result_.selected_poly_names)
```

### Semi-supervised with switchable kernel

```python
import numpy as np
from ic_knockoff_poly_reg import ICKnockoffPolyReg

rng = np.random.default_rng(0)
p = 15
X_unlabeled = rng.uniform(0.5, 3.0, size=(2000, p))  # no labels needed
X = rng.uniform(0.5, 3.0, size=(80, p))
y = X[:, 0] + 1.0 / X[:, 2] + 0.1 * rng.standard_normal(80)

# Phase 1 uses all 2080 rows; Phases 2-3 use only the 80 labeled rows
model = ICKnockoffPolyReg(degree=2, Q=0.10, backend="cpp", random_state=0)
model.fit(X, y, X_unlabeled=X_unlabeled)
print("Selected:", model.result_.selected_poly_names)
```

### Baselines (supervised only)

```python
from baselines import PolyLasso, PolyOMP, PolyCLIME, PolyKnockoff, SparsePolySTLSQ

for Cls in [PolyLasso, PolyOMP, PolyCLIME, PolyKnockoff, SparsePolySTLSQ]:
    model = Cls(degree=2)
    model.fit(X, y)
    rb = model.to_result_bundle(X, y)
    print(rb.method, rb.n_selected, rb.r_squared)
```

## Tests

```bash
cd python
pytest tests/ -v
```

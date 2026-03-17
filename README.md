# CIKnockoffPolyReg

**IC-Knock-Poly**: Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression.

Identifies sparse rational polynomial equations (e.g. `y = x₁² + 1/x₂`) in ultra-high dimensional settings (`p ≫ n`) while strictly controlling the **False Discovery Rate (FDR)** via a PoSI α-spending sequence.

## Repository Structure

| Directory | Language | Description |
|-----------|----------|-------------|
| `python/` | Python 3.10+ | Full IC-Knock-Poly algorithm (GMM + knockoffs + iterative pipeline) |
| `cpp/`    | C++17    | Core computational routines (matrix ops, polynomial expansion, FDR threshold) |
| `rust/`   | Rust 2021 | Statistical utilities (polynomial expansion, knockoff W-stats, PoSI threshold) |

## Algorithm Overview

### Phase 1 — Base Distribution Learning
Fit a penalised GMM with GraphLasso (L1 penalty λ) precision matrices to learn P(X).

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

### Python
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

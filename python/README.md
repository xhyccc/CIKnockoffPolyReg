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

## Quick Start

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

## Module Structure

| Module | Description |
|--------|-------------|
| `gmm_phase` | `PenalizedGMM`: EM + GraphLasso precision estimation |
| `knockoffs` | `ConditionalKnockoffGenerator`: equicorrelated Gaussian knockoffs |
| `polynomial` | `PolynomialDictionary`: rational polynomial dictionary `Φ(·)` |
| `posi_threshold` | `AlphaSpending`, `compute_knockoff_threshold`: PoSI FDR control |
| `algorithm` | `ICKnockoffPolyReg`: main iterative pipeline |
| `evaluation` | `compute_fdr`, `compute_tpr`, `memory_tracker`: metrics |

## Running Tests

```bash
cd python
pytest tests/ -v
```

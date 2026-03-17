# IC-Knock-Poly Rust Library

Rust implementation of core statistical routines for the **IC-Knock-Poly** algorithm.

## Modules

| Module | Description |
|--------|-------------|
| `matrix` | Dense matrix arithmetic, Cholesky decomposition, inversion, statistics |
| `polynomial` | Rational polynomial dictionary expansion Φ(·) |
| `knockoffs` | Knockoff W-statistic computation and Gaussian knockoff sampling |
| `posi` | PoSI α-spending sequences (Riemann Zeta & Geometric) and knockoff+ threshold |

## Building

```bash
cargo build --release
```

## Running the Demo

```bash
cargo run
```

## Running Tests

```bash
cargo test
```

## Requirements

- Rust ≥ 1.65 (2021 edition)
- No external dependencies (uses only the standard library)

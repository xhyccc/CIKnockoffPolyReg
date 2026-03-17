# IC-Knock-Poly C++ Library

C++ implementation of core computational routines for the **IC-Knock-Poly** algorithm.

## Modules

| Header | Description |
|--------|-------------|
| `matrix_ops.hpp` | Dense matrix arithmetic, Cholesky decomposition, inversion, block extraction |
| `polynomial.hpp` | Rational polynomial dictionary expansion Φ(X) |
| `knockoffs.hpp` | Knockoff W-statistic computation, Gaussian knockoff sampling |
| `posi.hpp` | PoSI α-spending sequences (Riemann Zeta & Geometric) and knockoff+ threshold |

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Running Tests

```bash
cd build
ctest --output-on-failure
# or directly:
./test_ic_knockoff
```

## Requirements

- CMake ≥ 3.16
- C++17 compliant compiler (GCC ≥ 8, Clang ≥ 7, MSVC ≥ 19.14)
- No external dependencies (standard library only)

"""C++ backend for the IC-Knock-Poly computational kernels.

Loads ``libic_knockoff_py.so`` (Linux/macOS) or ``ic_knockoff_py.dll``
(Windows) via :mod:`ctypes` and wraps the C-ABI functions in the
:class:`~.base.PolynomialKernel`, :class:`~.base.KnockoffKernel`, and
:class:`~.base.PosiKernel` interfaces.

Build the shared library first (from the repository root)::

    cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
    cmake --build cpp/build

The compiled library is searched for in the following order:

1. The path given by the environment variable ``LIBIC_KNOCKOFF_CPP_PATH``.
2. The default CMake build output directory ``<repo>/cpp/build/``.
3. Standard system library search paths (via :func:`ctypes.util.find_library`).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..polynomial import ExpandedFeatures
from .base import KnockoffKernel, PosiKernel, PolynomialKernel

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_LIB_NAMES = {
    "linux": "libic_knockoff_py.so",
    "darwin": "libic_knockoff_py.dylib",
    "win32": "ic_knockoff_py.dll",
}

_REPO_ROOT = Path(__file__).resolve().parents[4]  # …/CIKnockoffPolyReg
_DEFAULT_BUILD_DIR = _REPO_ROOT / "cpp" / "build"


def _find_cpp_library() -> ctypes.CDLL:
    """Locate and load the C++ shared library, raising RuntimeError if absent."""
    # 1. Environment variable override
    env_path = os.environ.get("LIBIC_KNOCKOFF_CPP_PATH")
    if env_path:
        candidates = [Path(env_path)]
    else:
        import platform
        sys = platform.system().lower()
        lib_name = _LIB_NAMES.get(sys, "libic_knockoff_py.so")
        candidates = [
            _DEFAULT_BUILD_DIR / lib_name,
        ]
        # 2. Sibling directories that cmake might use
        for subdir in ("Release", "Debug"):
            candidates.append(_DEFAULT_BUILD_DIR / subdir / lib_name)

    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))

    # 3. System library search
    found = ctypes.util.find_library("ic_knockoff_py")
    if found:
        return ctypes.CDLL(found)

    raise RuntimeError(
        "C++ shared library 'libic_knockoff_py' not found.\n"
        "Build it with:\n"
        "    cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release\n"
        "    cmake --build cpp/build\n"
        "Or set the environment variable LIBIC_KNOCKOFF_CPP_PATH to the "
        "full path of the compiled library file."
    )


# ---------------------------------------------------------------------------
# Type annotations for ctypes signatures
# ---------------------------------------------------------------------------

def _configure_lib(lib: ctypes.CDLL) -> None:
    """Attach argtypes / restype to all used C functions."""
    dbl_p = ctypes.POINTER(ctypes.c_double)
    int_p = ctypes.POINTER(ctypes.c_int)

    lib.ic_poly_n_expanded.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.ic_poly_n_expanded.restype = ctypes.c_int

    lib.ic_poly_expand.argtypes = [
        dbl_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double,
        dbl_p, int_p, int_p,
    ]
    lib.ic_poly_expand.restype = ctypes.c_int

    lib.ic_compute_w_statistics.argtypes = [dbl_p, dbl_p, ctypes.c_int, dbl_p]
    lib.ic_compute_w_statistics.restype = None

    lib.ic_equicorrelated_s_values.argtypes = [dbl_p, ctypes.c_int, ctypes.c_double, dbl_p]
    lib.ic_equicorrelated_s_values.restype = None

    lib.ic_sample_gaussian_knockoffs.argtypes = [
        dbl_p, ctypes.c_int, ctypes.c_int,
        dbl_p, dbl_p, ctypes.c_uint, dbl_p,
    ]
    lib.ic_sample_gaussian_knockoffs.restype = None

    lib.ic_alpha_spending_budget.argtypes = [
        ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
    ]
    lib.ic_alpha_spending_budget.restype = ctypes.c_double

    lib.ic_knockoff_threshold.argtypes = [
        dbl_p, ctypes.c_int, int_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
    ]
    lib.ic_knockoff_threshold.restype = ctypes.c_double


# ---------------------------------------------------------------------------
# Lazy-loaded library singleton
# ---------------------------------------------------------------------------

_cpp_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _cpp_lib
    if _cpp_lib is None:
        _cpp_lib = _find_cpp_library()
        _configure_lib(_cpp_lib)
    return _cpp_lib


# ---------------------------------------------------------------------------
# Helper: numpy array → ctypes pointer
# ---------------------------------------------------------------------------

def _dbl_ptr(arr: NDArray[np.float64]) -> ctypes.POINTER(ctypes.c_double):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _int_ptr(arr: NDArray[np.int32]) -> ctypes.POINTER(ctypes.c_int):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


# ---------------------------------------------------------------------------
# Kernel implementations
# ---------------------------------------------------------------------------

class CppPolynomialKernel(PolynomialKernel):
    """Polynomial expansion backed by the C++ kernel."""

    def n_expanded(self, n_base: int, degree: int, include_bias: bool) -> int:
        return _get_lib().ic_poly_n_expanded(n_base, degree, int(include_bias))

    def expand(
        self,
        X: NDArray[np.float64],
        degree: int,
        include_bias: bool,
        clip_threshold: float = 1e-8,
        base_names: Optional[list[str]] = None,
    ) -> ExpandedFeatures:
        lib = _get_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        n, p = X.shape
        n_cols = lib.ic_poly_n_expanded(p, degree, int(include_bias))

        out_matrix = np.zeros(n * n_cols, dtype=np.float64)
        out_bi = np.zeros(n_cols, dtype=np.int32)
        out_ex = np.zeros(n_cols, dtype=np.int32)

        ret = lib.ic_poly_expand(
            _dbl_ptr(X), n, p, degree, int(include_bias), clip_threshold,
            _dbl_ptr(out_matrix), _int_ptr(out_bi), _int_ptr(out_ex),
        )
        if ret < 0:
            raise RuntimeError("ic_poly_expand returned an error")

        matrix = out_matrix.reshape(n, n_cols)
        base_indices = out_bi.tolist()
        exponents = out_ex.tolist()

        # Build feature names from base_names + (base_idx, exponent)
        if base_names is None:
            base_names = [f"x{j}" for j in range(p)]
        names = _build_feature_names(base_names, base_indices, exponents)

        return ExpandedFeatures(
            matrix=matrix,
            feature_names=names,
            base_feature_indices=base_indices,
            power_exponents=exponents,
            interaction_indices=[None] * len(base_indices),
        )


class CppKnockoffKernel(KnockoffKernel):
    """Knockoff utilities backed by the C++ kernel."""

    def w_statistics(
        self,
        beta_original: NDArray[np.float64],
        beta_knockoff: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        lib = _get_lib()
        b_orig = np.ascontiguousarray(beta_original, dtype=np.float64)
        b_knock = np.ascontiguousarray(beta_knockoff, dtype=np.float64)
        p = len(b_orig)
        if p != len(b_knock):
            raise ValueError("beta_original and beta_knockoff must have the same length")
        out_w = np.zeros(p, dtype=np.float64)
        lib.ic_compute_w_statistics(_dbl_ptr(b_orig), _dbl_ptr(b_knock), p, _dbl_ptr(out_w))
        return out_w

    def equicorrelated_s_values(
        self,
        cov: NDArray[np.float64],
        reg: float = 1e-10,
    ) -> NDArray[np.float64]:
        lib = _get_lib()
        cov = np.ascontiguousarray(cov, dtype=np.float64)
        p = cov.shape[0]
        out_s = np.zeros(p, dtype=np.float64)
        lib.ic_equicorrelated_s_values(_dbl_ptr(cov), p, reg, _dbl_ptr(out_s))
        return out_s

    def sample_gaussian_knockoffs(
        self,
        X: NDArray[np.float64],
        mu: NDArray[np.float64],
        Sigma: NDArray[np.float64],
        seed: int = 42,
    ) -> NDArray[np.float64]:
        lib = _get_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        mu = np.ascontiguousarray(mu, dtype=np.float64)
        Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
        n, p = X.shape
        out = np.zeros(n * p, dtype=np.float64)
        lib.ic_sample_gaussian_knockoffs(
            _dbl_ptr(X), n, p, _dbl_ptr(mu), _dbl_ptr(Sigma),
            ctypes.c_uint(seed), _dbl_ptr(out),
        )
        return out.reshape(n, p)


class CppPosiKernel(PosiKernel):
    """PoSI budget and threshold backed by the C++ kernel."""

    _SEQ_CODES = {"riemann_zeta": 0, "geometric": 1}

    def alpha_spending_budget(
        self,
        t: int,
        Q: float,
        sequence: str,
        gamma: float = 0.5,
    ) -> float:
        lib = _get_lib()
        seq_code = self._SEQ_CODES.get(sequence, 0)
        return float(lib.ic_alpha_spending_budget(t, Q, seq_code, gamma))

    def knockoff_threshold(
        self,
        W: NDArray[np.float64],
        q_t: float,
        active_poly: Optional[set[int]] = None,
        offset: int = 1,
    ) -> float:
        lib = _get_lib()
        W = np.ascontiguousarray(W, dtype=np.float64)
        p = len(W)

        if active_poly:
            act_arr = np.array(sorted(active_poly), dtype=np.int32)
            n_active = len(act_arr)
            act_ptr = _int_ptr(act_arr)
        else:
            act_arr = np.zeros(0, dtype=np.int32)
            n_active = 0
            act_ptr = _int_ptr(act_arr)

        return float(lib.ic_knockoff_threshold(
            _dbl_ptr(W), p, act_ptr, n_active, q_t, offset,
        ))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _build_feature_names(
    base_names: list[str],
    base_indices: list[int],
    exponents: list[int],
    interaction_indices: list = None,
    interaction_exponents: list = None,
) -> list[str]:
    """Reconstruct feature names from base names + (base_index, exponent) pairs.
    
    For interaction terms (base_index == -2), uses interaction_indices and 
    interaction_exponents to build the proper name like "x0^2*x1^(-1)".
    """
    names = []
    for i, (bi, ex) in enumerate(zip(base_indices, exponents)):
        if bi == -1:
            # Bias column: base_feature_index == -1, exponent == 0
            names.append("1")
        elif bi == -2:
            # Interaction term: need to use interaction_indices and exponents
            if (interaction_indices is not None and i < len(interaction_indices) and
                interaction_exponents is not None and i < len(interaction_exponents)):
                inter = interaction_indices[i]
                inter_exp = interaction_exponents[i]
                if inter is not None and inter_exp is not None and len(inter) >= 2 and len(inter_exp) >= 2:
                    # Build interaction name with exponents
                    parts = []
                    for idx, exp in zip(inter, inter_exp):
                        if 0 <= idx < len(base_names):
                            base_name = base_names[idx]
                        else:
                            base_name = f"x{idx}"
                        
                        # Format with exponent
                        if exp == 1:
                            parts.append(base_name)
                        elif exp < 0:
                            parts.append(f"{base_name}^({exp})")
                        else:
                            parts.append(f"{base_name}^{exp}")
                    names.append("*".join(parts))
                else:
                    names.append(f"inter_{i}")
            else:
                names.append(f"inter_{i}")
        elif ex == 1:
            names.append(base_names[bi])
        elif ex > 1:
            names.append(f"{base_names[bi]}^{ex}")
        else:
            names.append(f"{base_names[bi]}^({ex})")
    return names

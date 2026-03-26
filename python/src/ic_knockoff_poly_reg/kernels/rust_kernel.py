"""Rust backend for the IC-Knock-Poly computational kernels.

Loads the cdylib built by Cargo via :mod:`ctypes` and wraps the C-ABI
functions in the :class:`~.base.PolynomialKernel`, :class:`~.base.KnockoffKernel`,
and :class:`~.base.PosiKernel` interfaces.

Build the shared library first (from the repository root)::

    cd rust && cargo build --release

The compiled library is searched for in the following order:

1. The path given by the environment variable ``LIBIC_KNOCKOFF_RUST_PATH``.
2. The default Cargo release output directory ``<repo>/rust/target/release/``.
3. Standard system library search paths (via :func:`ctypes.util.find_library`).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..polynomial import ExpandedFeatures
from .base import KnockoffKernel, PosiKernel, PolynomialKernel
from .cpp_kernel import _build_feature_names

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_LIB_NAMES = {
    "linux": "libic_knockoff_poly_reg.so",
    "darwin": "libic_knockoff_poly_reg.dylib",
    "win32": "ic_knockoff_poly_reg.dll",
}

_REPO_ROOT = Path(__file__).resolve().parents[4]  # …/CIKnockoffPolyReg
_DEFAULT_RELEASE_DIR = _REPO_ROOT / "rust" / "target" / "release"


def _find_rust_library() -> ctypes.CDLL:
    """Locate and load the Rust cdylib, raising RuntimeError if absent."""
    env_path = os.environ.get("LIBIC_KNOCKOFF_RUST_PATH")
    if env_path:
        candidates = [Path(env_path)]
    else:
        import platform
        sys = platform.system().lower()
        lib_name = _LIB_NAMES.get(sys, "libic_knockoff_poly_reg.so")
        candidates = [_DEFAULT_RELEASE_DIR / lib_name]

    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))

    found = ctypes.util.find_library("ic_knockoff_poly_reg")
    if found:
        return ctypes.CDLL(found)

    raise RuntimeError(
        "Rust shared library 'libic_knockoff_poly_reg' not found.\n"
        "Build it with:\n"
        "    cd rust && cargo build --release\n"
        "Or set the environment variable LIBIC_KNOCKOFF_RUST_PATH to the "
        "full path of the compiled library file."
    )


def _configure_lib(lib: ctypes.CDLL) -> None:
    """Attach argtypes / restype to all used C functions."""
    dbl_p = ctypes.POINTER(ctypes.c_double)
    int_p = ctypes.POINTER(ctypes.c_int)

    lib.ic_poly_n_expanded.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.ic_poly_n_expanded.restype = ctypes.c_int

    lib.ic_poly_expand.argtypes = [
        dbl_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        dbl_p, int_p, int_p, int_p, int_p, int_p, int_p,
    ]
    lib.ic_poly_expand.restype = ctypes.c_int

    lib.ic_compute_w_statistics.argtypes = [dbl_p, dbl_p, ctypes.c_int, dbl_p]
    lib.ic_compute_w_statistics.restype = None

    lib.ic_equicorrelated_s_values.argtypes = [dbl_p, ctypes.c_int, ctypes.c_double, dbl_p]
    lib.ic_equicorrelated_s_values.restype = None

    lib.ic_sample_gaussian_knockoffs.argtypes = [
        dbl_p, ctypes.c_int, ctypes.c_int,
        dbl_p, dbl_p, ctypes.c_uint64, dbl_p,
    ]
    lib.ic_sample_gaussian_knockoffs.restype = ctypes.c_int

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

_rust_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _rust_lib
    if _rust_lib is None:
        _rust_lib = _find_rust_library()
        _configure_lib(_rust_lib)
    return _rust_lib


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

class RustPolynomialKernel(PolynomialKernel):
    """Polynomial expansion backed by the Rust kernel."""

    def n_expanded(self, n_base: int, degree: int, include_bias: bool, include_interactions: bool = True) -> int:
        return _get_lib().ic_poly_n_expanded(n_base, degree, int(include_bias), int(include_interactions))

    def expand(
        self,
        X: NDArray[np.float64],
        degree: int,
        include_bias: bool,
        include_interactions: bool = True,
        clip_threshold: float = 1e-8,
        base_names: Optional[list[str]] = None,
    ) -> ExpandedFeatures:
        lib = _get_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        n, p = X.shape
        n_cols = lib.ic_poly_n_expanded(p, degree, int(include_bias), int(include_interactions))

        out_matrix = np.zeros(n * n_cols, dtype=np.float64)
        out_bi = np.zeros(n_cols, dtype=np.int32)
        out_ex = np.zeros(n_cols, dtype=np.int32)
        out_i1 = np.zeros(n_cols, dtype=np.int32)
        out_i2 = np.zeros(n_cols, dtype=np.int32)
        out_e1 = np.zeros(n_cols, dtype=np.int32)
        out_e2 = np.zeros(n_cols, dtype=np.int32)

        ret = lib.ic_poly_expand(
            _dbl_ptr(X), n, p, degree, int(include_bias), int(include_interactions), clip_threshold,
            _dbl_ptr(out_matrix), _int_ptr(out_bi), _int_ptr(out_ex), 
            _int_ptr(out_i1), _int_ptr(out_i2), _int_ptr(out_e1), _int_ptr(out_e2),
        )
        if ret < 0:
            raise RuntimeError("ic_poly_expand returned an error")

        matrix = out_matrix.reshape(n, n_cols)
        base_indices = out_bi.tolist()
        exponents = out_ex.tolist()
        i1_list = out_i1.tolist()
        i2_list = out_i2.tolist()
        e1_list = out_e1.tolist()
        e2_list = out_e2.tolist()

        # Build interaction_indices and interaction_exponents lists
        interaction_indices = []
        interaction_exponents = []
        for idx, (i1, i2, e1, e2) in enumerate(zip(i1_list, i2_list, e1_list, e2_list)):
            if i1 >= 0 and i2 >= 0:
                interaction_indices.append([int(i1), int(i2)])
                interaction_exponents.append([int(e1), int(e2)])
            else:
                interaction_indices.append(None)
                interaction_exponents.append(None)

        if base_names is None:
            base_names = [f"x{j}" for j in range(p)]
        names = _build_feature_names(base_names, base_indices, exponents, interaction_indices, interaction_exponents)

        # Store interaction_exponents in ExpandedFeatures for use by algorithm
        # We need to add this field to ExpandedFeatures or store it differently
        result = ExpandedFeatures(
            matrix=matrix,
            feature_names=names,
            base_feature_indices=base_indices,
            power_exponents=exponents,
            interaction_indices=interaction_indices,
        )
        # Attach interaction_exponents as an extra attribute
        result.interaction_exponents = interaction_exponents
        return result


class RustKnockoffKernel(KnockoffKernel):
    """Knockoff utilities backed by the Rust kernel."""

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
        ret = lib.ic_sample_gaussian_knockoffs(
            _dbl_ptr(X), n, p, _dbl_ptr(mu), _dbl_ptr(Sigma),
            ctypes.c_uint64(seed), _dbl_ptr(out),
        )
        if ret < 0:
            raise RuntimeError("ic_sample_gaussian_knockoffs failed (non-SPD covariance?)")
        return out.reshape(n, p)


class RustPosiKernel(PosiKernel):
    """PoSI budget and threshold backed by the Rust kernel."""

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

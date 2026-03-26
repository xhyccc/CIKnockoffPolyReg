"""Rust-backed polynomial regression baselines.

Thin Python wrappers around the Rust implementations of PolyLasso, PolyOMP,
and SparsePolySTLSQ.  Each class exposes the same interface as its pure-Python
counterpart in :mod:`poly_lasso`, :mod:`poly_omp`, and
:mod:`sparse_poly_stlsq` so they can be used as drop-in replacements.

The Rust shared library is loaded via :mod:`ctypes`; build it first::

    cd rust && cargo build --release

The compiled library is searched for in the following order:

1. The path given by ``LIBIC_KNOCKOFF_RUST_PATH`` environment variable.
2. The default Cargo release output ``<repo>/rust/target/release/``.
3. Standard system library search paths.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Make ic_knockoff_poly_reg importable
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from ic_knockoff_poly_reg.evaluation import ResultBundle

# ---------------------------------------------------------------------------
# Library loading (re-use the singleton from rust_kernel if already loaded)
# ---------------------------------------------------------------------------

_LIB_NAMES = {
    "linux": "libic_knockoff_poly_reg.so",
    "darwin": "libic_knockoff_poly_reg.dylib",
    "win32": "ic_knockoff_poly_reg.dll",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RELEASE_DIR = _REPO_ROOT / "rust" / "target" / "release"

_rust_lib: Optional[ctypes.CDLL] = None


def _find_and_load_lib() -> ctypes.CDLL:
    """Locate and load the Rust cdylib."""
    env_path = os.environ.get("LIBIC_KNOCKOFF_RUST_PATH")
    if env_path:
        candidates = [Path(env_path)]
    else:
        import platform
        plat = platform.system().lower()
        lib_name = _LIB_NAMES.get(plat, "libic_knockoff_poly_reg.so")
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
        "Or set LIBIC_KNOCKOFF_RUST_PATH to the full path of the compiled library."
    )


def _configure_lib(lib: ctypes.CDLL) -> None:
    """Attach argtypes / restype for baseline FFI functions."""
    dbl_p = ctypes.POINTER(ctypes.c_double)
    int_p = ctypes.POINTER(ctypes.c_int)

    # ic_poly_n_expanded(n_base, degree, include_bias) -> int
    lib.ic_poly_n_expanded.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.ic_poly_n_expanded.restype = ctypes.c_int

    # ic_baseline_poly_lasso_fit
    lib.ic_baseline_poly_lasso_fit.argtypes = [
        dbl_p, dbl_p,              # x_flat, y_flat
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, p, degree
        dbl_p, dbl_p, int_p, int_p,  # out_coef, out_intercept, out_base_indices, out_exponents
    ]
    lib.ic_baseline_poly_lasso_fit.restype = ctypes.c_int

    # ic_baseline_poly_omp_fit
    lib.ic_baseline_poly_omp_fit.argtypes = [
        dbl_p, dbl_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, p, degree, max_nonzero
        dbl_p, dbl_p, int_p, int_p,
    ]
    lib.ic_baseline_poly_omp_fit.restype = ctypes.c_int

    # ic_baseline_poly_stlsq_fit
    lib.ic_baseline_poly_stlsq_fit.argtypes = [
        dbl_p, dbl_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,  # n, p, degree, threshold
        dbl_p, dbl_p, int_p, int_p,
    ]
    lib.ic_baseline_poly_stlsq_fit.restype = ctypes.c_int


def _get_lib() -> ctypes.CDLL:
    global _rust_lib
    if _rust_lib is None:
        _rust_lib = _find_and_load_lib()
        _configure_lib(_rust_lib)
    return _rust_lib


# ---------------------------------------------------------------------------
# ctypes helpers
# ---------------------------------------------------------------------------

def _dbl_ptr(arr: NDArray[np.float64]) -> ctypes.POINTER(ctypes.c_double):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _int_ptr(arr: NDArray[np.int32]) -> ctypes.POINTER(ctypes.c_int):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


# ---------------------------------------------------------------------------
# Shared result-building logic
# ---------------------------------------------------------------------------

def _build_result_bundle(
    method: str,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
    coef_all: NDArray[np.float64],
    intercept: float,
    base_indices: NDArray[np.int32],
    exponents: NDArray[np.int32],
    dataset: str,
    true_poly_terms: Optional[list],
    elapsed_seconds: float,
    peak_memory_mb: float,
) -> ResultBundle:
    """Construct a :class:`ResultBundle` from the raw Rust output."""
    n_cols = len(coef_all)

    # Predict: Phi(X) @ coef + intercept
    # Re-expand X using Python polynomial expansion (same degree)
    from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
    poly_dict = PolynomialDictionary(degree=degree, include_bias=True)
    Phi = poly_dict.expand(X).matrix
    y_pred = Phi @ coef_all + intercept

    # Goodness-of-fit
    ss_res = float(np.sum((y - y_pred) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else 0.0
    n = len(y)
    n_sel = int(np.sum(coef_all != 0.0))
    dof = n - n_sel - 1
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    bic = n * np.log(ss_res / n + 1e-300) + (n_sel + 1) * np.log(n) if n > 0 else float("nan")
    aic = n * np.log(ss_res / n + 1e-300) + 2 * (n_sel + 1) if n > 0 else float("nan")

    # Selected terms
    sel_mask = coef_all != 0.0
    sel_coef = coef_all[sel_mask].tolist()
    sel_base = base_indices[sel_mask].tolist()
    sel_exp = exponents[sel_mask].tolist()
    # selected_terms: [[base_idx, exponent], ...] (excluding bias term with base_idx == -1)
    sel_terms = [
        [int(b), int(e)] for b, e in zip(sel_base, sel_exp) if b >= 0
    ]
    # selected_base_indices: sorted unique base feature indices (excluding bias)
    sel_unique_base = sorted(set(int(b) for b in sel_base if b >= 0))
    sel_names = [
        f"x{b}^{e}" if e >= 0 else f"x{b}^(-{abs(e)})"
        for b, e in sel_terms
    ]

    # FDR / TPR (if ground truth provided) - use polynomial term-level evaluation
    fdr = tpr = None
    n_tp = n_fp = n_fn = None
    if true_poly_terms is not None:
        from ic_knockoff_poly_reg.evaluation import compute_polynomial_term_metrics
        metrics = compute_polynomial_term_metrics(
            selected_terms=sel_terms,
            true_poly_terms=true_poly_terms,
        )
        fdr = metrics.fdr
        tpr = metrics.tpr
        n_tp = metrics.n_true_positives
        n_fp = metrics.n_false_positives
        n_fn = metrics.n_false_negatives

    return ResultBundle(
        method=method,
        dataset=dataset,
        selected_names=sel_names,
        selected_base_indices=sel_unique_base,
        selected_terms=sel_terms,
        coef=sel_coef,
        intercept=float(intercept),
        n_selected=n_sel,
        r_squared=float(r2),
        adj_r_squared=float(adj_r2),
        residual_ss=float(ss_res),
        total_ss=float(ss_tot),
        bic=float(bic),
        aic=float(aic),
        elapsed_seconds=float(elapsed_seconds),
        peak_memory_mb=float(peak_memory_mb),
        fdr=fdr,
        tpr=tpr,
        n_true_positives=n_tp,
        n_false_positives=n_fp,
        n_false_negatives=n_fn,
        params={"degree": degree},
    )


# ---------------------------------------------------------------------------
# RustPolyLasso
# ---------------------------------------------------------------------------

class RustPolyLasso:
    """PolyLasso backed by the Rust kernel (drop-in for :class:`PolyLasso`)."""

    def __init__(self, degree: int = 2, random_state: Optional[int] = None):
        self.degree = degree
        self.random_state = random_state
        self._coef_all: Optional[NDArray] = None
        self._intercept: float = 0.0
        self._base_indices: Optional[NDArray] = None
        self._exponents: Optional[NDArray] = None
        self._p: int = 0

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RustPolyLasso":
        lib = _get_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y.ravel(), dtype=np.float64)
        n, p = X.shape
        self._p = p

        n_cols = lib.ic_poly_n_expanded(p, self.degree, 1)
        out_coef = np.zeros(n_cols, dtype=np.float64)
        out_intercept = np.zeros(1, dtype=np.float64)
        out_bi = np.zeros(n_cols, dtype=np.int32)
        out_ex = np.zeros(n_cols, dtype=np.int32)

        ret = lib.ic_baseline_poly_lasso_fit(
            _dbl_ptr(X), _dbl_ptr(y),
            n, p, self.degree,
            _dbl_ptr(out_coef),
            _dbl_ptr(out_intercept),
            _int_ptr(out_bi),
            _int_ptr(out_ex),
        )
        if ret < 0:
            raise RuntimeError("ic_baseline_poly_lasso_fit returned an error")

        self._coef_all = out_coef
        self._intercept = float(out_intercept[0])
        self._base_indices = out_bi
        self._exponents = out_ex
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._coef_all is None:
            raise RuntimeError("Call fit() first")
        from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
        Phi = PolynomialDictionary(degree=self.degree, include_bias=True).expand(
            np.ascontiguousarray(X, dtype=np.float64)
        ).matrix
        return Phi @ self._coef_all + self._intercept

    def to_result_bundle(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        dataset: str = "",
        true_poly_terms: Optional[list] = None,
        elapsed_seconds: float = 0.0,
        peak_memory_mb: float = 0.0,
    ) -> ResultBundle:
        """Build a result bundle from the fitted model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            dataset: Dataset name.
            true_poly_terms: Ground-truth polynomial terms as [base_idx, exponent] pairs.
            elapsed_seconds: Elapsed time in seconds.
            peak_memory_mb: Peak memory usage in MB.
        
        Returns:
            ResultBundle containing the fitted model results.
        """
        if self._coef_all is None:
            raise RuntimeError("Call fit() first")
        return _build_result_bundle(
            "poly_lasso", X, np.asarray(y).ravel(), self.degree,
            self._coef_all, self._intercept,
            self._base_indices, self._exponents,
            dataset, true_poly_terms, elapsed_seconds, peak_memory_mb,
        )


# ---------------------------------------------------------------------------
# RustPolyOMP
# ---------------------------------------------------------------------------

class RustPolyOMP:
    """PolyOMP backed by the Rust kernel (drop-in for :class:`PolyOMP`)."""

    def __init__(self, degree: int = 2, max_nonzero: int = 0):
        self.degree = degree
        self.max_nonzero = max_nonzero
        self._coef_all: Optional[NDArray] = None
        self._intercept: float = 0.0
        self._base_indices: Optional[NDArray] = None
        self._exponents: Optional[NDArray] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RustPolyOMP":
        lib = _get_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y.ravel(), dtype=np.float64)
        n, p = X.shape

        n_cols = lib.ic_poly_n_expanded(p, self.degree, 1)
        out_coef = np.zeros(n_cols, dtype=np.float64)
        out_intercept = np.zeros(1, dtype=np.float64)
        out_bi = np.zeros(n_cols, dtype=np.int32)
        out_ex = np.zeros(n_cols, dtype=np.int32)

        ret = lib.ic_baseline_poly_omp_fit(
            _dbl_ptr(X), _dbl_ptr(y),
            n, p, self.degree, self.max_nonzero,
            _dbl_ptr(out_coef),
            _dbl_ptr(out_intercept),
            _int_ptr(out_bi),
            _int_ptr(out_ex),
        )
        if ret < 0:
            raise RuntimeError("ic_baseline_poly_omp_fit returned an error")

        self._coef_all = out_coef
        self._intercept = float(out_intercept[0])
        self._base_indices = out_bi
        self._exponents = out_ex
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._coef_all is None:
            raise RuntimeError("Call fit() first")
        from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
        Phi = PolynomialDictionary(degree=self.degree, include_bias=True).expand(
            np.ascontiguousarray(X, dtype=np.float64)
        ).matrix
        return Phi @ self._coef_all + self._intercept

    def to_result_bundle(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        dataset: str = "",
        true_poly_terms: Optional[list] = None,
        elapsed_seconds: float = 0.0,
        peak_memory_mb: float = 0.0,
    ) -> ResultBundle:
        """Build a result bundle from the fitted model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            dataset: Dataset name.
            true_poly_terms: Ground-truth polynomial terms as [base_idx, exponent] pairs.
            elapsed_seconds: Elapsed time in seconds.
            peak_memory_mb: Peak memory usage in MB.
        
        Returns:
            ResultBundle containing the fitted model results.
        """
        if self._coef_all is None:
            raise RuntimeError("Call fit() first")
        return _build_result_bundle(
            "poly_omp", X, np.asarray(y).ravel(), self.degree,
            self._coef_all, self._intercept,
            self._base_indices, self._exponents,
            dataset, true_poly_terms, elapsed_seconds, peak_memory_mb,
        )


# ---------------------------------------------------------------------------
# RustSparsePolySTLSQ
# ---------------------------------------------------------------------------

class RustSparsePolySTLSQ:
    """SparsePolySTLSQ backed by the Rust kernel (drop-in for :class:`SparsePolySTLSQ`)."""

    def __init__(self, degree: int = 2, threshold: float = -1.0):
        self.degree = degree
        self.threshold = threshold
        self._coef_all: Optional[NDArray] = None
        self._intercept: float = 0.0
        self._base_indices: Optional[NDArray] = None
        self._exponents: Optional[NDArray] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RustSparsePolySTLSQ":
        lib = _get_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y.ravel(), dtype=np.float64)
        n, p = X.shape

        n_cols = lib.ic_poly_n_expanded(p, self.degree, 1)
        out_coef = np.zeros(n_cols, dtype=np.float64)
        out_intercept = np.zeros(1, dtype=np.float64)
        out_bi = np.zeros(n_cols, dtype=np.int32)
        out_ex = np.zeros(n_cols, dtype=np.int32)

        ret = lib.ic_baseline_poly_stlsq_fit(
            _dbl_ptr(X), _dbl_ptr(y),
            n, p, self.degree, self.threshold,
            _dbl_ptr(out_coef),
            _dbl_ptr(out_intercept),
            _int_ptr(out_bi),
            _int_ptr(out_ex),
        )
        if ret < 0:
            raise RuntimeError("ic_baseline_poly_stlsq_fit returned an error")

        self._coef_all = out_coef
        self._intercept = float(out_intercept[0])
        self._base_indices = out_bi
        self._exponents = out_ex
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._coef_all is None:
            raise RuntimeError("Call fit() first")
        from ic_knockoff_poly_reg.polynomial import PolynomialDictionary
        Phi = PolynomialDictionary(degree=self.degree, include_bias=True).expand(
            np.ascontiguousarray(X, dtype=np.float64)
        ).matrix
        return Phi @ self._coef_all + self._intercept

    def to_result_bundle(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        dataset: str = "",
        true_poly_terms: Optional[list] = None,
        elapsed_seconds: float = 0.0,
        peak_memory_mb: float = 0.0,
    ) -> ResultBundle:
        """Build a result bundle from the fitted model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            dataset: Dataset name.
            true_poly_terms: Ground-truth polynomial terms as [base_idx, exponent] pairs.
            elapsed_seconds: Elapsed time in seconds.
            peak_memory_mb: Peak memory usage in MB.
        
        Returns:
            ResultBundle containing the fitted model results.
        """
        if self._coef_all is None:
            raise RuntimeError("Call fit() first")
        return _build_result_bundle(
            "sparse_poly_stlsq", X, np.asarray(y).ravel(), self.degree,
            self._coef_all, self._intercept,
            self._base_indices, self._exponents,
            dataset, true_poly_terms, elapsed_seconds, peak_memory_mb,
        )

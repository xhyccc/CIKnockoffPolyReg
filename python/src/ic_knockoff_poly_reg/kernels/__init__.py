"""IC-Knock-Poly computational kernel backends.

This sub-package provides pluggable computational kernels for the three
algorithmic sub-tasks of IC-Knock-Poly:

* **Polynomial expansion** – :class:`PolynomialKernel`
* **Knockoff statistics** – :class:`KnockoffKernel`
* **PoSI threshold / alpha-spending** – :class:`PosiKernel`

Three backends are available:

``"python"`` (default)
    Pure-Python / NumPy implementation.  Always available; no compilation
    required.

``"cpp"``
    C++17 implementation loaded via :mod:`ctypes`.  Requires building the
    shared library first::

        cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
        cmake --build cpp/build

    Set ``LIBIC_KNOCKOFF_CPP_PATH`` to override the default search path.

``"rust"``
    Rust implementation loaded via :mod:`ctypes`.  Requires building the
    cdylib first::

        cd rust && cargo build --release

    Set ``LIBIC_KNOCKOFF_RUST_PATH`` to override the default search path.

Usage example::

    from ic_knockoff_poly_reg.kernels import create_kernels

    poly_kernel, knockoff_kernel, posi_kernel = create_kernels("python")
    # or: create_kernels("cpp") / create_kernels("rust")

    ef = poly_kernel.expand(X, degree=2, include_bias=True)
    W  = knockoff_kernel.w_statistics(beta_orig, beta_knock)
    qt = posi_kernel.alpha_spending_budget(t=1, Q=0.1, sequence="riemann_zeta")

You can also use the :attr:`AVAILABLE_BACKENDS` set to discover which
backends are currently loadable (i.e. their shared libraries have been
compiled).
"""

from __future__ import annotations

from typing import NamedTuple

from .base import KnockoffKernel, PosiKernel, PolynomialKernel
from .python_kernel import PythonKnockoffKernel, PythonPosiKernel, PythonPolynomialKernel

__all__ = [
    # Abstract base classes
    "PolynomialKernel",
    "KnockoffKernel",
    "PosiKernel",
    # Concrete Python implementations
    "PythonPolynomialKernel",
    "PythonKnockoffKernel",
    "PythonPosiKernel",
    # Factory
    "create_kernels",
    "KernelSet",
    "AVAILABLE_BACKENDS",
]


class KernelSet(NamedTuple):
    """Container for all three kernel instances of a given backend."""

    poly: PolynomialKernel
    knockoff: KnockoffKernel
    posi: PosiKernel


def create_kernels(backend: str = "python") -> KernelSet:
    """Return a :class:`KernelSet` for the requested *backend*.

    Parameters
    ----------
    backend : str
        One of ``"python"``, ``"cpp"``, or ``"rust"``.

    Returns
    -------
    KernelSet
        Named tuple ``(poly, knockoff, posi)`` whose elements implement
        :class:`PolynomialKernel`, :class:`KnockoffKernel`, and
        :class:`PosiKernel` respectively.

    Raises
    ------
    ValueError
        If *backend* is not one of the supported values.
    RuntimeError
        If the requested native backend shared library cannot be found.
        Build instructions are included in the error message.
    """
    if backend == "python":
        return KernelSet(
            poly=PythonPolynomialKernel(),
            knockoff=PythonKnockoffKernel(),
            posi=PythonPosiKernel(),
        )
    if backend == "cpp":
        from .cpp_kernel import CppKnockoffKernel, CppPosiKernel, CppPolynomialKernel
        return KernelSet(
            poly=CppPolynomialKernel(),
            knockoff=CppKnockoffKernel(),
            posi=CppPosiKernel(),
        )
    if backend == "rust":
        from .rust_kernel import RustKnockoffKernel, RustPosiKernel, RustPolynomialKernel
        return KernelSet(
            poly=RustPolynomialKernel(),
            knockoff=RustKnockoffKernel(),
            posi=RustPosiKernel(),
        )
    raise ValueError(
        f"Unknown backend {backend!r}.  "
        "Supported values are: 'python', 'cpp', 'rust'."
    )


def _probe_backend(name: str) -> bool:
    """Return True if the *name* backend can be loaded without error."""
    if name == "python":
        return True
    try:
        if name == "cpp":
            from .cpp_kernel import _find_cpp_library
            _find_cpp_library()
        elif name == "rust":
            from .rust_kernel import _find_rust_library
            _find_rust_library()
        return True
    except (RuntimeError, OSError):
        return False


# Populated lazily on first access (import time would slow things down
# when native libs are absent, so we compute it once here).
AVAILABLE_BACKENDS: frozenset[str] = frozenset(
    b for b in ("python", "cpp", "rust") if _probe_backend(b)
)

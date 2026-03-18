"""Abstract base classes for the IC-Knock-Poly computational kernels.

Three kernel interfaces are defined, each corresponding to a distinct
algorithmic sub-task:

* :class:`PolynomialKernel` – rational polynomial dictionary expansion Φ(X).
* :class:`KnockoffKernel`   – knockoff W-statistics, s-values, and sampling.
* :class:`PosiKernel`       – PoSI alpha-spending budgets and knockoff+ threshold.

Concrete implementations are provided for three backends:

* ``"python"`` – pure Python / NumPy (always available, the default).
* ``"cpp"``    – C++17 via :mod:`ctypes`; requires building the shared library
                 (``cmake --build cpp/build``).
* ``"rust"``   – Rust via :mod:`ctypes`; requires building the shared library
                 (``cargo build --release`` inside ``rust/``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..polynomial import ExpandedFeatures


class PolynomialKernel(ABC):
    """Kernel interface for rational polynomial dictionary expansion Φ(X).

    Implementations must produce results that are numerically equivalent to
    :class:`~ic_knockoff_poly_reg.polynomial.PolynomialDictionary`.
    """

    @abstractmethod
    def n_expanded(self, n_base: int, degree: int, include_bias: bool) -> int:
        """Return the number of columns produced by :meth:`expand`.

        Parameters
        ----------
        n_base : int
            Number of base features.
        degree : int
            Maximum absolute exponent.
        include_bias : bool
            Whether a constant-1 bias column is included.
        """
        ...

    @abstractmethod
    def expand(
        self,
        X: NDArray[np.float64],
        degree: int,
        include_bias: bool,
        clip_threshold: float = 1e-8,
        base_names: Optional[list[str]] = None,
    ) -> ExpandedFeatures:
        """Expand base feature matrix X (n, p) to polynomial dictionary Φ(X).

        Parameters
        ----------
        X : (n_samples, n_features) array
        degree : int
            Maximum absolute exponent (>= 1).
        include_bias : bool
            Append a constant-1 column when True.
        clip_threshold : float
            Values |x| < clip_threshold are clamped before negative powers.
        base_names : list of str or None
            Names for the base features.  Defaults to ``["x0", "x1", ...]``.

        Returns
        -------
        ExpandedFeatures
        """
        ...


class KnockoffKernel(ABC):
    """Kernel interface for knockoff W-statistics, s-values, and sampling.

    Implementations must produce results that are numerically equivalent to
    the corresponding functions in
    :mod:`~ic_knockoff_poly_reg.knockoffs`.
    """

    @abstractmethod
    def w_statistics(
        self,
        beta_original: NDArray[np.float64],
        beta_knockoff: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute W_j = |beta_j| − |beta_tilde_j| for all j.

        Parameters
        ----------
        beta_original : 1-D array of length p
            Regression coefficients for the original features.
        beta_knockoff : 1-D array of length p
            Regression coefficients for the knockoff copies.

        Returns
        -------
        W : 1-D array of length p
        """
        ...

    @abstractmethod
    def equicorrelated_s_values(
        self,
        cov: NDArray[np.float64],
        reg: float = 1e-10,
    ) -> NDArray[np.float64]:
        """Compute equicorrelated s-values for covariance matrix Sigma.

        Parameters
        ----------
        cov : (p, p) symmetric positive-definite matrix
        reg : float
            Small regularisation for numerical stability.

        Returns
        -------
        s : 1-D array of length p
        """
        ...

    @abstractmethod
    def sample_gaussian_knockoffs(
        self,
        X: NDArray[np.float64],
        mu: NDArray[np.float64],
        Sigma: NDArray[np.float64],
        seed: int = 42,
    ) -> NDArray[np.float64]:
        """Sample equicorrelated Gaussian knockoffs X̃ for X ~ N(mu, Sigma).

        Parameters
        ----------
        X : (n_samples, p) array
        mu : 1-D array of length p – column means
        Sigma : (p, p) covariance matrix (must be SPD)
        seed : int – random seed

        Returns
        -------
        X_tilde : (n_samples, p) knockoff matrix
        """
        ...


class PosiKernel(ABC):
    """Kernel interface for PoSI alpha-spending and knockoff+ threshold.

    Implementations must produce results that are numerically equivalent to
    :class:`~ic_knockoff_poly_reg.posi_threshold.AlphaSpending` and
    :func:`~ic_knockoff_poly_reg.posi_threshold.compute_knockoff_threshold`.
    """

    @abstractmethod
    def alpha_spending_budget(
        self,
        t: int,
        Q: float,
        sequence: str,
        gamma: float = 0.5,
    ) -> float:
        """Return the FDR budget q_t for iteration t.

        Parameters
        ----------
        t : int
            Iteration index (>= 1).
        Q : float
            Global target FDR level in (0, 1).
        sequence : str
            ``"riemann_zeta"`` or ``"geometric"``.
        gamma : float
            Geometric decay (only used when sequence="geometric").

        Returns
        -------
        q_t : float
        """
        ...

    @abstractmethod
    def knockoff_threshold(
        self,
        W: NDArray[np.float64],
        q_t: float,
        active_poly: Optional[set[int]] = None,
        offset: int = 1,
    ) -> float:
        """Compute the knockoff+ threshold tau_t.

        Parameters
        ----------
        W : 1-D array of W-statistics for all candidate features.
        q_t : float
            FDR budget for the current iteration.
        active_poly : set of int or None
            Indices of features already selected (excluded from computation).
        offset : int
            1 for knockoff+ (default), 0 for standard knockoff.

        Returns
        -------
        tau_t : float
            Returns ``numpy.inf`` if no threshold satisfies the FDR condition.
        """
        ...

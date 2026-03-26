"""Rust-accelerated Penalized GMM.

This module provides Python bindings to the Rust GMM implementation
for significant speedup over the Python version.
"""

import numpy as np
import ctypes
from pathlib import Path
from typing import Optional, List

# Load the Rust shared library
_rust_lib = None

def _get_rust_lib():
    global _rust_lib
    if _rust_lib is None:
        # Find the shared library
        # CRITICAL: Must resolve FIRST before getting parents!
        # Path('/a/../b').parent.parent != Path('/a/../b').resolve().parent.parent
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        lib_paths = [
            repo_root / "rust" / "target" / "release" / "libic_knockoff_poly_reg.dylib",  # macOS
            repo_root / "rust" / "target" / "release" / "libic_knockoff_poly_reg.so",   # Linux
            repo_root / "rust" / "target" / "release" / "ic_knockoff_poly_reg.dll",     # Windows
        ]
        
        lib_path = None
        for p in lib_paths:
            if p.exists():
                lib_path = p
                break
        
        if lib_path is None:
            raise RuntimeError(f"Rust library not found. Searched in: {repo_root}. Please build with: cd rust && cargo build --release")
        
        _rust_lib = ctypes.CDLL(str(lib_path))
        
        # Set up function signatures
        _rust_lib.ic_gmm_fit.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # x_flat
            ctypes.c_int,                     # n
            ctypes.c_int,                     # p
            ctypes.c_int,                     # n_components
            ctypes.c_double,                  # alpha
            ctypes.c_int,                     # max_iter
            ctypes.c_uint64,                  # seed
            ctypes.POINTER(ctypes.c_double),  # out_weights
            ctypes.POINTER(ctypes.c_double),  # out_means
            ctypes.POINTER(ctypes.c_double),  # out_precisions
        ]
        _rust_lib.ic_gmm_fit.restype = ctypes.c_int
    
    return _rust_lib


class RustPenalizedGMM:
    """Rust-accelerated Penalized GMM with Graphical Lasso.
    
    This is significantly faster than the Python implementation,
    especially for larger datasets.
    
    Parameters
    ----------
    n_components : int
        Number of mixture components (1 for single Gaussian).
    alpha : float
        L1 regularization parameter for precision matrix.
    max_iter : int
        Maximum EM iterations.
    random_state : int or None
        Random seed for initialization.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        alpha: float = 0.1,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state or 42
        
        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.precisions_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
    
    @staticmethod
    def adaptive_alpha(n: int, p: int) -> float:
        """Compute adaptive alpha based on n and p (same logic as Rust).
        
        Formula: alpha = c * sqrt(log(p) / n)
        where c = 0.5 if p <= n, c = 1.0 if p > n
        
        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features.
            
        Returns
        -------
        alpha : float
            Adaptive regularization parameter.
        """
        import numpy as np
        n_eff = max(n, 10)
        p_eff = max(p, 2)
        
        # Adaptive c: 0.5 for standard, 1.0 for high-dim (p > n)
        c = 0.5 if p_eff <= n_eff else 1.0
        
        alpha = c * np.sqrt(np.log(p_eff) / n_eff)
        
        # Clamp to [0.01, 1.0]
        return float(np.clip(alpha, 0.01, 1.0))
    
    @classmethod
    def with_adaptive_alpha(
        cls,
        n_components: int = 2,
        n: int = 100,
        p: int = 10,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ) -> "RustPenalizedGMM":
        """Create a RustPenalizedGMM with adaptive alpha.
        
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        n : int
            Number of samples (for alpha calculation).
        p : int
            Number of features (for alpha calculation).
        max_iter : int
            Maximum EM iterations.
        random_state : int or None
            Random seed for initialization.
            
        Returns
        -------
        RustPenalizedGMM
            GMM with adaptively computed alpha.
        """
        alpha = cls.adaptive_alpha(n, p)
        return cls(n_components=n_components, alpha=alpha, max_iter=max_iter, random_state=random_state)
    
    def fit(self, X: np.ndarray) -> "RustPenalizedGMM":
        """Fit the GMM to data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        k = self.n_components
        
        # Prepare output arrays
        weights = np.zeros(k, dtype=np.float64)
        means = np.zeros((k, p), dtype=np.float64)
        precisions = np.zeros((k, p, p), dtype=np.float64)
        
        # Call Rust function
        lib = _get_rust_lib()
        result = lib.ic_gmm_fit(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_int(p),
            ctypes.c_int(k),
            ctypes.c_double(self.alpha),
            ctypes.c_int(self.max_iter),
            ctypes.c_uint64(self.random_state),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            precisions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        
        if result < 0:
            raise RuntimeError("GMM fitting failed in Rust")
        
        self.weights_ = weights
        self.means_ = means
        self.precisions_ = precisions
        
        # Compute covariances as inverse of precisions
        self.covariances_ = np.zeros_like(precisions)
        for i in range(k):
            try:
                self.covariances_[i] = np.linalg.inv(precisions[i])
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                self.covariances_[i] = np.linalg.pinv(precisions[i])
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities of each component.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict.
            
        Returns
        -------
        resp : ndarray of shape (n_samples, n_components)
            Posterior probabilities.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        k = self.n_components
        
        resp = np.zeros((n, k))
        
        for i in range(n):
            densities = np.zeros(k)
            for j in range(k):
                diff = X[i] - self.means_[j]
                # Compute Gaussian density
                log_det = np.linalg.slogdet(self.precisions_[j])[1]
                quad = diff @ self.precisions_[j] @ diff
                densities[j] = self.weights_[j] * np.exp(-0.5 * (log_det + quad))
            
            total = densities.sum()
            if total > 0:
                resp[i] = densities / total
            else:
                resp[i] = 1.0 / k
        
        return resp

    def compute_conditional_params(
        self,
        X_A: np.ndarray,
        A_indices: np.ndarray,
        B_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute conditional Gaussian parameters for X_B | X_A.
        
        Simplified version for single-component GMM (most common case).
        For multi-component, uses the component with highest weight.
        """
        X_A = np.asarray(X_A, dtype=np.float64)
        A_indices = np.asarray(A_indices, dtype=int)
        B_indices = np.asarray(B_indices, dtype=int)
        
        n = X_A.shape[0]
        
        # For simplicity, use the component with highest weight
        # (for n_components=1, this is the only component)
        comp_idx = np.argmax(self.weights_)
        mu = self.means_[comp_idx]
        Theta = self.precisions_[comp_idx]
        
        mu_A = mu[A_indices]
        mu_B = mu[B_indices]
        
        Theta_BB = Theta[np.ix_(B_indices, B_indices)]
        Theta_BA = Theta[np.ix_(B_indices, A_indices)]
        
        # Conditional precision
        cond_precision = Theta_BB
        
        # Conditional mean: mu_B - Theta_BB^{-1} Theta_BA (x_A - mu_A)
        try:
            Theta_BB_inv = np.linalg.inv(Theta_BB)
        except np.linalg.LinAlgError:
            Theta_BB_inv = np.linalg.pinv(Theta_BB)
        
        delta = X_A - mu_A[np.newaxis, :]
        cond_means = mu_B[np.newaxis, :] - (delta @ Theta_BA.T) @ Theta_BB_inv
        
        # Dummy log_resp (single component)
        log_resp = np.zeros((n, self.n_components))
        
        return cond_means, cond_precision, log_resp

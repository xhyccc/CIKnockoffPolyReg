"""IC-Knock-Poly: Iterative Conditional Knockoffs for Sparse Rational Polynomial Regression.

Implements the IC-Knock-Poly algorithm that identifies sparse rational polynomial
equations in ultra-high dimensional settings (p >> n) while strictly controlling
the False Discovery Rate (FDR) via a PoSI alpha-spending sequence.
"""

from .algorithm import ICKnockoffPolyReg
from .gmm_phase import PenalizedGMM
from .knockoffs import ConditionalKnockoffGenerator
from .polynomial import PolynomialDictionary
from .posi_threshold import AlphaSpending, compute_knockoff_threshold
from .evaluation import compute_fdr, compute_tpr, DiscoveryMetrics, ResultBundle
from .kernels import (
    create_kernels,
    KernelSet,
    PolynomialKernel,
    KnockoffKernel,
    PosiKernel,
    PythonPolynomialKernel,
    PythonKnockoffKernel,
    PythonPosiKernel,
    AVAILABLE_BACKENDS,
)

__all__ = [
    "ICKnockoffPolyReg",
    "PenalizedGMM",
    "ConditionalKnockoffGenerator",
    "PolynomialDictionary",
    "AlphaSpending",
    "compute_knockoff_threshold",
    "compute_fdr",
    "compute_tpr",
    "DiscoveryMetrics",
    "ResultBundle",
    # Kernel backend API
    "create_kernels",
    "KernelSet",
    "PolynomialKernel",
    "KnockoffKernel",
    "PosiKernel",
    "PythonPolynomialKernel",
    "PythonKnockoffKernel",
    "PythonPosiKernel",
    "AVAILABLE_BACKENDS",
]

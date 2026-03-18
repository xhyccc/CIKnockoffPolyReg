"""Simulation experiments for IC-Knock-Poly.

This sub-package provides:

* ``data_generator`` — synthesised datasets drawn from a GMM with a
  k-sparse rational polynomial response.
* ``run_simulation`` — simulation sweep over varying dimensions, sample
  sizes, and evaluation settings (supervised vs. semi-supervised).

Quick start
-----------
Run the default simulation sweep from the command line::

    python -m simulations.run_simulation

Or call the Python API::

    from simulations.run_simulation import run_simulation_suite, default_configs
    results = run_simulation_suite(default_configs())
"""

from .data_generator import SimulatedDataset, generate_gmm_features, generate_simulation
from .run_simulation import (
    SimulationConfig,
    SimulationResult,
    run_simulation,
    run_simulation_suite,
    default_configs,
)

__all__ = [
    "SimulatedDataset",
    "generate_gmm_features",
    "generate_simulation",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
    "run_simulation_suite",
    "default_configs",
]

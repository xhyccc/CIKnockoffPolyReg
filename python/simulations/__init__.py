"""Simulation experiments for IC-Knock-Poly.

This sub-package provides:

* ``data_generator`` — synthesised datasets drawn from a GMM with a
  k-sparse rational polynomial response.
* ``run_simulation`` — simulation sweep over varying dimensions, sample
  sizes, evaluation settings, polynomial degrees, and non-zero element
  counts (supervised vs. semi-supervised).
* ``visualize`` — publication-quality plots for prediction error,
  scalability, selection quality metrics, and non-zero identification.

Quick start
-----------
Run the default simulation sweep from the command line::

    python -m simulations.run_simulation

    # degree × nonzero sweep (degree=[2,3], k=[5,10,15,20])
    python -m simulations.run_simulation --sweep degree_nonzero --output results/dn

Or call the Python API::

    from simulations.run_simulation import (
        run_simulation_suite, default_configs, sweep_degree_nonzero_configs,
    )
    from simulations.visualize import plot_all

    results = run_simulation_suite(default_configs())
    plot_all(results, output_dir="figures/")
"""

from .data_generator import SimulatedDataset, generate_gmm_features, generate_simulation
from .run_simulation import (
    SimulationConfig,
    SimulationResult,
    TrialResult,
    run_simulation,
    run_simulation_suite,
    default_configs,
    sweep_degree_nonzero_configs,
)

__all__ = [
    "SimulatedDataset",
    "generate_gmm_features",
    "generate_simulation",
    "SimulationConfig",
    "SimulationResult",
    "TrialResult",
    "run_simulation",
    "run_simulation_suite",
    "default_configs",
    "sweep_degree_nonzero_configs",
]

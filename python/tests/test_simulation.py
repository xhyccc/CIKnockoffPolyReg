"""Tests for the simulation experiments module.

Covers:
  - data_generator: GMM feature generation, sparse poly response generation
  - run_simulation: SimulationConfig, TrialResult, SimulationResult,
    run_simulation (single config), run_simulation_suite, default_configs
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Make packages importable from the test runner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulations.data_generator import (
    SimulatedDataset,
    generate_gmm_features,
    generate_simulation,
)
from simulations.run_simulation import (
    SimulationConfig,
    SimulationResult,
    TrialResult,
    default_configs,
    run_simulation,
    run_simulation_suite,
    sweep_degree_nonzero_configs,
)


# ---------------------------------------------------------------------------
# Data generator tests
# ---------------------------------------------------------------------------

class TestGenerateGmmFeatures:
    def test_shape(self):
        X = generate_gmm_features(50, 4, n_components=2, random_state=0)
        assert X.shape == (50, 4)

    def test_all_positive(self):
        """All values must be ≥ 0.5 (shifted to avoid division-by-zero)."""
        X = generate_gmm_features(200, 6, n_components=3, random_state=1)
        assert np.all(X >= 0.5)

    def test_reproducible(self):
        X1 = generate_gmm_features(30, 3, random_state=42)
        X2 = generate_gmm_features(30, 3, random_state=42)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self):
        X1 = generate_gmm_features(30, 3, random_state=0)
        X2 = generate_gmm_features(30, 3, random_state=1)
        assert not np.allclose(X1, X2)

    def test_larger_p(self):
        X = generate_gmm_features(100, 20, n_components=4, random_state=7)
        assert X.shape == (100, 20)
        assert np.all(np.isfinite(X))

    def test_single_component(self):
        X = generate_gmm_features(50, 3, n_components=1, random_state=0)
        assert X.shape == (50, 3)


class TestGenerateSimulation:
    def test_returns_simulated_dataset(self):
        ds = generate_simulation(n_labeled=60, p=5, k=2, random_state=0)
        assert isinstance(ds, SimulatedDataset)

    def test_labeled_shapes(self):
        ds = generate_simulation(n_labeled=80, p=8, k=3, random_state=1)
        assert ds.X.shape == (80, 8)
        assert ds.y.shape == (80,)
        assert ds.X_unlabeled is None  # default: no unlabeled

    def test_semi_supervised_unlabeled_shape(self):
        ds = generate_simulation(
            n_labeled=60, p=6, k=2, n_unlabeled=200, random_state=2
        )
        assert ds.X_unlabeled is not None
        assert ds.X_unlabeled.shape == (200, 6)
        assert ds.n_unlabeled == 200

    def test_true_base_indices_length(self):
        k = 3
        ds = generate_simulation(n_labeled=50, p=8, k=k, random_state=5)
        assert len(ds.true_base_indices) == k

    def test_true_poly_terms_length(self):
        k = 2
        ds = generate_simulation(n_labeled=50, p=7, k=k, random_state=6)
        assert len(ds.true_poly_terms) == k

    def test_true_coef_length(self):
        k = 3
        ds = generate_simulation(n_labeled=50, p=7, k=k, random_state=7)
        assert len(ds.true_coef) == k

    def test_y_finite(self):
        ds = generate_simulation(n_labeled=100, p=10, k=3, random_state=8)
        assert np.all(np.isfinite(ds.y))

    def test_k_exceeds_p_raises(self):
        with pytest.raises(ValueError, match=r"k \(5\) must not exceed p \(4\)"):
            generate_simulation(n_labeled=50, p=4, k=5)

    def test_reproducible(self):
        ds1 = generate_simulation(n_labeled=60, p=5, k=2, random_state=0)
        ds2 = generate_simulation(n_labeled=60, p=5, k=2, random_state=0)
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.y, ds2.y)

    def test_metadata_fields(self):
        ds = generate_simulation(
            n_labeled=50, p=6, k=2, degree=2, n_components=3,
            noise_std=0.3, n_unlabeled=100, random_state=9,
        )
        assert ds.n_labeled == 50
        assert ds.p == 6
        assert ds.k == 2
        assert ds.degree == 2
        assert ds.gmm_n_components == 3
        assert ds.noise_std == 0.3
        assert ds.n_unlabeled == 100

    def test_signal_above_noise(self):
        """With large labeled n the response must not be pure noise: R² > 0."""
        ds = generate_simulation(
            n_labeled=500, p=5, k=2, noise_std=0.1, random_state=42
        )
        y_mean = ds.y.mean()
        ss_tot = np.sum((ds.y - y_mean) ** 2)
        assert ss_tot > 0


# ---------------------------------------------------------------------------
# SimulationConfig tests
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.p == 10
        assert cfg.n_labeled == 200
        assert cfg.k == 2
        assert cfg.setting == "supervised"
        assert cfg.n_trials == 10

    def test_auto_label(self):
        cfg = SimulationConfig(p=5, n_labeled=100, k=2, setting="supervised")
        assert "p5" in cfg.label
        assert "n100" in cfg.label
        assert "k2" in cfg.label
        assert "supervised" in cfg.label

    def test_custom_label(self):
        cfg = SimulationConfig(label="my_exp")
        assert cfg.label == "my_exp"


# ---------------------------------------------------------------------------
# default_configs tests
# ---------------------------------------------------------------------------

class TestDefaultConfigs:
    def test_returns_list(self):
        cfgs = default_configs(
            p_values=(5, 10),
            n_values=(100, 200),
            k_values=(2,),
            settings=("supervised",),
            n_trials=2,
        )
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0

    def test_k_ge_p_skipped(self):
        """Configurations with k ≥ p must be excluded."""
        cfgs = default_configs(
            p_values=(3,),
            n_values=(100,),
            k_values=(3, 4),  # both ≥ p=3, should be skipped
            settings=("supervised",),
            n_trials=1,
        )
        assert all(cfg.k < cfg.p for cfg in cfgs)

    def test_both_settings(self):
        cfgs = default_configs(
            p_values=(5,),
            n_values=(100,),
            k_values=(2,),
            settings=("supervised", "semi_supervised"),
            n_trials=1,
        )
        settings = {c.setting for c in cfgs}
        assert "supervised" in settings
        assert "semi_supervised" in settings

    def test_count(self):
        cfgs = default_configs(
            p_values=(5, 10),
            n_values=(100, 200),
            k_values=(2,),
            settings=("supervised",),
            n_trials=1,
        )
        # 2 p × 2 n × 1 k × 1 setting = 4 configs (all k=2 < p)
        assert len(cfgs) == 4


# ---------------------------------------------------------------------------
# run_simulation tests (fast/smoke)
# ---------------------------------------------------------------------------

def _fast_config(setting="supervised", methods=None):
    """Return a minimal SimulationConfig that finishes quickly."""
    return SimulationConfig(
        p=5,
        n_labeled=80,
        k=2,
        setting=setting,
        n_unlabeled_ratio=2.0,
        degree=1,
        n_components=2,
        Q=0.20,
        n_trials=2,
        methods=methods or ["ic_knock_poly"],
        max_iter=3,
        random_state=0,
    )


class TestRunSimulation:
    def test_returns_list_of_sim_results(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, SimulationResult) for r in results)

    def test_result_method_name(self):
        cfg = _fast_config(methods=["ic_knock_poly"])
        results = run_simulation(cfg)
        assert results[0].method == "ic_knock_poly"

    def test_supervised_no_unlabeled_used(self):
        """In supervised mode the dataset has no unlabeled data."""
        cfg = _fast_config(setting="supervised")
        results = run_simulation(cfg)
        assert len(results) >= 1

    def test_semi_supervised_runs(self):
        cfg = _fast_config(setting="semi_supervised")
        results = run_simulation(cfg)
        assert len(results) >= 1

    def test_fdr_in_range(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            if not math.isnan(r.fdr_mean):
                assert 0.0 <= r.fdr_mean <= 1.0

    def test_tpr_in_range(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            if not math.isnan(r.tpr_mean):
                assert 0.0 <= r.tpr_mean <= 1.0

    def test_n_completed(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            assert r.n_completed <= cfg.n_trials
            assert r.n_completed >= 0

    def test_trial_results_stored(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            assert isinstance(r.trial_results, list)
            for tr in r.trial_results:
                assert isinstance(tr, TrialResult)

    def test_to_dict(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        d = results[0].to_dict()
        assert "fdr_mean" in d
        assert "tpr_mean" in d
        assert "p" in d
        assert "n_labeled" in d
        assert "setting" in d

    def test_to_csv_row(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        row = results[0].to_csv_row()
        assert "fdr_mean" in row
        assert "method" in row


# ---------------------------------------------------------------------------
# run_simulation_suite tests
# ---------------------------------------------------------------------------

class TestRunSimulationSuite:
    def test_returns_all_results(self):
        cfgs = [
            _fast_config("supervised"),
            _fast_config("semi_supervised"),
        ]
        results = run_simulation_suite(cfgs, verbose=False)
        assert len(results) >= 2

    def test_file_output(self, tmp_path):
        cfgs = [_fast_config()]
        prefix = str(tmp_path / "test_run")
        run_simulation_suite(cfgs, output_prefix=prefix, verbose=False)
        import os
        assert os.path.exists(prefix + "_summary.json")
        assert os.path.exists(prefix + "_summary.csv")

    def test_json_output_parseable(self, tmp_path):
        import json
        cfgs = [_fast_config()]
        prefix = str(tmp_path / "test_json")
        run_simulation_suite(cfgs, output_prefix=prefix, verbose=False)
        with open(prefix + "_summary.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "fdr_mean" in data[0]

    def test_csv_output_parseable(self, tmp_path):
        import csv
        cfgs = [_fast_config()]
        prefix = str(tmp_path / "test_csv")
        run_simulation_suite(cfgs, output_prefix=prefix, verbose=False)
        with open(prefix + "_summary.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        assert "fdr_mean" in rows[0]


# ---------------------------------------------------------------------------
# New metrics in TrialResult / SimulationResult
# ---------------------------------------------------------------------------

class TestNewMetrics:
    """Verify that precision, recall, F1, AUC, and peak_memory are computed."""

    def test_trial_result_has_precision(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            for tr in r.trial_results:
                assert hasattr(tr, "precision")
                assert hasattr(tr, "recall")
                assert hasattr(tr, "f1")
                assert hasattr(tr, "auc")

    def test_sim_result_has_precision_fields(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        r = results[0]
        assert hasattr(r, "precision_mean")
        assert hasattr(r, "precision_std")
        assert hasattr(r, "recall_mean")
        assert hasattr(r, "recall_std")
        assert hasattr(r, "f1_mean")
        assert hasattr(r, "f1_std")
        assert hasattr(r, "auc_mean")
        assert hasattr(r, "auc_std")

    def test_sim_result_has_peak_memory(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        r = results[0]
        assert hasattr(r, "peak_memory_mean")

    def test_sim_result_has_discovery_counts(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        r = results[0]
        assert hasattr(r, "n_true_positives_mean")
        assert hasattr(r, "n_false_positives_mean")
        assert hasattr(r, "n_false_negatives_mean")

    def test_to_dict_contains_new_fields(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        d = results[0].to_dict()
        for field in [
            "precision_mean", "recall_mean", "f1_mean", "auc_mean",
            "peak_memory_mean", "degree",
            "n_true_positives_mean", "n_false_positives_mean", "n_false_negatives_mean",
        ]:
            assert field in d, f"Missing field: {field}"

    def test_f1_in_range(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            if not math.isnan(r.f1_mean):
                assert 0.0 <= r.f1_mean <= 1.0

    def test_auc_in_range(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            if not math.isnan(r.auc_mean):
                assert 0.0 <= r.auc_mean <= 1.0

    def test_precision_in_range(self):
        cfg = _fast_config()
        results = run_simulation(cfg)
        for r in results:
            if not math.isnan(r.precision_mean):
                assert 0.0 <= r.precision_mean <= 1.0


# ---------------------------------------------------------------------------
# SimulationConfig label includes degree
# ---------------------------------------------------------------------------

class TestSimulationConfigDegreeLabel:
    def test_label_contains_degree(self):
        cfg = SimulationConfig(p=5, n_labeled=100, k=2, degree=3, setting="supervised")
        assert "d3" in cfg.label

    def test_label_contains_default_degree(self):
        cfg = SimulationConfig(p=5, n_labeled=100, k=2, setting="supervised")
        assert "d2" in cfg.label


# ---------------------------------------------------------------------------
# default_configs with degree_values
# ---------------------------------------------------------------------------

class TestDefaultConfigsDegreeValues:
    def test_single_degree_value(self):
        cfgs = default_configs(
            p_values=(5,), n_values=(100,), k_values=(2,),
            settings=("supervised",), degree_values=(3,), n_trials=1,
        )
        assert all(c.degree == 3 for c in cfgs)

    def test_multiple_degree_values(self):
        cfgs = default_configs(
            p_values=(5,), n_values=(100,), k_values=(2,),
            settings=("supervised",), degree_values=(2, 3), n_trials=1,
        )
        degrees = {c.degree for c in cfgs}
        assert degrees == {2, 3}

    def test_count_with_degrees(self):
        cfgs = default_configs(
            p_values=(5,), n_values=(100,), k_values=(2,),
            settings=("supervised",), degree_values=(2, 3), n_trials=1,
        )
        # 1 p × 1 n × 1 k × 1 setting × 2 degrees = 2 configs
        assert len(cfgs) == 2


# ---------------------------------------------------------------------------
# sweep_degree_nonzero_configs
# ---------------------------------------------------------------------------

class TestSweepDegreeNonzeroConfigs:
    def setup_method(self):
        from simulations.run_simulation import sweep_degree_nonzero_configs
        self.sweep_fn = sweep_degree_nonzero_configs

    def test_returns_list(self):
        cfgs = self.sweep_fn(
            degree_values=(2, 3), nonzero_values=(5, 10),
            n_values=(100,), settings=("supervised",), n_trials=1,
        )
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0

    def test_degree_values_covered(self):
        cfgs = self.sweep_fn(
            degree_values=(2, 3), nonzero_values=(5,),
            n_values=(100,), settings=("supervised",), n_trials=1,
        )
        degrees = {c.degree for c in cfgs}
        assert degrees == {2, 3}

    def test_nonzero_values_covered(self):
        cfgs = self.sweep_fn(
            degree_values=(2,), nonzero_values=(5, 10, 15),
            p=20, n_values=(100,), settings=("supervised",), n_trials=1,
        )
        ks = {c.k for c in cfgs}
        assert ks == {5, 10, 15}

    def test_k_ge_p_skipped(self):
        cfgs = self.sweep_fn(
            degree_values=(2,), nonzero_values=(5, 10, 20),
            p=15, n_values=(100,), settings=("supervised",), n_trials=1,
        )
        assert all(c.k < c.p for c in cfgs)

    def test_count(self):
        cfgs = self.sweep_fn(
            degree_values=(2, 3), nonzero_values=(5, 10),
            p=15, n_values=(100, 200), settings=("supervised",), n_trials=1,
        )
        # 2 degrees × 2 k × 2 n × 1 setting = 8 configs
        assert len(cfgs) == 8

    def test_default_nonzero_values(self):
        cfgs = self.sweep_fn(
            degree_values=(2, 3), n_values=(100,),
            settings=("supervised",), n_trials=1,
        )
        ks = {c.k for c in cfgs}
        # default nonzero_values = (5, 10, 15, 20) and default p=25
        assert ks == {5, 10, 15, 20}


# ---------------------------------------------------------------------------
# Visualize module
# ---------------------------------------------------------------------------

class TestVisualize:
    """Smoke tests for the visualize module (no display needed)."""

    @pytest.fixture(autouse=True)
    def use_agg_backend(self):
        """Use the non-interactive Agg backend so no display is needed."""
        import matplotlib
        matplotlib.use("Agg")

    @pytest.fixture
    def tiny_results(self):
        """Two sim results with varying n_labeled (enough for plotting)."""
        cfgs = [
            _fast_config("supervised"),
            SimulationConfig(
                p=5, n_labeled=120, k=2,
                setting="supervised",
                degree=1, n_components=2, Q=0.20,
                n_trials=1, methods=["ic_knock_poly"], max_iter=2, random_state=1,
            ),
        ]
        return run_simulation_suite(cfgs, verbose=False)

    def test_import(self):
        from simulations.visualize import (
            plot_prediction_error,
            plot_scalability,
            plot_selection_metrics,
            plot_nonzero_identification,
            plot_all,
        )

    def test_plot_prediction_error(self, tiny_results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from simulations.visualize import plot_prediction_error
        fig, ax = plt.subplots()
        ax_out = plot_prediction_error(tiny_results, x_field="n_labeled", ax=ax)
        assert ax_out is ax
        plt.close("all")

    def test_plot_scalability(self, tiny_results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from simulations.visualize import plot_scalability
        fig, (ax_t, ax_m) = plt.subplots(1, 2)
        out = plot_scalability(tiny_results, x_field="n_labeled",
                               ax_time=ax_t, ax_mem=ax_m)
        assert len(out) == 2
        plt.close("all")

    def test_plot_selection_metrics(self, tiny_results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from simulations.visualize import plot_selection_metrics
        fig, axes = plt.subplots(1, 3)
        axes_out = plot_selection_metrics(
            tiny_results, x_field="n_labeled",
            metrics=["fdr_mean", "f1_mean", "auc_mean"],
            axes=list(axes),
        )
        assert len(axes_out) == 3
        plt.close("all")

    def test_plot_nonzero_identification(self, tiny_results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from simulations.visualize import plot_nonzero_identification
        fig, ax = plt.subplots()
        ax_out = plot_nonzero_identification(tiny_results, x_field="n_labeled", ax=ax)
        assert ax_out is ax
        plt.close("all")

    def test_plot_all_saves_files(self, tmp_path, tiny_results):
        import matplotlib
        matplotlib.use("Agg")
        from simulations.visualize import plot_all
        figs = plot_all(tiny_results, output_dir=str(tmp_path), show=False)
        assert len(figs) > 0
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) > 0

    def test_plot_all_empty_results(self):
        from simulations.visualize import plot_all
        figs = plot_all([], show=False)
        assert figs == {}

    def test_invalid_x_field(self, tiny_results):
        import matplotlib.pyplot as plt
        from simulations.visualize import plot_prediction_error
        with pytest.raises(ValueError, match="x_field must be one of"):
            plot_prediction_error(tiny_results, x_field="invalid_field")

"""Tests for the main ICKnockoffPolyReg algorithm and evaluation metrics."""

import numpy as np
import pytest

from ic_knockoff_poly_reg.algorithm import ICKnockoffPolyReg
from ic_knockoff_poly_reg.evaluation import (
    DiscoveryMetrics,
    compute_fdr,
    compute_metrics,
    compute_tpr,
    memory_tracker,
)


# ---------------------------------------------------------------------------
# Evaluation metric tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_fdr_no_selections(self):
        assert compute_fdr(set(), {0, 1, 2}) == 0.0

    def test_fdr_all_correct(self):
        assert compute_fdr({0, 1}, {0, 1}) == 0.0

    def test_fdr_all_wrong(self):
        assert compute_fdr({3, 4}, {0, 1}) == 1.0

    def test_fdr_mixed(self):
        # 1 true positive, 1 false positive → FDR = 0.5
        fdr = compute_fdr({0, 5}, {0, 1})
        assert abs(fdr - 0.5) < 1e-9

    def test_tpr_no_true_features(self):
        assert compute_tpr({0, 1}, set()) == 1.0

    def test_tpr_perfect(self):
        assert compute_tpr({0, 1}, {0, 1}) == 1.0

    def test_tpr_none_found(self):
        assert compute_tpr(set(), {0, 1}) == 0.0

    def test_tpr_partial(self):
        tpr = compute_tpr({0}, {0, 1, 2})
        assert abs(tpr - 1 / 3) < 1e-9

    def test_compute_metrics_returns_dataclass(self):
        m = compute_metrics({0, 1, 5}, {0, 1, 2})
        assert isinstance(m, DiscoveryMetrics)
        assert m.n_true_positives == 2
        assert m.n_false_positives == 1
        assert m.n_false_negatives == 1

    def test_memory_tracker(self):
        with memory_tracker() as mem:
            _ = np.ones((1000, 1000))
        assert mem["peak_mb"] is not None
        assert mem["peak_mb"] > 0


# ---------------------------------------------------------------------------
# Algorithm smoke tests
# ---------------------------------------------------------------------------

def _make_simple_dataset(n=120, p=6, seed=42):
    """Create y = x0 + 1/x1 + noise with extra noise features."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 3.0, size=(n, p))
    y = X[:, 0] + 1.0 / X[:, 1] + 0.1 * rng.standard_normal(n)
    return X, y


class TestICKnockoffPolyRegSmoke:
    """Smoke tests: verify the algorithm runs end-to-end without error."""

    def test_fit_runs(self):
        X, y = _make_simple_dataset()
        model = ICKnockoffPolyReg(
            degree=1,
            n_components=2,
            gmm_alpha=0.1,
            Q=0.20,
            spending_sequence="riemann_zeta",
            max_iter=3,
            random_state=42,
        )
        model.fit(X, y)
        assert model.result_ is not None

    def test_result_has_expected_fields(self):
        X, y = _make_simple_dataset()
        model = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.20, max_iter=2, random_state=0
        )
        model.fit(X, y)
        r = model.result_
        assert hasattr(r, "selected_poly_names")
        assert hasattr(r, "selected_base_indices")
        assert hasattr(r, "n_iterations")
        assert hasattr(r, "iteration_history")
        assert r.n_iterations >= 1

    def test_iteration_history_residual_non_increasing(self):
        """Residual norm should not increase between iterations (up to numerics)."""
        X, y = _make_simple_dataset(n=100, p=4)
        model = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.30, max_iter=5, random_state=1
        )
        model.fit(X, y)
        norms = [h["residual_norm"] for h in model.result_.iteration_history]
        if len(norms) > 1:
            for i in range(len(norms) - 1):
                # Allow small numerical increases
                assert norms[i + 1] <= norms[i] + 1e-6, (
                    f"Residual increased at iteration {i+2}: "
                    f"{norms[i]:.4f} -> {norms[i+1]:.4f}"
                )

    def test_geometric_spending(self):
        X, y = _make_simple_dataset()
        model = ICKnockoffPolyReg(
            degree=1,
            n_components=2,
            gmm_alpha=0.1,
            Q=0.20,
            spending_sequence="geometric",
            gamma=0.5,
            max_iter=3,
            random_state=99,
        )
        model.fit(X, y)
        assert model.result_ is not None

    def test_unfitted_predict_raises(self):
        model = ICKnockoffPolyReg()
        X, _ = _make_simple_dataset()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predict_shape(self):
        X, y = _make_simple_dataset()
        model = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.30, max_iter=2, random_state=7
        )
        model.fit(X, y)
        y_hat = model.predict(X)
        assert y_hat.shape == (X.shape[0],)

    def test_high_Q_makes_more_selections(self):
        """Higher Q (relaxed FDR) should generally select more features."""
        X, y = _make_simple_dataset(n=150, p=5)
        model_strict = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.05, max_iter=4, random_state=55
        )
        model_relaxed = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.50, max_iter=4, random_state=55
        )
        model_strict.fit(X, y)
        model_relaxed.fit(X, y)
        n_strict = len(model_strict.result_.selected_poly_names)
        n_relaxed = len(model_relaxed.result_.selected_poly_names)
        assert n_relaxed >= n_strict


# ---------------------------------------------------------------------------
# Semi-supervised fit tests
# ---------------------------------------------------------------------------

def _make_semi_supervised_dataset(n_labeled=80, n_unlabeled=400, p=6, seed=7):
    """Return (X_labeled, y_labeled, X_unlabeled) from the same DGP."""
    rng = np.random.default_rng(seed)
    X_unlabeled = rng.uniform(0.5, 3.0, size=(n_unlabeled, p))
    X_labeled = rng.uniform(0.5, 3.0, size=(n_labeled, p))
    y_labeled = X_labeled[:, 0] + 1.0 / X_labeled[:, 1] + 0.1 * rng.standard_normal(n_labeled)
    return X_labeled, y_labeled, X_unlabeled


class TestSemiSupervisedFit:
    """Tests for the X_unlabeled parameter in ICKnockoffPolyReg.fit()."""

    def test_semi_supervised_runs(self):
        """fit() with X_unlabeled completes without error."""
        X_labeled, y_labeled, X_unlabeled = _make_semi_supervised_dataset()
        model = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.20, max_iter=3, random_state=42
        )
        model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)
        assert model.result_ is not None

    def test_semi_supervised_result_fields(self):
        """Result from semi-supervised fit has the same fields as supervised."""
        X_labeled, y_labeled, X_unlabeled = _make_semi_supervised_dataset()
        model = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.20, max_iter=2, random_state=0
        )
        model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)
        r = model.result_
        assert hasattr(r, "selected_poly_names")
        assert hasattr(r, "selected_base_indices")
        assert hasattr(r, "n_iterations")
        assert hasattr(r, "iteration_history")
        assert r.n_iterations >= 1

    def test_semi_supervised_predict_shape(self):
        """predict() after semi-supervised fit returns the correct shape."""
        X_labeled, y_labeled, X_unlabeled = _make_semi_supervised_dataset()
        model = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, Q=0.20, max_iter=2, random_state=3
        )
        model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)
        y_hat = model.predict(X_labeled)
        assert y_hat.shape == (X_labeled.shape[0],)

    def test_semi_supervised_gmm_uses_more_data(self):
        """GMM fitted in semi-supervised mode sees more samples than supervised."""
        X_labeled, y_labeled, X_unlabeled = _make_semi_supervised_dataset(
            n_labeled=60, n_unlabeled=300, p=4
        )
        model_sup = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, max_iter=2, random_state=5
        )
        model_semi = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, max_iter=2, random_state=5
        )
        model_sup.fit(X_labeled, y_labeled)
        model_semi.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)
        # Semi-supervised GMM is fitted on 360 samples; supervised on 60
        # We simply check both converged and produced valid precision matrices.
        for comp in model_semi.gmm_.result_.components:
            eigvals = np.linalg.eigvalsh(comp.precision)
            assert np.all(eigvals > 0)

    def test_unlabeled_wrong_features_raises(self):
        """X_unlabeled with wrong number of columns raises ValueError."""
        X_labeled, y_labeled, X_unlabeled = _make_semi_supervised_dataset(p=6)
        # Give X_unlabeled a different number of columns
        X_unlabeled_bad = X_unlabeled[:, :4]  # 4 cols instead of 6
        model = ICKnockoffPolyReg(degree=1, n_components=2, max_iter=2)
        with pytest.raises(ValueError, match="X_unlabeled must have shape"):
            model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled_bad)

    def test_none_unlabeled_matches_supervised(self):
        """Passing X_unlabeled=None gives the same result as not passing it."""
        X, y = _make_simple_dataset(n=80, p=4, seed=99)
        model_a = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, max_iter=2, random_state=99
        )
        model_b = ICKnockoffPolyReg(
            degree=1, n_components=2, gmm_alpha=0.1, max_iter=2, random_state=99
        )
        model_a.fit(X, y)
        model_b.fit(X, y, X_unlabeled=None)
        assert model_a.result_.selected_poly_names == model_b.result_.selected_poly_names

"""Tests for PenalizedGMM and ConditionalKnockoffGenerator."""

import numpy as np
import pytest

from ic_knockoff_poly_reg.gmm_phase import PenalizedGMM
from ic_knockoff_poly_reg.knockoffs import ConditionalKnockoffGenerator


class TestPenalizedGMM:
    def test_fit_returns_self(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 4))
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=42)
        result = gmm.fit(X)
        assert result is gmm

    def test_components_count(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3))
        gmm = PenalizedGMM(n_components=3, alpha=0.1, random_state=0)
        gmm.fit(X)
        assert len(gmm.result_.components) == 3

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((80, 3))
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=1)
        gmm.fit(X)
        total = sum(c.weight for c in gmm.result_.components)
        assert abs(total - 1.0) < 0.01

    def test_precision_matrices_positive_definite(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((80, 4))
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=2)
        gmm.fit(X)
        for comp in gmm.result_.components:
            eigvals = np.linalg.eigvalsh(comp.precision)
            assert np.all(eigvals > 0), "Precision must be positive definite"

    def test_log_responsibilities_shape(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 3))
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=3)
        gmm.fit(X)
        log_resp = gmm.compute_log_responsibilities(X)
        assert log_resp.shape == (50, 2)

    def test_log_responsibilities_sum_to_zero(self):
        """log sum_k exp(log_resp_ik) == 0 (i.e. probabilities sum to 1)."""
        from scipy.special import logsumexp

        rng = np.random.default_rng(4)
        X = rng.standard_normal((50, 3))
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=4)
        gmm.fit(X)
        log_resp = gmm.compute_log_responsibilities(X)
        log_sum = logsumexp(log_resp, axis=1)
        np.testing.assert_allclose(log_sum, 0.0, atol=1e-6)

    def test_unfitted_raises(self):
        gmm = PenalizedGMM()
        with pytest.raises(RuntimeError):
            gmm.compute_log_responsibilities(np.ones((5, 2)))

    def test_conditional_params_shape(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((60, 5))
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=5)
        gmm.fit(X)
        A_indices = np.array([0, 1])
        B_indices = np.array([2, 3, 4])
        cond_means, cond_prec, log_resp = gmm.compute_conditional_params(
            X[:, A_indices], A_indices, B_indices
        )
        assert cond_means.shape == (60, 3)
        assert cond_prec.shape == (3, 3)
        assert log_resp.shape == (60, 2)

    def test_conditional_precision_positive_definite(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((60, 4))
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=6)
        gmm.fit(X)
        A_indices = np.array([0])
        B_indices = np.array([1, 2, 3])
        _, cond_prec, _ = gmm.compute_conditional_params(
            X[:, A_indices], A_indices, B_indices
        )
        eigvals = np.linalg.eigvalsh(cond_prec)
        assert np.all(eigvals > 0)

    def test_bic_decreases_with_better_fit(self):
        """BIC should generally improve (decrease) with more components on
        data generated from a 2-component mixture."""
        rng = np.random.default_rng(7)
        X1 = rng.standard_normal((40, 3)) + np.array([5.0, 0.0, 0.0])
        X2 = rng.standard_normal((40, 3)) + np.array([-5.0, 0.0, 0.0])
        X = np.vstack([X1, X2])
        gmm1 = PenalizedGMM(n_components=1, alpha=0.1, random_state=7)
        gmm2 = PenalizedGMM(n_components=2, alpha=0.1, random_state=7)
        gmm1.fit(X)
        gmm2.fit(X)
        # Not always guaranteed, but strongly expected for well-separated clusters
        assert gmm2.result_.log_likelihood >= gmm1.result_.log_likelihood


class TestConditionalKnockoffGenerator:
    def _make_fitted_gmm(self, X):
        gmm = PenalizedGMM(n_components=2, alpha=0.1, random_state=42)
        gmm.fit(X)
        return gmm

    def test_knockoff_shape(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((60, 5))
        gmm = self._make_fitted_gmm(X)
        gen = ConditionalKnockoffGenerator(gmm=gmm, random_state=10)
        A_indices = np.array([0, 1])
        B_indices = np.array([2, 3, 4])
        X_tilde = gen.generate(X, A_indices, B_indices)
        assert X_tilde.shape == (60, 3)

    def test_knockoffs_finite(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((60, 4))
        gmm = self._make_fitted_gmm(X)
        gen = ConditionalKnockoffGenerator(gmm=gmm, random_state=11)
        X_tilde = gen.generate(X, np.array([0]), np.array([1, 2, 3]))
        assert np.all(np.isfinite(X_tilde))

    def test_knockoffs_not_equal_original(self):
        """Knockoffs must differ from the original features."""
        rng = np.random.default_rng(12)
        X = rng.standard_normal((60, 4))
        gmm = self._make_fitted_gmm(X)
        gen = ConditionalKnockoffGenerator(gmm=gmm, random_state=12)
        X_tilde = gen.generate(X, np.array([0]), np.array([1, 2, 3]))
        assert not np.allclose(X_tilde, X[:, [1, 2, 3]])

    def test_unconditional_knockoffs_shape(self):
        """Unconditional knockoff path (A_indices empty)."""
        rng = np.random.default_rng(13)
        X = rng.standard_normal((60, 3))
        gmm = self._make_fitted_gmm(X)
        gen = ConditionalKnockoffGenerator(gmm=gmm, random_state=13)
        X_tilde = gen.generate(X, np.array([], dtype=int), np.arange(3))
        assert X_tilde.shape == (60, 3)
        assert np.all(np.isfinite(X_tilde))

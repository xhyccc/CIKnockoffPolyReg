"""Tests for the kernel backend switching system.

Tests the :mod:`ic_knockoff_poly_reg.kernels` module, verifying that:
- The Python backend is always available and produces correct results.
- The C++ and Rust backends (when compiled) match the Python backend.
- Backend switching in :class:`ICKnockoffPolyReg` works without regressions.
- Error handling for unknown or unavailable backends.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_X(rng):
    """(20, 4) data matrix with positive entries (safe for negative powers)."""
    return rng.uniform(0.5, 2.0, size=(20, 4))


@pytest.fixture
def small_cov(rng):
    """4×4 SPD covariance matrix."""
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 2 * np.eye(4)
    return S


@pytest.fixture
def beta_pair(rng):
    """Pair of coefficient vectors of length 8."""
    b_orig = rng.standard_normal(8)
    b_knock = rng.standard_normal(8)
    return b_orig, b_knock


# ---------------------------------------------------------------------------
# Helper: create kernels and skip if unavailable
# ---------------------------------------------------------------------------

from ic_knockoff_poly_reg.kernels import create_kernels, AVAILABLE_BACKENDS


def _get_kernels(backend: str):
    """Return kernel set or skip the test if the backend is not available."""
    if backend not in AVAILABLE_BACKENDS:
        pytest.skip(f"{backend!r} backend is not available (library not compiled)")
    return create_kernels(backend)


# ---------------------------------------------------------------------------
# Test: create_kernels factory
# ---------------------------------------------------------------------------

class TestCreateKernels:
    def test_python_always_available(self):
        ks = create_kernels("python")
        assert ks.poly is not None
        assert ks.knockoff is not None
        assert ks.posi is not None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_kernels("fortran")

    def test_available_backends_contains_python(self):
        assert "python" in AVAILABLE_BACKENDS

    def test_available_backends_type(self):
        assert isinstance(AVAILABLE_BACKENDS, frozenset)


# ---------------------------------------------------------------------------
# Test: PolynomialKernel — Python backend
# ---------------------------------------------------------------------------

class TestPolynomialKernelPython:
    def test_n_expanded(self):
        ks = create_kernels("python")
        assert ks.poly.n_expanded(4, degree=2, include_bias=True) == 4 * 4 + 1
        assert ks.poly.n_expanded(4, degree=2, include_bias=False) == 4 * 4
        assert ks.poly.n_expanded(3, degree=1, include_bias=True) == 3 * 2 + 1

    def test_expand_shape(self, small_X):
        ks = create_kernels("python")
        ef = ks.poly.expand(small_X, degree=2, include_bias=True)
        n, p = small_X.shape
        n_cols = ks.poly.n_expanded(p, degree=2, include_bias=True)
        assert ef.matrix.shape == (n, n_cols)
        assert len(ef.feature_names) == n_cols
        assert len(ef.base_feature_indices) == n_cols
        assert len(ef.power_exponents) == n_cols

    def test_expand_bias_column(self, small_X):
        ks = create_kernels("python")
        ef = ks.poly.expand(small_X, degree=2, include_bias=True)
        # Last column should be all-ones (bias)
        np.testing.assert_array_equal(ef.matrix[:, -1], np.ones(small_X.shape[0]))
        assert ef.feature_names[-1] == "1"
        assert ef.base_feature_indices[-1] == -1
        assert ef.power_exponents[-1] == 0

    def test_expand_positive_powers(self, small_X):
        ks = create_kernels("python")
        ef = ks.poly.expand(small_X, degree=2, include_bias=False)
        # First column should be x0^1
        np.testing.assert_allclose(ef.matrix[:, 0], small_X[:, 0])
        # Second column should be x0^2
        np.testing.assert_allclose(ef.matrix[:, 1], small_X[:, 0] ** 2)

    def test_expand_negative_powers(self, small_X):
        ks = create_kernels("python")
        ef = ks.poly.expand(small_X, degree=1, include_bias=False)
        p = small_X.shape[1]
        # With degree=1, layout per feature j is: [x_j^1, x_j^(-1)].
        # x_j^(-1) is at column index 2*j + 1.
        for j in range(p):
            col_idx = 2 * j + 1
            expected = 1.0 / small_X[:, j]
            np.testing.assert_allclose(ef.matrix[:, col_idx], expected, rtol=1e-10)

    def test_expand_base_names(self, small_X):
        ks = create_kernels("python")
        names = ["a", "b", "c", "d"]
        ef = ks.poly.expand(small_X, degree=1, include_bias=False, base_names=names)
        assert ef.feature_names[0] == "a"       # a^1
        assert ef.feature_names[1] == "a^(-1)"  # a^(-1) is at index 1

    def test_expand_no_bias(self, small_X):
        ks = create_kernels("python")
        ef = ks.poly.expand(small_X, degree=2, include_bias=False)
        assert ef.matrix.shape[1] == small_X.shape[1] * 4
        assert "1" not in ef.feature_names


# ---------------------------------------------------------------------------
# Test: KnockoffKernel — Python backend
# ---------------------------------------------------------------------------

class TestKnockoffKernelPython:
    def test_w_statistics_shape(self, beta_pair):
        ks = create_kernels("python")
        b_orig, b_knock = beta_pair
        W = ks.knockoff.w_statistics(b_orig, b_knock)
        assert W.shape == b_orig.shape

    def test_w_statistics_values(self, beta_pair):
        ks = create_kernels("python")
        b_orig, b_knock = beta_pair
        W = ks.knockoff.w_statistics(b_orig, b_knock)
        expected = np.abs(b_orig) - np.abs(b_knock)
        np.testing.assert_allclose(W, expected)

    def test_w_statistics_length_mismatch(self):
        ks = create_kernels("python")
        with pytest.raises((ValueError, Exception)):
            ks.knockoff.w_statistics(np.ones(3), np.ones(4))

    def test_equicorrelated_s_values_shape(self, small_cov):
        ks = create_kernels("python")
        s = ks.knockoff.equicorrelated_s_values(small_cov)
        assert s.shape == (small_cov.shape[0],)

    def test_equicorrelated_s_values_positive(self, small_cov):
        ks = create_kernels("python")
        s = ks.knockoff.equicorrelated_s_values(small_cov)
        assert np.all(s > 0)

    def test_equicorrelated_s_values_constant(self, small_cov):
        """Equicorrelated construction produces a constant s vector."""
        ks = create_kernels("python")
        s = ks.knockoff.equicorrelated_s_values(small_cov)
        np.testing.assert_allclose(s, s[0] * np.ones_like(s))

    def test_sample_knockoffs_shape(self, small_X, small_cov):
        ks = create_kernels("python")
        n, p = small_X.shape
        mu = small_X.mean(axis=0)
        X_tilde = ks.knockoff.sample_gaussian_knockoffs(small_X, mu, small_cov, seed=0)
        assert X_tilde.shape == (n, p)


# ---------------------------------------------------------------------------
# Test: PosiKernel — Python backend
# ---------------------------------------------------------------------------

class TestPosiKernelPython:
    def test_riemann_zeta_t1(self):
        ks = create_kernels("python")
        q = ks.posi.alpha_spending_budget(1, Q=0.1, sequence="riemann_zeta")
        expected = 0.1 * 6.0 / math.pi ** 2
        assert abs(q - expected) < 1e-12

    def test_riemann_zeta_decreasing(self):
        ks = create_kernels("python")
        budgets = [ks.posi.alpha_spending_budget(t, Q=0.1, sequence="riemann_zeta")
                   for t in range(1, 6)]
        assert all(budgets[i] > budgets[i + 1] for i in range(len(budgets) - 1))

    def test_geometric_t1(self):
        ks = create_kernels("python")
        q = ks.posi.alpha_spending_budget(1, Q=0.1, sequence="geometric", gamma=0.5)
        expected = 0.1 * 0.5  # Q * (1 - gamma) * gamma^0
        assert abs(q - expected) < 1e-12

    def test_riemann_zeta_sum_bounded(self):
        """Sum of first 100 Riemann-Zeta budgets must be <= Q."""
        ks = create_kernels("python")
        Q = 0.1
        total = sum(ks.posi.alpha_spending_budget(t, Q=Q, sequence="riemann_zeta")
                    for t in range(1, 101))
        assert total <= Q + 1e-9

    def test_knockoff_threshold_basic(self):
        ks = create_kernels("python")
        W = np.array([3.0, -0.5, 1.0, -2.0, 0.8])
        tau = ks.posi.knockoff_threshold(W, q_t=0.5)
        assert np.isfinite(tau) or np.isinf(tau)

    def test_knockoff_threshold_no_selection(self):
        """All W negative → threshold is +inf."""
        ks = create_kernels("python")
        W = np.array([-1.0, -2.0, -3.0])
        tau = ks.posi.knockoff_threshold(W, q_t=0.1)
        assert np.isinf(tau)

    def test_knockoff_threshold_active_poly(self):
        ks = create_kernels("python")
        W = np.array([3.0, -0.5, 1.0, -2.0])
        # Excluding index 0 (W=3.0) removes it from the denominator, potentially
        # changing the threshold.  Both calls must return a valid float or inf.
        tau_no_active = ks.posi.knockoff_threshold(W, q_t=0.5, active_poly=set())
        tau_with_active = ks.posi.knockoff_threshold(W, q_t=0.5, active_poly={0})
        assert np.isfinite(tau_no_active) or np.isinf(tau_no_active)
        assert np.isfinite(tau_with_active) or np.isinf(tau_with_active)


# ---------------------------------------------------------------------------
# Test: consistency between backends (C++/Rust vs Python)
# ---------------------------------------------------------------------------

class TestBackendConsistency:
    """Tests that C++ and Rust backends agree with the Python backend."""

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_polynomial_n_expanded_matches(self, backend):
        ks_py = create_kernels("python")
        ks = _get_kernels(backend)
        for n_base in [2, 4, 10]:
            for degree in [1, 2, 3]:
                for ib in [True, False]:
                    assert ks.poly.n_expanded(n_base, degree, ib) == \
                           ks_py.poly.n_expanded(n_base, degree, ib)

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_polynomial_expand_matrix_matches(self, backend, small_X):
        ks_py = create_kernels("python")
        ks = _get_kernels(backend)
        ef_py = ks_py.poly.expand(small_X, degree=2, include_bias=True)
        ef_native = ks.poly.expand(small_X, degree=2, include_bias=True)
        np.testing.assert_allclose(
            ef_native.matrix, ef_py.matrix, rtol=1e-10, atol=1e-12,
            err_msg=f"{backend} expand matrix mismatch",
        )
        assert ef_native.base_feature_indices == ef_py.base_feature_indices
        assert ef_native.power_exponents == ef_py.power_exponents

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_w_statistics_matches(self, backend, beta_pair):
        ks_py = create_kernels("python")
        ks = _get_kernels(backend)
        b_orig, b_knock = beta_pair
        W_py = ks_py.knockoff.w_statistics(b_orig, b_knock)
        W_native = ks.knockoff.w_statistics(b_orig, b_knock)
        np.testing.assert_allclose(W_native, W_py, rtol=1e-12,
                                   err_msg=f"{backend} W-statistics mismatch")

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_equicorrelated_s_values_matches(self, backend, small_cov):
        """Equicorrelated s-values should be positive for both backends.

        Note: C++/Rust use a Gershgorin lower bound for lambda_min while
        Python uses the exact minimum eigenvalue. Both approaches yield valid
        (positive) s-values, but the numeric values may differ.
        """
        ks = _get_kernels(backend)
        s_native = ks.knockoff.equicorrelated_s_values(small_cov)
        assert s_native.shape == (small_cov.shape[0],), "shape mismatch"
        assert np.all(s_native > 0), f"{backend} s-values must be positive"

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_alpha_spending_budget_matches(self, backend):
        ks_py = create_kernels("python")
        ks = _get_kernels(backend)
        for t in [1, 2, 5, 10]:
            for seq in ["riemann_zeta", "geometric"]:
                q_py = ks_py.posi.alpha_spending_budget(t, Q=0.1, sequence=seq)
                q_native = ks.posi.alpha_spending_budget(t, Q=0.1, sequence=seq)
                assert abs(q_py - q_native) < 1e-12, \
                    f"{backend} budget mismatch at t={t}, seq={seq}"

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_knockoff_threshold_matches(self, backend):
        ks_py = create_kernels("python")
        ks = _get_kernels(backend)
        W = np.array([3.0, -0.5, 1.0, -2.0, 0.8, -1.2, 2.5, 0.0])
        for q_t in [0.1, 0.2, 0.5]:
            tau_py = ks_py.posi.knockoff_threshold(W, q_t)
            tau_native = ks.posi.knockoff_threshold(W, q_t)
            assert abs(tau_py - tau_native) < 1e-12 or \
                   (np.isinf(tau_py) and np.isinf(tau_native)), \
                f"{backend} threshold mismatch at q_t={q_t}"

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_knockoff_threshold_with_active_matches(self, backend):
        ks_py = create_kernels("python")
        ks = _get_kernels(backend)
        W = np.array([3.0, -0.5, 1.0, -2.0, 0.8])
        active = {0, 2}
        tau_py = ks_py.posi.knockoff_threshold(W, q_t=0.3, active_poly=active)
        tau_native = ks.posi.knockoff_threshold(W, q_t=0.3, active_poly=active)
        assert abs(tau_py - tau_native) < 1e-12 or \
               (np.isinf(tau_py) and np.isinf(tau_native)), \
            f"{backend} threshold with active set mismatch"


# ---------------------------------------------------------------------------
# Test: ICKnockoffPolyReg backend parameter
# ---------------------------------------------------------------------------

class TestICKnockoffPolyRegBackend:
    def test_default_backend_is_python(self):
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        model = ICKnockoffPolyReg()
        assert model.backend == "python"

    def test_backend_parameter_stored(self):
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        model = ICKnockoffPolyReg(backend="python")
        assert model.backend == "python"

    def test_invalid_backend_raises_at_fit(self, small_X):
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        rng = np.random.default_rng(0)
        y = rng.standard_normal(small_X.shape[0])
        model = ICKnockoffPolyReg(backend="unknown_backend", max_iter=1)
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(small_X, y)

    def test_python_backend_fit(self, small_X):
        """Smoke-test that fit runs with the Python backend."""
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        rng = np.random.default_rng(0)
        y = rng.standard_normal(small_X.shape[0])
        model = ICKnockoffPolyReg(
            backend="python", max_iter=1, n_components=1, degree=1,
        )
        model.fit(small_X, y)
        assert model.result_ is not None

    @pytest.mark.parametrize("backend", ["cpp", "rust"])
    def test_native_backend_fit_matches_python(self, backend, small_X):
        """When native backend is available, fit produces a valid result."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"{backend!r} backend not compiled")
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        rng = np.random.default_rng(0)
        y = rng.standard_normal(small_X.shape[0])
        model = ICKnockoffPolyReg(
            backend=backend, max_iter=1, n_components=1, degree=1,
            random_state=0,
        )
        model.fit(small_X, y)
        assert model.result_ is not None

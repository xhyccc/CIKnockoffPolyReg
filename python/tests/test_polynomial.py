"""Tests for PolynomialDictionary expansion."""

import math

import numpy as np
import pytest

from ic_knockoff_poly_reg.polynomial import PolynomialDictionary


class TestPolynomialDictionary:
    def test_basic_expansion_shape(self):
        """Each feature produces 2*degree columns; +1 if bias included."""
        X = np.random.default_rng(0).standard_normal((50, 3))
        poly = PolynomialDictionary(degree=2, include_bias=True)
        result = poly.expand(X)
        # 3 features * 2 * 2 powers + 1 bias = 13
        assert result.matrix.shape == (50, 13)
        assert len(result.feature_names) == 13
        assert len(result.power_exponents) == 13

    def test_no_bias(self):
        X = np.ones((10, 2))
        poly = PolynomialDictionary(degree=1, include_bias=False)
        result = poly.expand(X)
        assert result.matrix.shape == (10, 4)  # 2 features * 2 powers

    def test_positive_powers_correct(self):
        X = np.array([[2.0, 3.0]])
        poly = PolynomialDictionary(degree=2, include_bias=False)
        result = poly.expand(X)
        names = result.feature_names
        exps = result.power_exponents
        # x0^1, x0^2, x0^{-1}, x0^{-2}, x1^1, x1^2, x1^{-1}, x1^{-2}
        row = result.matrix[0]
        assert math.isclose(row[names.index("x0")], 2.0)
        assert math.isclose(row[names.index("x0^2")], 4.0)
        assert math.isclose(row[names.index("x0^(-1)")], 0.5)
        assert math.isclose(row[names.index("x0^(-2)")], 0.25)
        assert math.isclose(row[names.index("x1")], 3.0)

    def test_negative_power_clipping(self):
        """Very small values must not produce infinities."""
        X = np.array([[0.0, 1e-15]])
        poly = PolynomialDictionary(degree=1, include_bias=False)
        result = poly.expand(X)
        assert np.all(np.isfinite(result.matrix))

    def test_bias_column_is_ones(self):
        X = np.random.default_rng(1).standard_normal((20, 2))
        poly = PolynomialDictionary(degree=1, include_bias=True)
        result = poly.expand(X)
        # Last column should be the bias (all ones)
        assert result.feature_names[-1] == "1"
        assert np.allclose(result.matrix[:, -1], 1.0)

    def test_custom_base_names(self):
        X = np.ones((5, 2))
        poly = PolynomialDictionary(degree=1, include_bias=False)
        result = poly.expand(X, base_names=["foo", "bar"])
        assert "foo" in result.feature_names
        assert "bar" in result.feature_names

    def test_base_names_length_mismatch_raises(self):
        X = np.ones((5, 2))
        poly = PolynomialDictionary(degree=1)
        with pytest.raises(ValueError):
            poly.expand(X, base_names=["only_one"])

    def test_n_expanded_features(self):
        poly = PolynomialDictionary(degree=3, include_bias=True)
        assert poly.n_expanded_features(4) == 4 * 2 * 3 + 1

    def test_degree_zero_raises(self):
        with pytest.raises(ValueError):
            PolynomialDictionary(degree=0)

    def test_base_feature_indices_correct(self):
        X = np.ones((5, 3))
        poly = PolynomialDictionary(degree=2, include_bias=False)
        result = poly.expand(X)
        # Each base feature generates 4 columns (2 pos + 2 neg powers)
        for j in range(3):
            indices = [i for i, b in enumerate(result.base_feature_indices) if b == j]
            assert len(indices) == 4

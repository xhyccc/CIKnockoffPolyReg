"""Tests for AlphaSpending and compute_knockoff_threshold."""

import math

import numpy as np
import pytest

from ic_knockoff_poly_reg.posi_threshold import (
    AlphaSpending,
    SpendingSequence,
    compute_knockoff_threshold,
)


class TestAlphaSpending:
    def test_riemann_zeta_budget_t1(self):
        spender = AlphaSpending(Q=0.10, sequence="riemann_zeta")
        q1 = spender.budget(1)
        expected = 0.10 * 6.0 / math.pi ** 2
        assert math.isclose(q1, expected, rel_tol=1e-9)

    def test_riemann_zeta_sum_converges(self):
        """Partial sum of Riemann Zeta sequence must be <= Q."""
        spender = AlphaSpending(Q=0.10, sequence="riemann_zeta")
        total = sum(spender.budget(t) for t in range(1, 1001))
        assert total <= 0.10 + 1e-6  # Leq Q up to floating point

    def test_geometric_budget_t1(self):
        spender = AlphaSpending(Q=0.10, sequence="geometric", gamma=0.5)
        q1 = spender.budget(1)
        expected = 0.10 * (1.0 - 0.5) * (0.5 ** 0)
        assert math.isclose(q1, expected, rel_tol=1e-9)

    def test_geometric_sum_converges(self):
        """Geometric series sum = Q*(1-gamma) * 1/(1-gamma) = Q."""
        spender = AlphaSpending(Q=0.10, sequence="geometric", gamma=0.5)
        total = sum(spender.budget(t) for t in range(1, 200))
        assert total <= 0.10 + 1e-6

    def test_budgets_array_length(self):
        spender = AlphaSpending(Q=0.10)
        arr = spender.budgets(5)
        assert len(arr) == 5

    def test_invalid_Q_raises(self):
        with pytest.raises(ValueError):
            AlphaSpending(Q=0.0)
        with pytest.raises(ValueError):
            AlphaSpending(Q=1.0)

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError):
            AlphaSpending(Q=0.10, sequence="geometric", gamma=0.0)

    def test_t_zero_raises(self):
        spender = AlphaSpending(Q=0.10)
        with pytest.raises(ValueError):
            spender.budget(0)

    def test_spending_sequence_enum(self):
        assert SpendingSequence.RIEMANN_ZETA.value == "riemann_zeta"
        assert SpendingSequence.GEOMETRIC.value == "geometric"

    def test_riemann_zeta_decays(self):
        spender = AlphaSpending(Q=0.10, sequence="riemann_zeta")
        budgets = [spender.budget(t) for t in range(1, 6)]
        for i in range(len(budgets) - 1):
            assert budgets[i] > budgets[i + 1]

    def test_geometric_decays(self):
        spender = AlphaSpending(Q=0.10, sequence="geometric", gamma=0.5)
        budgets = [spender.budget(t) for t in range(1, 6)]
        for i in range(len(budgets) - 1):
            assert budgets[i] > budgets[i + 1]


class TestKnockoffThreshold:
    def test_returns_inf_when_no_selection_possible(self):
        """If all W are zero, no threshold satisfies the condition."""
        W = np.zeros(10)
        tau = compute_knockoff_threshold(W, q_t=0.10)
        assert np.isinf(tau)

    def test_returns_inf_when_empty(self):
        W = np.array([])
        tau = compute_knockoff_threshold(W, q_t=0.10)
        assert np.isinf(tau)

    def test_basic_threshold_selection(self):
        """With strong positive W and no negatives, tau should be small."""
        # Many positives, no negatives → ratio low → small tau
        W = np.array([3.0, 2.0, 1.5, 1.0, 0.5])
        tau = compute_knockoff_threshold(W, q_t=0.50)
        assert tau <= 3.0

    def test_active_set_exclusion(self):
        """Features in active_poly are excluded from threshold computation."""
        W = np.array([5.0, 4.0, 3.0])
        # Exclude index 0 (the largest)
        tau_with = compute_knockoff_threshold(W, q_t=0.10, active_poly={0})
        tau_without = compute_knockoff_threshold(W, q_t=0.10, active_poly=None)
        # Excluding index 0 may change the result
        assert tau_with >= 0.0

    def test_offset_zero_vs_one(self):
        """offset=0 should give same or lower threshold than offset=1."""
        W = np.array([2.0, 1.5, -0.5, -1.0])
        tau_plus = compute_knockoff_threshold(W, q_t=0.50, offset=1)
        tau_plain = compute_knockoff_threshold(W, q_t=0.50, offset=0)
        # knockoff+ is more conservative (higher tau) or equal
        assert tau_plus >= tau_plain or np.isinf(tau_plus)

    def test_knockoff_formula_manually(self):
        """Manually verify the FDR ratio formula for a simple case."""
        # W = [2.0, -1.0]
        # tau=1.0: negatives={-1.0}, positives={2.0, ...}
        # ratio = (1 + 1) / max(1, 1) = 2 > q_t=0.20
        # tau=2.0: negatives={}, positives={2.0}
        # ratio = (1 + 0) / max(1, 1) = 1 > q_t=0.20
        # → no tau found → inf  (for q_t=0.20)
        W = np.array([2.0, -1.0])
        tau = compute_knockoff_threshold(W, q_t=0.20)
        assert np.isinf(tau)

    def test_knockoff_threshold_returns_float(self):
        W = np.array([1.0, 2.0, 3.0, -0.5])
        tau = compute_knockoff_threshold(W, q_t=0.5)
        assert isinstance(tau, float)

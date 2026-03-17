/**
 * @file posi.hpp
 * @brief PoSI alpha-spending sequences and knockoff+ threshold computation.
 *
 * Implements the FDR control machinery described in Section 3 of the
 * IC-Knock-Poly paper:
 *
 *   1. Alpha-spending budgets q_t that satisfy sum_{t>=1} q_t <= Q:
 *      - Riemann Zeta:  q_t = Q * 6 / (pi^2 * t^2)
 *      - Geometric:     q_t = Q * (1-gamma) * gamma^(t-1)
 *
 *   2. The knockoff+ threshold:
 *      tau_t = min{ tau > 0 :
 *          (1 + |{j not in A_poly : W_j <= -tau}|)
 *          / max(1, |{j not in A_poly : W_j >= tau}|)  <= q_t }
 */

#pragma once

#include <cstddef>
#include <limits>
#include <set>
#include <vector>

namespace ic_knockoff {

// ---------------------------------------------------------------------------
// Alpha-spending sequences
// ---------------------------------------------------------------------------

/// Available alpha-spending sequence types.
enum class SpendingSequence { RiemannZeta, Geometric };

/**
 * @brief Return the FDR budget q_t for iteration t (1-indexed).
 *
 * @param t         Iteration index (>= 1).
 * @param Q         Global target FDR level in (0, 1).
 * @param sequence  Spending sequence type.
 * @param gamma     Geometric decay (only used for Geometric sequence).
 * @return Budget q_t.
 * @throws std::invalid_argument if t < 1, Q not in (0,1), or gamma not in (0,1).
 */
double alpha_spending_budget(int t, double Q,
                             SpendingSequence sequence = SpendingSequence::RiemannZeta,
                             double gamma = 0.5);

/**
 * @brief Compute the first max_t budget values q_1, ..., q_{max_t}.
 */
std::vector<double> alpha_spending_budgets(int max_t, double Q,
                                           SpendingSequence sequence,
                                           double gamma = 0.5);

// ---------------------------------------------------------------------------
// Knockoff+ threshold computation
// ---------------------------------------------------------------------------

/**
 * @brief Compute the knockoff+ threshold tau_t.
 *
 * tau_t = min{ tau in candidates :
 *     (offset + |{j not in active_poly : W[j] <= -tau}|)
 *     / max(1, |{j not in active_poly : W[j] >= tau}|) <= q_t }
 *
 * Returns +infinity if no such tau exists (no features selected).
 *
 * @param W           W-statistic vector for all candidates.
 * @param q_t         Current iteration FDR budget.
 * @param active_poly Set of indices already in the active polynomial set
 *                    (excluded from the threshold computation).
 * @param offset      1 for knockoff+ (default), 0 for standard knockoff.
 * @return tau_t, or std::numeric_limits<double>::infinity() if none found.
 */
double knockoff_threshold(const std::vector<double>& W,
                          double q_t,
                          const std::set<std::size_t>& active_poly = {},
                          int offset = 1);

} // namespace ic_knockoff

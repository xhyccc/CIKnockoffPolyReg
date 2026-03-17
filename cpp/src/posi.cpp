#include "ic_knockoff/posi.hpp"
#include <algorithm>
#include <cmath>

// pi^2 computed at file scope to avoid repeating the expression
namespace {
    // Use 4*atan(1)*4*atan(1) = pi^2 for portability across all C++ standards
    const double kPi2 = std::acos(-1.0) * std::acos(-1.0);
}
#include <limits>
#include <stdexcept>

namespace ic_knockoff {

double alpha_spending_budget(int t, double Q, SpendingSequence sequence, double gamma) {
    if (t < 1)
        throw std::invalid_argument("alpha_spending_budget: t must be >= 1");
    if (Q <= 0.0 || Q >= 1.0)
        throw std::invalid_argument("alpha_spending_budget: Q must be in (0,1)");
    if (sequence == SpendingSequence::Geometric) {
        if (gamma <= 0.0 || gamma >= 1.0)
            throw std::invalid_argument(
                "alpha_spending_budget: gamma must be in (0,1) for Geometric sequence");
        return Q * (1.0 - gamma) * std::pow(gamma, static_cast<double>(t - 1));
    }
    // Riemann Zeta: q_t = Q * 6 / (pi^2 * t^2)
    const double pi2 = kPi2;
    return Q * 6.0 / (pi2 * static_cast<double>(t) * static_cast<double>(t));
}

std::vector<double> alpha_spending_budgets(int max_t, double Q,
                                           SpendingSequence sequence, double gamma) {
    std::vector<double> budgets(static_cast<std::size_t>(max_t));
    for (int t = 1; t <= max_t; ++t)
        budgets[static_cast<std::size_t>(t - 1)] =
            alpha_spending_budget(t, Q, sequence, gamma);
    return budgets;
}

double knockoff_threshold(const std::vector<double>& W,
                          double q_t,
                          const std::set<std::size_t>& active_poly,
                          int offset) {
    // Collect candidate W values (excluding active_poly indices)
    std::vector<double> W_cand;
    W_cand.reserve(W.size());
    for (std::size_t j = 0; j < W.size(); ++j) {
        if (active_poly.find(j) == active_poly.end())
            W_cand.push_back(W[j]);
    }
    if (W_cand.empty())
        return std::numeric_limits<double>::infinity();

    // Collect unique positive |W| values as threshold candidates
    std::vector<double> candidates;
    for (double w : W_cand)
        if (w != 0.0) candidates.push_back(std::abs(w));
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    if (candidates.empty())
        return std::numeric_limits<double>::infinity();

    for (double tau : candidates) {
        int n_neg = 0, n_pos = 0;
        for (double w : W_cand) {
            if (w <= -tau) ++n_neg;
            if (w >= tau) ++n_pos;
        }
        double ratio = static_cast<double>(offset + n_neg) /
                       std::max(1, n_pos);
        if (ratio <= q_t)
            return tau;
    }
    return std::numeric_limits<double>::infinity();
}

} // namespace ic_knockoff

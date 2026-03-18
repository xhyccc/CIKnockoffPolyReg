/**
 * @file py_bindings.cpp
 * @brief C-ABI wrapper implementations for Python ctypes integration.
 *
 * Thin shims that convert between flat C arrays and the IC-Knock-Poly C++
 * types (Matrix, Vec), then delegate to the existing library functions.
 */

#include "py_bindings.hpp"

#include "ic_knockoff/knockoffs.hpp"
#include "ic_knockoff/matrix_ops.hpp"
#include "ic_knockoff/polynomial.hpp"
#include "ic_knockoff/posi.hpp"

#include <algorithm>
#include <set>
#include <vector>

using namespace ic_knockoff;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Matrix flat_to_matrix(const double* data, int rows, int cols) {
    Matrix M((std::size_t)rows, (std::size_t)cols, 0.0);
    std::copy(data, data + rows * cols, M.data.begin());
    return M;
}

// ---------------------------------------------------------------------------
// Polynomial expansion
// ---------------------------------------------------------------------------

extern "C" {

int ic_poly_n_expanded(int n_base, int degree, int include_bias) {
    return (int)n_expanded_features((std::size_t)n_base, degree,
                                    include_bias != 0);
}

int ic_poly_expand(
    const double* X_flat, int n, int p,
    int degree, int include_bias, double clip_threshold,
    double* out_matrix,
    int*    out_base_indices,
    int*    out_exponents)
{
    if (!X_flat || !out_matrix || !out_base_indices || !out_exponents)
        return -1;

    Matrix X = flat_to_matrix(X_flat, n, p);

    ExpandedFeatures ef = polynomial_expand(
        X, degree, include_bias != 0, clip_threshold, {});

    int n_cols = (int)ef.info.size();

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n_cols; ++j)
            out_matrix[i * n_cols + j] = ef.matrix(i, j);

    for (int j = 0; j < n_cols; ++j) {
        out_base_indices[j] = ef.info[j].base_feature_index;
        out_exponents[j]    = ef.info[j].exponent;
    }

    return n_cols;
}

// ---------------------------------------------------------------------------
// Knockoff W-statistics
// ---------------------------------------------------------------------------

void ic_compute_w_statistics(
    const double* beta_orig, const double* beta_knock, int p,
    double* out_w)
{
    Vec b_orig(beta_orig, beta_orig + p);
    Vec b_knock(beta_knock, beta_knock + p);
    Vec w = compute_w_statistics(b_orig, b_knock);
    std::copy(w.begin(), w.end(), out_w);
}

// ---------------------------------------------------------------------------
// Equicorrelated s-values
// ---------------------------------------------------------------------------

void ic_equicorrelated_s_values(
    const double* cov_flat, int p, double reg,
    double* out_s)
{
    Matrix Sigma = flat_to_matrix(cov_flat, p, p);
    Vec s = equicorrelated_s_values(Sigma, reg);
    std::copy(s.begin(), s.end(), out_s);
}

// ---------------------------------------------------------------------------
// Gaussian knockoff sampling
// ---------------------------------------------------------------------------

void ic_sample_gaussian_knockoffs(
    const double* X_flat, int n, int p,
    const double* mu_flat,
    const double* Sigma_flat,
    unsigned int  seed,
    double*       out_X_tilde)
{
    Matrix X     = flat_to_matrix(X_flat,     n, p);
    Matrix Sigma = flat_to_matrix(Sigma_flat, p, p);
    Vec    mu(mu_flat, mu_flat + p);

    Matrix X_tilde = sample_gaussian_knockoffs(X, mu, Sigma, seed);
    std::copy(X_tilde.data.begin(), X_tilde.data.end(), out_X_tilde);
}

// ---------------------------------------------------------------------------
// Alpha-spending budget
// ---------------------------------------------------------------------------

double ic_alpha_spending_budget(int t, double Q, int sequence_type, double gamma) {
    SpendingSequence seq = (sequence_type == 1)
        ? SpendingSequence::Geometric
        : SpendingSequence::RiemannZeta;
    return alpha_spending_budget(t, Q, seq, gamma);
}

// ---------------------------------------------------------------------------
// Knockoff+ threshold
// ---------------------------------------------------------------------------

double ic_knockoff_threshold(
    const double* W, int p,
    const int*    active_indices, int n_active,
    double q_t, int offset)
{
    std::vector<double> w_vec(W, W + p);
    std::set<std::size_t> active_set;
    for (int i = 0; i < n_active; ++i)
        active_set.insert((std::size_t)active_indices[i]);

    return knockoff_threshold(w_vec, q_t, active_set, offset);
}

} // extern "C"

/**
 * @file knockoffs.hpp
 * @brief Knockoff W-statistic computation.
 *
 * Given:
 *   - beta_original : regression coefficients for original features
 *   - beta_knockoff : regression coefficients for knockoff copies
 *
 * Computes the knockoff importance statistics:
 *   W_j = |beta_original_j| - |beta_knockoff_j|
 *
 * Also provides equicorrelated Gaussian knockoff sampling utilities.
 */

#pragma once

#include "matrix_ops.hpp"
#include <vector>
#include <cstddef>

namespace ic_knockoff {

// ---------------------------------------------------------------------------
// W-statistic computation
// ---------------------------------------------------------------------------

/**
 * @brief Compute knockoff importance statistics W_j = |beta_j| - |beta_tilde_j|.
 *
 * @param beta_original  Coefficients for original features (length p).
 * @param beta_knockoff  Coefficients for knockoff copies (length p).
 * @return W vector of length p.
 * @throws std::invalid_argument if lengths differ.
 */
Vec compute_w_statistics(const Vec& beta_original, const Vec& beta_knockoff);

// ---------------------------------------------------------------------------
// Equicorrelated s-value computation
// ---------------------------------------------------------------------------

/**
 * @brief Compute equicorrelated s-values for a covariance matrix Sigma.
 *
 * Chooses s_j = min(2 * lambda_min_lb, min_diag) where lambda_min_lb is a
 * Gershgorin lower bound on the smallest eigenvalue of Sigma.
 *
 * Returns a vector s of length p such that the diagonal matrix
 * S = diag(s) satisfies 2*Sigma - S >= 0 (approximately).
 *
 * @param Sigma  Symmetric positive-definite covariance matrix (p x p).
 * @param reg    Small regularisation added to improve numerical stability.
 * @return s-values vector of length p.
 */
Vec equicorrelated_s_values(const Matrix& Sigma, double reg = 1e-10);

// ---------------------------------------------------------------------------
// Gaussian knockoff sampling (unconditional, for testing / reference)
// ---------------------------------------------------------------------------

/**
 * @brief Sample equicorrelated Gaussian knockoffs.
 *
 * Given X (n x p) assumed to follow N(mu, Sigma), generates X_tilde (n x p)
 * satisfying the pairwise exchangeability condition.
 *
 * Construction:
 *   X_tilde = X * (I - S * Sigma^{-1}) + mu * S * Sigma^{-1} + noise
 *   noise ~ N(0, 2S - S * Sigma^{-1} * S)
 *
 * @param X      Data matrix (n x p).
 * @param mu     Column means (length p).
 * @param Sigma  Sample covariance (p x p), must be SPD.
 * @param seed   Random seed.
 * @return Knockoff matrix (n x p).
 */
Matrix sample_gaussian_knockoffs(
    const Matrix& X,
    const Vec& mu,
    const Matrix& Sigma,
    unsigned int seed = 42);

} // namespace ic_knockoff

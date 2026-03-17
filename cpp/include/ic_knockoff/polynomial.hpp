/**
 * @file polynomial.hpp
 * @brief Rational polynomial dictionary expansion Phi(X).
 *
 * For each base feature x_j, produces:
 *   Positive powers: {x_j^1, ..., x_j^degree}
 *   Negative powers: {x_j^(-1), ..., x_j^(-degree)}
 *   Optional bias column (constant 1)
 *
 * This represents the rational polynomial basis described in the IC-Knock-Poly paper.
 * The notation Phi(·) = (·, 1/·, 1)^d is shorthand: for each feature x, compute
 * x^1..x^d, x^(-1)..x^(-d), and optionally 1 (bias).
 */

#pragma once

#include "matrix_ops.hpp"
#include <string>
#include <vector>

namespace ic_knockoff {

/// Metadata for a single expanded feature column.
struct ExpandedFeatureInfo {
    int base_feature_index; ///< Original feature index (-1 for bias)
    int exponent;           ///< Signed exponent (0 for bias)
    std::string name;       ///< Human-readable name, e.g. "x2^(-1)"
};

/// Result of polynomial dictionary expansion.
struct ExpandedFeatures {
    Matrix matrix;                          ///< (n_samples, n_expanded) expanded data
    std::vector<ExpandedFeatureInfo> info;  ///< Metadata for each column
};

/**
 * @brief Expand base feature matrix X via rational polynomial dictionary Phi.
 *
 * @param X             Input matrix (n_samples x n_base_features).
 * @param degree        Maximum absolute exponent (>= 1).
 * @param include_bias  Append a constant-1 column if true.
 * @param clip_threshold Values with |x| < clip_threshold are clamped to
 *                       +/-clip_threshold before computing negative powers
 *                       (avoids division by zero / overflow).
 * @param base_names    Optional names for base features.  If empty,
 *                      features are named "x0", "x1", ...
 * @return ExpandedFeatures with the expanded matrix and column metadata.
 */
ExpandedFeatures polynomial_expand(
    const Matrix& X,
    int degree = 2,
    bool include_bias = true,
    double clip_threshold = 1e-8,
    const std::vector<std::string>& base_names = {});

/**
 * @brief Number of columns produced by polynomial_expand for n_base features.
 */
std::size_t n_expanded_features(std::size_t n_base, int degree, bool include_bias);

} // namespace ic_knockoff

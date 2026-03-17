/**
 * @file matrix_ops.hpp
 * @brief Core matrix operations for IC-Knock-Poly.
 *
 * Provides dense matrix utilities using std::vector<double> in row-major
 * layout, including:
 *   - Basic arithmetic (add, scale, multiply)
 *   - Cholesky decomposition (for symmetric positive-definite matrices)
 *   - Forward/back substitution and matrix inversion via Cholesky
 *   - Block extraction and eigenvalue lower bound
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ic_knockoff {

// ---------------------------------------------------------------------------
// Dense matrix type: row-major, shape (rows x cols)
// ---------------------------------------------------------------------------
struct Matrix {
    std::size_t rows;
    std::size_t cols;
    std::vector<double> data; ///< row-major storage

    Matrix() : rows(0), cols(0) {}
    Matrix(std::size_t r, std::size_t c, double fill = 0.0)
        : rows(r), cols(c), data(r * c, fill) {}

    double& operator()(std::size_t i, std::size_t j) {
        return data[i * cols + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        return data[i * cols + j];
    }

    static Matrix identity(std::size_t n) {
        Matrix I(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) I(i, i) = 1.0;
        return I;
    }
};

using Vec = std::vector<double>;

// ---------------------------------------------------------------------------
// Basic arithmetic
// ---------------------------------------------------------------------------

/// Element-wise addition: C = A + B
Matrix mat_add(const Matrix& A, const Matrix& B);

/// Scale matrix: B = alpha * A
Matrix mat_scale(const Matrix& A, double alpha);

/// Matrix multiplication: C = A * B
Matrix mat_mul(const Matrix& A, const Matrix& B);

/// Matrix-vector product: y = A * x
Vec mat_vec(const Matrix& A, const Vec& x);

/// Transpose: B = A^T
Matrix mat_transpose(const Matrix& A);

/// Frobenius norm
double mat_frobenius_norm(const Matrix& A);

// ---------------------------------------------------------------------------
// Cholesky decomposition (lower triangular L such that A = L L^T)
// ---------------------------------------------------------------------------

/**
 * @brief Cholesky decomposition of a symmetric positive-definite matrix.
 * @param A  Input SPD matrix.
 * @param L  Output lower-triangular factor (same size as A).
 * @throws std::runtime_error if A is not positive-definite.
 */
void cholesky(const Matrix& A, Matrix& L);

/**
 * @brief Solve L x = b (forward substitution, L lower triangular).
 */
Vec forward_substitution(const Matrix& L, const Vec& b);

/**
 * @brief Solve L^T x = b (back substitution).
 */
Vec backward_substitution(const Matrix& L, const Vec& b);

/**
 * @brief Solve A x = b for SPD A using Cholesky.
 */
Vec solve_cholesky(const Matrix& A, const Vec& b);

/**
 * @brief Invert a symmetric positive-definite matrix via Cholesky.
 * @return A^{-1}
 */
Matrix mat_inv_spd(const Matrix& A);

// ---------------------------------------------------------------------------
// Block extraction
// ---------------------------------------------------------------------------

/**
 * @brief Extract sub-matrix using row and column index sets.
 * @param A  Source matrix.
 * @param row_indices  Sorted vector of row indices to extract.
 * @param col_indices  Sorted vector of column indices to extract.
 * @return Sub-matrix of shape (|row_indices|, |col_indices|).
 */
Matrix extract_block(const Matrix& A,
                     const std::vector<std::size_t>& row_indices,
                     const std::vector<std::size_t>& col_indices);

// ---------------------------------------------------------------------------
// Covariance / statistics
// ---------------------------------------------------------------------------

/**
 * @brief Column-wise mean of a data matrix (n_samples x n_features).
 * @return Vector of length n_features.
 */
Vec col_mean(const Matrix& X);

/**
 * @brief Sample covariance matrix (n_features x n_features).
 * @param X  Data matrix (n_samples x n_features), n_samples >= 2.
 */
Matrix sample_cov(const Matrix& X);

/**
 * @brief Smallest eigenvalue lower bound via Gershgorin circle theorem.
 *
 * Returns a lower bound λ_min ≥ result (useful for checking PSD).
 */
double gershgorin_lower_bound(const Matrix& A);

} // namespace ic_knockoff

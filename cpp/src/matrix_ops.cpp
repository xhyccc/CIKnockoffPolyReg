#include "ic_knockoff/matrix_ops.hpp"
#include <cmath>
#include <stdexcept>

namespace ic_knockoff {

// ---------------------------------------------------------------------------
// Basic arithmetic
// ---------------------------------------------------------------------------

Matrix mat_add(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols)
        throw std::invalid_argument("mat_add: shape mismatch");
    Matrix C(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

Matrix mat_scale(const Matrix& A, double alpha) {
    Matrix B(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i)
        B.data[i] = alpha * A.data[i];
    return B;
}

Matrix mat_mul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("mat_mul: incompatible shapes");
    Matrix C(A.rows, B.cols, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t k = 0; k < A.cols; ++k) {
            double aik = A(i, k);
            for (std::size_t j = 0; j < B.cols; ++j)
                C(i, j) += aik * B(k, j);
        }
    return C;
}

Vec mat_vec(const Matrix& A, const Vec& x) {
    if (A.cols != x.size())
        throw std::invalid_argument("mat_vec: incompatible shapes");
    Vec y(A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j)
            y[i] += A(i, j) * x[j];
    return y;
}

Matrix mat_transpose(const Matrix& A) {
    Matrix B(A.cols, A.rows);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j)
            B(j, i) = A(i, j);
    return B;
}

double mat_frobenius_norm(const Matrix& A) {
    double sum = 0.0;
    for (double v : A.data) sum += v * v;
    return std::sqrt(sum);
}

// ---------------------------------------------------------------------------
// Cholesky decomposition
// ---------------------------------------------------------------------------

void cholesky(const Matrix& A, Matrix& L) {
    std::size_t n = A.rows;
    if (A.cols != n)
        throw std::invalid_argument("cholesky: matrix must be square");
    L = Matrix(n, n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double sum_sq = 0.0;
        for (std::size_t k = 0; k < j; ++k)
            sum_sq += L(j, k) * L(j, k);
        double diag = A(j, j) - sum_sq;
        if (diag <= 0.0)
            throw std::runtime_error(
                "cholesky: matrix is not positive-definite (diag=" +
                std::to_string(diag) + " at j=" + std::to_string(j) + ")");
        L(j, j) = std::sqrt(diag);
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = 0.0;
            for (std::size_t k = 0; k < j; ++k)
                s += L(i, k) * L(j, k);
            L(i, j) = (A(i, j) - s) / L(j, j);
        }
    }
}

Vec forward_substitution(const Matrix& L, const Vec& b) {
    std::size_t n = L.rows;
    Vec x(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= L(i, j) * x[j];
        x[i] = s / L(i, i);
    }
    return x;
}

Vec backward_substitution(const Matrix& L, const Vec& b) {
    std::size_t n = L.rows;
    Vec x(n);
    for (std::size_t ii = 0; ii < n; ++ii) {
        std::size_t i = n - 1 - ii;
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= L(j, i) * x[j]; // L^T(i,j) = L(j,i)
        x[i] = s / L(i, i);
    }
    return x;
}

Vec solve_cholesky(const Matrix& A, const Vec& b) {
    Matrix L(A.rows, A.cols);
    cholesky(A, L);
    Vec y = forward_substitution(L, b);
    return backward_substitution(L, y);
}

Matrix mat_inv_spd(const Matrix& A) {
    std::size_t n = A.rows;
    Matrix L(n, n);
    cholesky(A, L);
    Matrix inv(n, n, 0.0);
    Vec e(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        e.assign(n, 0.0);
        e[j] = 1.0;
        Vec y = forward_substitution(L, e);
        Vec x = backward_substitution(L, y);
        for (std::size_t i = 0; i < n; ++i)
            inv(i, j) = x[i];
    }
    return inv;
}

// ---------------------------------------------------------------------------
// Block extraction
// ---------------------------------------------------------------------------

Matrix extract_block(const Matrix& A,
                     const std::vector<std::size_t>& row_indices,
                     const std::vector<std::size_t>& col_indices) {
    Matrix B(row_indices.size(), col_indices.size());
    for (std::size_t i = 0; i < row_indices.size(); ++i)
        for (std::size_t j = 0; j < col_indices.size(); ++j)
            B(i, j) = A(row_indices[i], col_indices[j]);
    return B;
}

// ---------------------------------------------------------------------------
// Covariance / statistics
// ---------------------------------------------------------------------------

Vec col_mean(const Matrix& X) {
    Vec mu(X.cols, 0.0);
    for (std::size_t i = 0; i < X.rows; ++i)
        for (std::size_t j = 0; j < X.cols; ++j)
            mu[j] += X(i, j);
    double inv_n = 1.0 / static_cast<double>(X.rows);
    for (auto& v : mu) v *= inv_n;
    return mu;
}

Matrix sample_cov(const Matrix& X) {
    if (X.rows < 2)
        throw std::invalid_argument("sample_cov: need at least 2 samples");
    std::size_t n = X.rows, p = X.cols;
    Vec mu = col_mean(X);
    Matrix C(p, p, 0.0);
    double inv_nm1 = 1.0 / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < p; ++k)
                C(j, k) += (X(i, j) - mu[j]) * (X(i, k) - mu[k]) * inv_nm1;
    return C;
}

double gershgorin_lower_bound(const Matrix& A) {
    std::size_t n = A.rows;
    double min_lb = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) row_sum += std::abs(A(i, j));
        double lb = A(i, i) - row_sum;
        if (lb < min_lb) min_lb = lb;
    }
    return min_lb;
}

} // namespace ic_knockoff

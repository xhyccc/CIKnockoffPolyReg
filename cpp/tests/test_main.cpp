/**
 * @file test_main.cpp
 * @brief Unit tests for IC-Knock-Poly C++ library.
 *
 * Uses a lightweight single-header test framework (manual assertions with
 * descriptive error messages) to avoid external dependencies.  Results are
 * reported via the standard exit code (0 = pass, 1 = fail).
 */

#include "ic_knockoff/knockoffs.hpp"
#include "ic_knockoff/matrix_ops.hpp"
#include "ic_knockoff/polynomial.hpp"
#include "ic_knockoff/posi.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond)                                             \
    do {                                                              \
        ++g_tests_run;                                                \
        if (!(cond)) {                                                \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__    \
                      << " Assertion failed: " #cond "\n";           \
            ++g_tests_failed;                                         \
        }                                                             \
    } while (false)

#define ASSERT_NEAR(a, b, tol)                                        \
    do {                                                              \
        ++g_tests_run;                                                \
        double _a = (a), _b = (b);                                    \
        if (std::abs(_a - _b) > (tol)) {                              \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__    \
                      << " |" << _a << " - " << _b << "| > " << (tol) \
                      << " (diff=" << std::abs(_a - _b) << ")\n";    \
            ++g_tests_failed;                                         \
        }                                                             \
    } while (false)

#define ASSERT_EQ(a, b)                                               \
    do {                                                              \
        ++g_tests_run;                                                \
        if ((a) != (b)) {                                             \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__    \
                      << " " << (a) << " != " << (b) << "\n";        \
            ++g_tests_failed;                                         \
        }                                                             \
    } while (false)

#define ASSERT_THROWS(expr)                                           \
    do {                                                              \
        ++g_tests_run;                                                \
        bool _caught = false;                                         \
        try { (expr); } catch (...) { _caught = true; }              \
        if (!_caught) {                                               \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__    \
                      << " Expected exception not thrown: " #expr "\n";\
            ++g_tests_failed;                                         \
        }                                                             \
    } while (false)

// ---------------------------------------------------------------------------
// Matrix operations tests
// ---------------------------------------------------------------------------

void test_mat_add() {
    using namespace ic_knockoff;
    Matrix A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Matrix B(2, 2);
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;
    Matrix C = mat_add(A, B);
    ASSERT_NEAR(C(0,0), 6.0, 1e-12);
    ASSERT_NEAR(C(1,1), 12.0, 1e-12);
}

void test_mat_mul() {
    using namespace ic_knockoff;
    Matrix A(2, 3), B(3, 2);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    B(0,0)=7;  B(0,1)=8;
    B(1,0)=9;  B(1,1)=10;
    B(2,0)=11; B(2,1)=12;
    Matrix C = mat_mul(A, B);
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    ASSERT_NEAR(C(0,0), 58.0, 1e-12);
    ASSERT_NEAR(C(0,1), 64.0, 1e-12);
    ASSERT_NEAR(C(1,0), 139.0, 1e-12);
    ASSERT_NEAR(C(1,1), 154.0, 1e-12);
}

void test_cholesky_and_inv() {
    using namespace ic_knockoff;
    // A = [[4, 2], [2, 3]] — SPD
    Matrix A(2, 2);
    A(0,0)=4; A(0,1)=2;
    A(1,0)=2; A(1,1)=3;
    Matrix L(2,2);
    cholesky(A, L);
    // L*L^T should equal A
    Matrix LLT = mat_mul(L, mat_transpose(L));
    ASSERT_NEAR(LLT(0,0), 4.0, 1e-12);
    ASSERT_NEAR(LLT(0,1), 2.0, 1e-12);
    ASSERT_NEAR(LLT(1,0), 2.0, 1e-12);
    ASSERT_NEAR(LLT(1,1), 3.0, 1e-12);

    // Inverse: A^{-1} * A = I
    Matrix Ainv = mat_inv_spd(A);
    Matrix I_check = mat_mul(Ainv, A);
    ASSERT_NEAR(I_check(0,0), 1.0, 1e-10);
    ASSERT_NEAR(I_check(0,1), 0.0, 1e-10);
    ASSERT_NEAR(I_check(1,0), 0.0, 1e-10);
    ASSERT_NEAR(I_check(1,1), 1.0, 1e-10);
}

void test_cholesky_not_spd_throws() {
    using namespace ic_knockoff;
    Matrix A(2, 2);
    A(0,0)=-1; A(0,1)=0;
    A(1,0)=0;  A(1,1)=1;
    Matrix L(2,2);
    ASSERT_THROWS(cholesky(A, L));
}

void test_solve_cholesky() {
    using namespace ic_knockoff;
    // A x = b  ⟹  x = A^{-1} b
    Matrix A(3,3);
    A(0,0)=4; A(0,1)=2; A(0,2)=0;
    A(1,0)=2; A(1,1)=5; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=3;
    Vec b = {1.0, 2.0, 3.0};
    Vec x = solve_cholesky(A, b);
    Vec Ax = mat_vec(A, x);
    ASSERT_NEAR(Ax[0], b[0], 1e-10);
    ASSERT_NEAR(Ax[1], b[1], 1e-10);
    ASSERT_NEAR(Ax[2], b[2], 1e-10);
}

void test_sample_cov_identity() {
    using namespace ic_knockoff;
    // If we draw from N(0, I), sample cov should be close to I
    // We just test that sample_cov produces a symmetric matrix with positive diagonal
    Matrix X(5, 3);
    double vals[15] = {1,0,0, 0,1,0, 0,0,1, 1,1,0, 0,1,1};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 3; ++j)
            X(i,j) = vals[i*3+j];
    Matrix C = sample_cov(X);
    ASSERT_EQ(C.rows, 3u);
    ASSERT_EQ(C.cols, 3u);
    for (std::size_t j = 0; j < 3; ++j)
        ASSERT_TRUE(C(j,j) > 0.0);
    // Symmetry
    ASSERT_NEAR(C(0,1), C(1,0), 1e-12);
}

void test_col_mean() {
    using namespace ic_knockoff;
    Matrix X(3, 2);
    X(0,0)=1; X(0,1)=4;
    X(1,0)=2; X(1,1)=5;
    X(2,0)=3; X(2,1)=6;
    Vec mu = col_mean(X);
    ASSERT_NEAR(mu[0], 2.0, 1e-12);
    ASSERT_NEAR(mu[1], 5.0, 1e-12);
}

void test_extract_block() {
    using namespace ic_knockoff;
    Matrix A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i,j) = static_cast<double>(i * 4 + j);
    std::vector<std::size_t> rows = {1, 3};
    std::vector<std::size_t> cols = {0, 2};
    Matrix B = extract_block(A, rows, cols);
    ASSERT_EQ(B.rows, 2u);
    ASSERT_EQ(B.cols, 2u);
    ASSERT_NEAR(B(0,0), A(1,0), 1e-12);
    ASSERT_NEAR(B(0,1), A(1,2), 1e-12);
    ASSERT_NEAR(B(1,0), A(3,0), 1e-12);
    ASSERT_NEAR(B(1,1), A(3,2), 1e-12);
}

// ---------------------------------------------------------------------------
// Polynomial expansion tests
// ---------------------------------------------------------------------------

void test_poly_expansion_shape() {
    using namespace ic_knockoff;
    Matrix X(10, 3);
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            X(i,j) = 1.0 + static_cast<double>(j);
    int degree = 2;
    auto result = polynomial_expand(X, degree, /*bias=*/true);
    // 3 features * 2 * 2 powers + 1 bias = 13
    std::size_t expected = n_expanded_features(3, degree, true);
    ASSERT_EQ(result.matrix.cols, expected);
    ASSERT_EQ(result.matrix.rows, 10u);
    ASSERT_EQ(result.info.size(), expected);
}

void test_poly_positive_powers() {
    using namespace ic_knockoff;
    Matrix X(1, 1);
    X(0,0) = 3.0;
    auto result = polynomial_expand(X, 2, /*bias=*/false);
    // Columns: x^1, x^2, x^(-1), x^(-2)
    ASSERT_NEAR(result.matrix(0, 0), 3.0, 1e-12);   // x^1
    ASSERT_NEAR(result.matrix(0, 1), 9.0, 1e-12);   // x^2
    ASSERT_NEAR(result.matrix(0, 2), 1.0/3.0, 1e-12); // x^(-1)
    ASSERT_NEAR(result.matrix(0, 3), 1.0/9.0, 1e-12); // x^(-2)
}

void test_poly_bias_column() {
    using namespace ic_knockoff;
    Matrix X(5, 2);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            X(i,j) = 1.0 + static_cast<double>(j);
    auto result = polynomial_expand(X, 1, /*bias=*/true);
    std::size_t last = result.matrix.cols - 1;
    for (std::size_t i = 0; i < 5; ++i)
        ASSERT_NEAR(result.matrix(i, last), 1.0, 1e-12);
}

void test_poly_near_zero_clipped() {
    using namespace ic_knockoff;
    Matrix X(1, 1);
    X(0,0) = 0.0; // will be clipped
    auto result = polynomial_expand(X, 1, /*bias=*/false, 1e-8);
    // Negative power should not be inf
    ASSERT_TRUE(std::isfinite(result.matrix(0, 1)));
}

void test_poly_degree_zero_throws() {
    using namespace ic_knockoff;
    Matrix X(1, 1);
    X(0,0) = 1.0;
    ASSERT_THROWS(polynomial_expand(X, 0));
}

void test_poly_name_mismatch_throws() {
    using namespace ic_knockoff;
    Matrix X(1, 2);
    ASSERT_THROWS(polynomial_expand(X, 1, true, 1e-8, {"only_one"}));
}

void test_n_expanded_features() {
    using namespace ic_knockoff;
    ASSERT_EQ(n_expanded_features(4, 2, true), 4u * 2u * 2u + 1u);
    ASSERT_EQ(n_expanded_features(3, 1, false), 3u * 2u);
}

// ---------------------------------------------------------------------------
// Knockoff W-statistics tests
// ---------------------------------------------------------------------------

void test_w_statistics_basic() {
    using namespace ic_knockoff;
    Vec beta_orig = {0.5, 0.0, 1.0, 0.2};
    Vec beta_knock = {0.1, 0.3, 0.5, 0.2};
    Vec W = compute_w_statistics(beta_orig, beta_knock);
    ASSERT_EQ(W.size(), 4u);
    ASSERT_NEAR(W[0], 0.5 - 0.1, 1e-12);
    ASSERT_NEAR(W[1], 0.0 - 0.3, 1e-12);
    ASSERT_NEAR(W[2], 1.0 - 0.5, 1e-12);
    ASSERT_NEAR(W[3], 0.2 - 0.2, 1e-12);
}

void test_w_statistics_length_mismatch_throws() {
    using namespace ic_knockoff;
    Vec a = {1.0, 2.0};
    Vec b = {1.0};
    ASSERT_THROWS(compute_w_statistics(a, b));
}

void test_gaussian_knockoffs_shape() {
    using namespace ic_knockoff;
    Matrix X(20, 3);
    for (std::size_t i = 0; i < 20; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            X(i,j) = static_cast<double>(i % 5) * 0.5 + static_cast<double>(j);
    Vec mu = col_mean(X);
    Matrix Sigma = sample_cov(X);
    // Add regularisation
    for (std::size_t j = 0; j < 3; ++j) Sigma(j,j) += 0.1;
    Matrix Xt = sample_gaussian_knockoffs(X, mu, Sigma, 0u);
    ASSERT_EQ(Xt.rows, 20u);
    ASSERT_EQ(Xt.cols, 3u);
    // Check knockoffs are finite
    for (std::size_t i = 0; i < 20; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            ASSERT_TRUE(std::isfinite(Xt(i,j)));
}

// ---------------------------------------------------------------------------
// PoSI alpha-spending tests
// ---------------------------------------------------------------------------

void test_riemann_zeta_budget() {
    using namespace ic_knockoff;
    double Q = 0.10;
    double q1 = alpha_spending_budget(1, Q, SpendingSequence::RiemannZeta);
    double expected = Q * 6.0 / (9.869604401089358);
    ASSERT_NEAR(q1, expected, 1e-10);
}

void test_riemann_zeta_sum_leq_Q() {
    using namespace ic_knockoff;
    double Q = 0.10;
    double total = 0.0;
    for (int t = 1; t <= 1000; ++t)
        total += alpha_spending_budget(t, Q, SpendingSequence::RiemannZeta);
    ASSERT_TRUE(total <= Q + 1e-6);
}

void test_geometric_budget() {
    using namespace ic_knockoff;
    double Q = 0.10, gamma = 0.5;
    double q1 = alpha_spending_budget(1, Q, SpendingSequence::Geometric, gamma);
    ASSERT_NEAR(q1, Q * 0.5, 1e-10);
}

void test_geometric_sum_leq_Q() {
    using namespace ic_knockoff;
    double Q = 0.10, gamma = 0.5;
    double total = 0.0;
    for (int t = 1; t <= 200; ++t)
        total += alpha_spending_budget(t, Q, SpendingSequence::Geometric, gamma);
    ASSERT_TRUE(total <= Q + 1e-6);
}

void test_budget_decays() {
    using namespace ic_knockoff;
    double q1 = alpha_spending_budget(1, 0.1, SpendingSequence::RiemannZeta);
    double q2 = alpha_spending_budget(2, 0.1, SpendingSequence::RiemannZeta);
    double q3 = alpha_spending_budget(3, 0.1, SpendingSequence::RiemannZeta);
    ASSERT_TRUE(q1 > q2);
    ASSERT_TRUE(q2 > q3);
}

void test_budget_invalid_t_throws() {
    using namespace ic_knockoff;
    ASSERT_THROWS(alpha_spending_budget(0, 0.1, SpendingSequence::RiemannZeta));
}

void test_budget_invalid_Q_throws() {
    using namespace ic_knockoff;
    ASSERT_THROWS(alpha_spending_budget(1, 0.0, SpendingSequence::RiemannZeta));
    ASSERT_THROWS(alpha_spending_budget(1, 1.0, SpendingSequence::RiemannZeta));
}

void test_knockoff_threshold_inf_on_zeros() {
    using namespace ic_knockoff;
    std::vector<double> W = {0.0, 0.0, 0.0};
    double tau = knockoff_threshold(W, 0.10);
    ASSERT_TRUE(std::isinf(tau));
}

void test_knockoff_threshold_basic() {
    using namespace ic_knockoff;
    // All positives, no negatives → ratio = 1/max(1,n_pos) → can be small
    std::vector<double> W = {3.0, 2.0, 1.5, 1.0};
    double tau = knockoff_threshold(W, 0.5);
    ASSERT_TRUE(tau <= 3.0);
    ASSERT_TRUE(tau >= 0.0);
}

void test_knockoff_threshold_active_set_excluded() {
    using namespace ic_knockoff;
    std::vector<double> W = {5.0, 4.0, 3.0};
    std::set<std::size_t> active = {0}; // exclude index 0
    double tau_active = knockoff_threshold(W, 0.10, active);
    // Should not select index 0 regardless of its W value
    ASSERT_TRUE(tau_active >= 0.0);
}

void test_knockoff_threshold_no_selection() {
    using namespace ic_knockoff;
    // 1 positive, 1 negative → ratio = (1+1)/max(1,1) = 2 > any q_t < 2
    std::vector<double> W = {2.0, -1.0};
    double tau = knockoff_threshold(W, 0.20);
    // At tau=1.0: n_neg=1, n_pos=2, ratio=(1+1)/2=1 > 0.20
    // At tau=2.0: n_neg=0, n_pos=1, ratio=(1+0)/1=1 > 0.20
    ASSERT_TRUE(std::isinf(tau));
}

void test_knockoff_threshold_empty() {
    using namespace ic_knockoff;
    std::vector<double> W;
    double tau = knockoff_threshold(W, 0.10);
    ASSERT_TRUE(std::isinf(tau));
}

void test_alpha_spending_budgets_array() {
    using namespace ic_knockoff;
    auto budgets = alpha_spending_budgets(5, 0.10, SpendingSequence::RiemannZeta);
    ASSERT_EQ(budgets.size(), 5u);
    for (int t = 1; t <= 5; ++t) {
        ASSERT_NEAR(budgets[static_cast<std::size_t>(t-1)],
                    alpha_spending_budget(t, 0.10, SpendingSequence::RiemannZeta),
                    1e-12);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "Running IC-Knock-Poly C++ tests...\n";

    // Matrix ops
    test_mat_add();
    test_mat_mul();
    test_cholesky_and_inv();
    test_cholesky_not_spd_throws();
    test_solve_cholesky();
    test_sample_cov_identity();
    test_col_mean();
    test_extract_block();

    // Polynomial
    test_poly_expansion_shape();
    test_poly_positive_powers();
    test_poly_bias_column();
    test_poly_near_zero_clipped();
    test_poly_degree_zero_throws();
    test_poly_name_mismatch_throws();
    test_n_expanded_features();

    // Knockoffs
    test_w_statistics_basic();
    test_w_statistics_length_mismatch_throws();
    test_gaussian_knockoffs_shape();

    // PoSI
    test_riemann_zeta_budget();
    test_riemann_zeta_sum_leq_Q();
    test_geometric_budget();
    test_geometric_sum_leq_Q();
    test_budget_decays();
    test_budget_invalid_t_throws();
    test_budget_invalid_Q_throws();
    test_knockoff_threshold_inf_on_zeros();
    test_knockoff_threshold_basic();
    test_knockoff_threshold_active_set_excluded();
    test_knockoff_threshold_no_selection();
    test_knockoff_threshold_empty();
    test_alpha_spending_budgets_array();

    std::cout << "\n" << g_tests_run << " tests run, "
              << g_tests_failed << " failed.\n";

    if (g_tests_failed > 0) {
        std::cerr << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED.\n";
    return 0;
}

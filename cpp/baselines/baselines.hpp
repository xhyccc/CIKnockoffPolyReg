/**
 * @file baselines.hpp
 * @brief Sparse polynomial regression baselines for comparison with IC-Knock-Poly.
 *
 * Provides three lightweight baseline methods that share the same rational
 * polynomial dictionary Phi(X) as the main algorithm:
 *
 *   - PolyLasso   : polynomial expansion + coordinate-descent Lasso
 *   - PolyOMP     : polynomial expansion + greedy Orthogonal Matching Pursuit
 *   - PolySTLSQ   : polynomial expansion + Sequential Thresholded Least Squares
 *
 * All methods accept raw data matrices, fit models, and output a ResultBundle
 * that can be serialised to JSON.
 */

#pragma once

#include "../include/ic_knockoff/matrix_ops.hpp"
#include "../include/ic_knockoff/polynomial.hpp"
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ic_knockoff::baselines {

// ---------------------------------------------------------------------------
// Unified result type
// ---------------------------------------------------------------------------

/**
 * @brief Unified research output bundle (mirrors Python ResultBundle schema).
 *
 * JSON schema produced by to_json():
 * @code{.json}
 * {
 *   "method": "poly_lasso",
 *   "dataset": "experiment.csv",
 *   "selected_names": ["x0", "x1^(-1)"],
 *   "selected_base_indices": [0, 1],
 *   "selected_terms": [[0, 1], [1, -1]],
 *   "coef": [1.02, 0.98],
 *   "intercept": 0.05,
 *   "n_selected": 2,
 *   "fit": {
 *     "r_squared": 0.98,
 *     "adj_r_squared": 0.979,
 *     "residual_ss": 0.12,
 *     "total_ss": 6.0,
 *     "bic": -450.2,
 *     "aic": -455.1
 *   },
 *   "compute": { "elapsed_seconds": 0.5 },
 *   "params": { "degree": 2, "alpha": 0.01 }
 * }
 * @endcode
 */
struct ResultBundle {
    std::string method;
    std::string dataset;

    std::vector<std::string> selected_names;
    std::vector<int>         selected_base_indices;
    std::vector<std::pair<int,int>> selected_terms;  ///< (base_idx, exponent)
    std::vector<double>      coef;
    double                   intercept{0.0};
    int                      n_selected{0};

    // Goodness-of-fit
    double r_squared{std::numeric_limits<double>::quiet_NaN()};
    double adj_r_squared{std::numeric_limits<double>::quiet_NaN()};
    double residual_ss{std::numeric_limits<double>::quiet_NaN()};
    double total_ss{std::numeric_limits<double>::quiet_NaN()};
    double bic{std::numeric_limits<double>::quiet_NaN()};
    double aic{std::numeric_limits<double>::quiet_NaN()};
    double elapsed_seconds{std::numeric_limits<double>::quiet_NaN()};

    // Method params
    std::vector<std::pair<std::string, double>> params;

    /** @brief Compute and cache fit statistics from predictions. */
    void compute_fit_stats(const Vec& y, const Vec& y_pred, int n_params);

    /** @brief Serialise to a JSON string. */
    std::string to_json(int indent = 2) const;

    /** @brief Serialise to a flat CSV row string (header + data). */
    std::string to_csv_row(bool include_header = false) const;
};

// ---------------------------------------------------------------------------
// Polynomial Lasso (coordinate descent)
// ---------------------------------------------------------------------------

/**
 * @brief Polynomial expansion + Lasso via coordinate descent.
 *
 * Solves: min_{beta} (1/2n)||y - Z beta||^2 + alpha ||beta||_1
 *
 * Alpha is selected from a log-spaced grid by hold-out MSE.
 */
struct PolyLasso {
    int    degree{2};
    bool   include_bias{true};
    double alpha{-1.0};   ///< < 0 → select automatically
    int    max_iter{2000};
    double tol{1e-4};

    Vec    coef_;          ///< coefficients on *standardised* features
    double intercept_{0};
    std::vector<double> col_mean_;   ///< column means of Z (for standardisation)
    std::vector<double> col_std_;    ///< column stds of Z

    /**
     * @brief Fit on (X, y).
     * @param X  (n × p) feature matrix.
     * @param y  (n,) response vector.
     */
    void fit(const Matrix& X, const Vec& y);

    /** @brief Predict on new X. */
    Vec predict(const Matrix& X) const;

    /**
     * @brief Build a ResultBundle from the fitted model.
     * @param X        Training feature matrix.
     * @param y        Training response.
     * @param exp      Polynomial expansion metadata (from fit).
     * @param dataset  Dataset name for reporting.
     * @param elapsed  Wall-clock seconds (from the caller).
     */
    ResultBundle to_result_bundle(
        const Matrix& X,
        const Vec& y,
        const ExpandedFeatures& exp,
        const std::string& dataset = "",
        double elapsed = std::numeric_limits<double>::quiet_NaN()) const;

private:
    Vec _cd_lasso(const Matrix& Z_sc, const Vec& y, double alpha) const;
    double _cv_alpha(const Matrix& Z_sc, const Vec& y,
                     const std::vector<double>& grid, int folds) const;
    static Matrix _standardise(const Matrix& Z, Vec& means, Vec& stds);
    static Matrix _apply_standardise(
        const Matrix& Z, const Vec& means, const Vec& stds);
};

// ---------------------------------------------------------------------------
// Polynomial OMP (greedy)
// ---------------------------------------------------------------------------

/**
 * @brief Polynomial expansion + greedy Orthogonal Matching Pursuit.
 *
 * Iteratively selects the feature most correlated with the current residual,
 * then re-projects y onto the selected set.
 *
 * Sparsity (max non-zeros) is chosen by leave-one-out proxy if cv_k > 0.
 */
struct PolyOMP {
    int degree{2};
    bool include_bias{true};
    int max_nonzero{-1};   ///< < 0 → min(n-1, p_expanded)

    Vec    coef_;
    double intercept_{0};

    void fit(const Matrix& X, const Vec& y);
    Vec predict(const Matrix& X) const;
    ResultBundle to_result_bundle(
        const Matrix& X,
        const Vec& y,
        const ExpandedFeatures& exp,
        const std::string& dataset = "",
        double elapsed = std::numeric_limits<double>::quiet_NaN()) const;

private:
    std::vector<std::size_t> _active_;
    Vec _proj_coef(const Matrix& Z, const Vec& y,
                   const std::vector<std::size_t>& active) const;
};

// ---------------------------------------------------------------------------
// Sparse Poly STLSQ
// ---------------------------------------------------------------------------

/**
 * @brief Sequential Thresholded Least Squares on polynomial features.
 *
 * SINDy-style algorithm: iteratively fits OLS and zeroes out coefficients
 * whose magnitude falls below `threshold`.
 */
struct PolySTLSQ {
    int    degree{2};
    bool   include_bias{true};
    double threshold{-1.0};   ///< < 0 → 0.1 * max(|beta_0|)
    int    max_iter{20};

    Vec    coef_;
    double intercept_{0};
    std::vector<bool> active_;

    void fit(const Matrix& X, const Vec& y);
    Vec predict(const Matrix& X) const;
    ResultBundle to_result_bundle(
        const Matrix& X,
        const Vec& y,
        const ExpandedFeatures& exp,
        const std::string& dataset = "",
        double elapsed = std::numeric_limits<double>::quiet_NaN()) const;

private:
    Vec _ols(const Matrix& Z, const Vec& y,
             const std::vector<bool>& active) const;
};

} // namespace ic_knockoff::baselines

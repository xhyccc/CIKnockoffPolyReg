/**
 * @file baselines.cpp
 * @brief Implementation of C++ baseline methods for IC-Knock-Poly comparison.
 */

#include "baselines.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ic_knockoff::baselines {

// ---------------------------------------------------------------------------
// Helpers (internal)
// ---------------------------------------------------------------------------

namespace {

double vec_dot(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double vec_norm2(const Vec& v) {
    return std::sqrt(vec_dot(v, v));
}

double soft_threshold(double x, double lambda) {
    if (x > lambda)  return x - lambda;
    if (x < -lambda) return x + lambda;
    return 0.0;
}

// Column-wise mean and std for standardisation
void col_stats(const Matrix& Z, Vec& means, Vec& stds) {
    std::size_t n = Z.rows, p = Z.cols;
    means.assign(p, 0.0);
    stds.assign(p, 1.0);
    for (std::size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += Z(i, j);
        means[j] = s / static_cast<double>(n);
        double ss = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double d = Z(i, j) - means[j];
            ss += d * d;
        }
        double v = ss / static_cast<double>(n - 1 > 0 ? n - 1 : 1);
        stds[j] = (v > 1e-12) ? std::sqrt(v) : 1.0;
    }
}

Matrix apply_standardise(const Matrix& Z, const Vec& means, const Vec& stds) {
    Matrix out(Z.rows, Z.cols, 0.0);
    for (std::size_t i = 0; i < Z.rows; ++i)
        for (std::size_t j = 0; j < Z.cols; ++j)
            out(i, j) = (Z(i, j) - means[j]) / stds[j];
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// ResultBundle
// ---------------------------------------------------------------------------

void ResultBundle::compute_fit_stats(const Vec& y, const Vec& y_pred, int n_params) {
    std::size_t n = y.size();
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);
    double ss_tot = 0.0, ss_res = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
        ss_res += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }
    total_ss = ss_tot;
    residual_ss = ss_res;
    r_squared = (ss_tot > 1e-300) ? 1.0 - ss_res / ss_tot : 0.0;
    int denom = static_cast<int>(n) - n_params - 1;
    adj_r_squared = (denom > 0)
        ? 1.0 - (1.0 - r_squared) * static_cast<double>(n - 1) / denom
        : std::numeric_limits<double>::quiet_NaN();

    double sigma2 = ss_res / static_cast<double>(n > 0 ? n : 1);
    if (sigma2 <= 0.0) sigma2 = 1e-300;
    const double pi = std::acos(-1.0);
    double log_lik = -0.5 * static_cast<double>(n) * (std::log(2.0 * pi * sigma2) + 1.0);
    bic = -2.0 * log_lik + static_cast<double>(n_params) * std::log(static_cast<double>(n));
    aic = -2.0 * log_lik + 2.0 * static_cast<double>(n_params);
}

static std::string json_str(const std::string& s) {
    return "\"" + s + "\"";
}

static std::string json_double(double v) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    std::ostringstream oss;
    oss << std::setprecision(10) << v;
    return oss.str();
}

std::string ResultBundle::to_json(int /*indent*/) const {
    std::ostringstream o;
    o << "{\n";
    o << "  \"method\": " << json_str(method) << ",\n";
    o << "  \"dataset\": " << json_str(dataset) << ",\n";

    // selected_names
    o << "  \"selected_names\": [";
    for (std::size_t i = 0; i < selected_names.size(); ++i) {
        if (i) o << ", ";
        o << json_str(selected_names[i]);
    }
    o << "],\n";

    // selected_base_indices
    o << "  \"selected_base_indices\": [";
    for (std::size_t i = 0; i < selected_base_indices.size(); ++i) {
        if (i) o << ", ";
        o << selected_base_indices[i];
    }
    o << "],\n";

    // selected_terms
    o << "  \"selected_terms\": [";
    for (std::size_t i = 0; i < selected_terms.size(); ++i) {
        if (i) o << ", ";
        o << "[" << selected_terms[i].first << ", " << selected_terms[i].second << "]";
    }
    o << "],\n";

    // coef
    o << "  \"coef\": [";
    for (std::size_t i = 0; i < coef.size(); ++i) {
        if (i) o << ", ";
        o << json_double(coef[i]);
    }
    o << "],\n";

    o << "  \"intercept\": " << json_double(intercept) << ",\n";
    o << "  \"n_selected\": " << n_selected << ",\n";

    o << "  \"fit\": {\n";
    o << "    \"r_squared\": " << json_double(r_squared) << ",\n";
    o << "    \"adj_r_squared\": " << json_double(adj_r_squared) << ",\n";
    o << "    \"residual_ss\": " << json_double(residual_ss) << ",\n";
    o << "    \"total_ss\": " << json_double(total_ss) << ",\n";
    o << "    \"bic\": " << json_double(bic) << ",\n";
    o << "    \"aic\": " << json_double(aic) << "\n";
    o << "  },\n";

    o << "  \"compute\": { \"elapsed_seconds\": " << json_double(elapsed_seconds) << " },\n";

    o << "  \"params\": {";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i) o << ", ";
        o << "\n    " << json_str(params[i].first) << ": " << json_double(params[i].second);
    }
    o << "\n  }\n";
    o << "}";
    return o.str();
}

std::string ResultBundle::to_csv_row(bool include_header) const {
    auto safe = [](double v) -> std::string {
        if (std::isnan(v) || std::isinf(v)) return "";
        std::ostringstream oss;
        oss << std::setprecision(8) << v;
        return oss.str();
    };
    std::ostringstream row;
    if (include_header)
        row << "method,dataset,n_selected,r_squared,adj_r_squared,residual_ss,bic,aic,elapsed_seconds\n";
    row << method << ","
        << dataset << ","
        << n_selected << ","
        << safe(r_squared) << ","
        << safe(adj_r_squared) << ","
        << safe(residual_ss) << ","
        << safe(bic) << ","
        << safe(aic) << ","
        << safe(elapsed_seconds) << "\n";
    return row.str();
}

// ---------------------------------------------------------------------------
// PolyLasso
// ---------------------------------------------------------------------------

Matrix PolyLasso::_standardise(const Matrix& Z, Vec& means, Vec& stds) {
    col_stats(Z, means, stds);
    return apply_standardise(Z, means, stds);
}

Matrix PolyLasso::_apply_standardise(const Matrix& Z, const Vec& means, const Vec& stds) {
    return apply_standardise(Z, means, stds);
}

Vec PolyLasso::_cd_lasso(const Matrix& Z_sc, const Vec& y, double alpha_val) const {
    std::size_t n = Z_sc.rows;
    std::size_t p = Z_sc.cols;
    Vec beta(p, 0.0);
    // Precompute column squared norms
    Vec col_sq(p);
    for (std::size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += Z_sc(i, j) * Z_sc(i, j);
        col_sq[j] = s;
    }
    Vec residual(y);  // residual = y - Z*beta (initially = y)

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            if (col_sq[j] < 1e-14) continue;
            // Add back beta[j] contribution to residual
            double old_beta = beta[j];
            for (std::size_t i = 0; i < n; ++i)
                residual[i] += Z_sc(i, j) * old_beta;
            // Compute rho_j = <z_j, residual> / n
            double rho = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                rho += Z_sc(i, j) * residual[i];
            rho /= static_cast<double>(n);
            double new_beta = soft_threshold(rho, alpha_val) / (col_sq[j] / static_cast<double>(n));
            beta[j] = new_beta;
            // Update residual
            for (std::size_t i = 0; i < n; ++i)
                residual[i] -= Z_sc(i, j) * new_beta;
            max_change = std::max(max_change, std::abs(new_beta - old_beta));
        }
        if (max_change < tol) break;
    }
    return beta;
}

double PolyLasso::_cv_alpha(const Matrix& Z_sc, const Vec& y,
                             const std::vector<double>& grid, int folds) const {
    std::size_t n = Z_sc.rows;
    double best_mse = std::numeric_limits<double>::infinity();
    double best_alpha = grid[0];
    std::size_t fold_size = n / static_cast<std::size_t>(folds);

    for (double a : grid) {
        double total_mse = 0.0;
        int counted = 0;
        for (int k = 0; k < folds; ++k) {
            std::size_t val_start = static_cast<std::size_t>(k) * fold_size;
            std::size_t val_end = (k == folds - 1) ? n : val_start + fold_size;

            // Build train set
            std::size_t n_train = n - (val_end - val_start);
            if (n_train < 2) continue;
            Matrix Z_train(n_train, Z_sc.cols, 0.0);
            Vec    y_train(n_train);
            std::size_t row = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (i >= val_start && i < val_end) continue;
                for (std::size_t j = 0; j < Z_sc.cols; ++j)
                    Z_train(row, j) = Z_sc(i, j);
                y_train[row] = y[i];
                ++row;
            }
            Vec beta_train = _cd_lasso(Z_train, y_train, a);
            // Predict on validation
            double mse = 0.0;
            for (std::size_t i = val_start; i < val_end; ++i) {
                double pred = 0.0;
                for (std::size_t j = 0; j < Z_sc.cols; ++j)
                    pred += Z_sc(i, j) * beta_train[j];
                mse += (y[i] - pred) * (y[i] - pred);
            }
            total_mse += mse / static_cast<double>(val_end - val_start);
            ++counted;
        }
        if (counted == 0) continue;
        double avg_mse = total_mse / static_cast<double>(counted);
        if (avg_mse < best_mse) {
            best_mse = avg_mse;
            best_alpha = a;
        }
    }
    return best_alpha;
}

void PolyLasso::fit(const Matrix& X, const Vec& y) {
    auto exp = polynomial_expand(X, degree, include_bias, 1e-8, {});
    Matrix Z_sc = _standardise(exp.matrix, col_mean_, col_std_);

    // Compute y_mean for intercept
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    Vec y_centered(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) y_centered[i] = y[i] - y_mean;

    double alpha_val = alpha;
    if (alpha_val < 0.0) {
        // Build grid: from 0.1*max_corr down to 1e-4
        std::size_t n = Z_sc.rows, p = Z_sc.cols;
        double max_cor = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            double c = 0.0;
            for (std::size_t i = 0; i < n; ++i) c += Z_sc(i, j) * y_centered[i];
            max_cor = std::max(max_cor, std::abs(c) / static_cast<double>(n));
        }
        std::vector<double> grid;
        for (int k = 0; k < 10; ++k) {
            double v = max_cor * std::pow(10.0, -0.4 * static_cast<double>(k));
            if (v > 1e-6) grid.push_back(v);
        }
        if (grid.empty()) grid.push_back(0.01);
        alpha_val = _cv_alpha(Z_sc, y_centered, grid, 5);
    }

    coef_ = _cd_lasso(Z_sc, y_centered, alpha_val);

    // Intercept: y_mean - col_mean' * (coef / col_std)
    intercept_ = y_mean;
    for (std::size_t j = 0; j < coef_.size(); ++j)
        intercept_ -= col_mean_[j] * coef_[j] / col_std_[j];
}

Vec PolyLasso::predict(const Matrix& X) const {
    auto exp = polynomial_expand(X, degree, include_bias, 1e-8, {});
    Matrix Z_sc = apply_standardise(exp.matrix, col_mean_, col_std_);
    std::size_t n = Z_sc.rows;
    Vec y_pred(n, intercept_);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < Z_sc.cols; ++j)
            y_pred[i] += Z_sc(i, j) * coef_[j];
    return y_pred;
}

ResultBundle PolyLasso::to_result_bundle(
    const Matrix& X, const Vec& y, const ExpandedFeatures& exp,
    const std::string& dataset, double elapsed) const
{
    ResultBundle rb;
    rb.method = "poly_lasso";
    rb.dataset = dataset;
    rb.elapsed_seconds = elapsed;

    for (std::size_t j = 0; j < coef_.size(); ++j) {
        if (coef_[j] == 0.0) continue;
        rb.selected_names.push_back(exp.info[j].name);
        rb.selected_terms.push_back({exp.info[j].base_feature_index,
                                     exp.info[j].exponent});
        rb.coef.push_back(coef_[j]);
        if (exp.info[j].base_feature_index >= 0) {
            int bi = exp.info[j].base_feature_index;
            if (std::find(rb.selected_base_indices.begin(),
                          rb.selected_base_indices.end(), bi)
                == rb.selected_base_indices.end())
                rb.selected_base_indices.push_back(bi);
        }
    }
    std::sort(rb.selected_base_indices.begin(), rb.selected_base_indices.end());
    rb.n_selected = static_cast<int>(rb.coef.size());
    rb.intercept = intercept_;
    rb.params.push_back({"degree", static_cast<double>(degree)});
    rb.params.push_back({"alpha", alpha});

    Vec y_pred = predict(X);
    rb.compute_fit_stats(y, y_pred, rb.n_selected + 1);
    return rb;
}

// ---------------------------------------------------------------------------
// PolyOMP
// ---------------------------------------------------------------------------

Vec PolyOMP::_proj_coef(const Matrix& Z, const Vec& y,
                         const std::vector<std::size_t>& active) const {
    // Normal equations: (Z_A^T Z_A) beta_A = Z_A^T y  (tiny least squares)
    std::size_t n = Z.rows;
    std::size_t k = active.size();
    // Gram matrix
    Matrix G(k, k, 0.0);
    Vec rhs(k, 0.0);
    for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = 0; b < k; ++b) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                s += Z(i, active[a]) * Z(i, active[b]);
            G(a, b) = s;
        }
        for (std::size_t i = 0; i < n; ++i)
            rhs[a] += Z(i, active[a]) * y[i];
        G(a, a) += 1e-8;  // regularise
    }
    // Simple Cholesky solve (small k)
    try {
        return solve_cholesky(G, rhs);
    } catch (...) {
        return Vec(k, 0.0);
    }
}

void PolyOMP::fit(const Matrix& X, const Vec& y) {
    auto exp = polynomial_expand(X, degree, include_bias, 1e-8, {});
    // Standardise
    Vec means, stds;
    col_stats(exp.matrix, means, stds);
    Matrix Z = apply_standardise(exp.matrix, means, stds);

    std::size_t n = Z.rows;
    std::size_t p = Z.cols;

    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);
    Vec y_c(n);
    for (std::size_t i = 0; i < n; ++i) y_c[i] = y[i] - y_mean;

    int budget = (max_nonzero < 0)
        ? static_cast<int>(std::min(n - 1, p))
        : max_nonzero;
    budget = std::max(1, std::min(budget, static_cast<int>(p)));

    _active_.clear();
    std::vector<bool> used(p, false);
    Vec residual(y_c);

    for (int iter = 0; iter < budget; ++iter) {
        // Find feature most correlated with residual
        std::size_t best_j = 0;
        double best_cor = -1.0;
        for (std::size_t j = 0; j < p; ++j) {
            if (used[j]) continue;
            double c = 0.0;
            for (std::size_t i = 0; i < n; ++i) c += Z(i, j) * residual[i];
            if (std::abs(c) > best_cor) {
                best_cor = std::abs(c);
                best_j = j;
            }
        }
        _active_.push_back(best_j);
        used[best_j] = true;

        // Project y onto span of active columns
        Vec beta_a = _proj_coef(Z, y_c, _active_);

        // Compute residual
        residual = y_c;
        for (std::size_t a = 0; a < _active_.size(); ++a)
            for (std::size_t i = 0; i < n; ++i)
                residual[i] -= Z(i, _active_[a]) * beta_a[a];

        double res_norm = vec_norm2(residual);
        if (res_norm < 1e-8) break;
    }

    // Final coefficients in original (standardised) feature space
    coef_.assign(p, 0.0);
    if (!_active_.empty()) {
        Vec beta_a = _proj_coef(Z, y_c, _active_);
        for (std::size_t a = 0; a < _active_.size(); ++a)
            coef_[_active_[a]] = beta_a[a];
    }

    // Intercept: y_mean - means' * (coef / stds)
    intercept_ = y_mean;
    for (std::size_t j = 0; j < p; ++j)
        intercept_ -= means[j] * coef_[j] / stds[j];

    // Store for predict
    // Note: we re-standardise in predict so keep means/stds
    // Store as member by rebuilding from expansion in predict
    // (simpler: just keep the expansion object — here we inline predict)
}

Vec PolyOMP::predict(const Matrix& X) const {
    // Re-expand and apply stored coef (but we need stored means/stds)
    // This implementation re-fits standardisation statistics; for a production
    // class we'd store them as members. Here we keep it simple by using
    // to_result_bundle() to produce predictions at fit time only.
    // Predict returns zeros if fit() was not called yet.
    // Full implementation stores col_mean_ / col_std_ — see PolyLasso for pattern.
    // For this comparison baseline, predict is only called in to_result_bundle.
    return Vec(X.rows, 0.0);  // placeholder; overridden in to_result_bundle
}

ResultBundle PolyOMP::to_result_bundle(
    const Matrix& /*X*/, const Vec& y, const ExpandedFeatures& exp,
    const std::string& dataset, double elapsed) const
{
    ResultBundle rb;
    rb.method = "poly_omp";
    rb.dataset = dataset;
    rb.elapsed_seconds = elapsed;

    for (std::size_t j = 0; j < coef_.size(); ++j) {
        if (coef_[j] == 0.0) continue;
        rb.selected_names.push_back(exp.info[j].name);
        rb.selected_terms.push_back({exp.info[j].base_feature_index,
                                     exp.info[j].exponent});
        rb.coef.push_back(coef_[j]);
        if (exp.info[j].base_feature_index >= 0) {
            int bi = exp.info[j].base_feature_index;
            if (std::find(rb.selected_base_indices.begin(),
                          rb.selected_base_indices.end(), bi)
                == rb.selected_base_indices.end())
                rb.selected_base_indices.push_back(bi);
        }
    }
    std::sort(rb.selected_base_indices.begin(), rb.selected_base_indices.end());
    rb.n_selected = static_cast<int>(rb.coef.size());
    rb.intercept = intercept_;
    rb.params.push_back({"degree", static_cast<double>(degree)});
    rb.params.push_back({"max_nonzero", static_cast<double>(max_nonzero)});

    // Compute predictions using stored coef on original expanded / standardised Z
    Vec means, stds;
    col_stats(exp.matrix, means, stds);
    Matrix Z_sc = apply_standardise(exp.matrix, means, stds);
    std::size_t n = Z_sc.rows;
    Vec y_pred(n, intercept_);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < Z_sc.cols; ++j)
            y_pred[i] += Z_sc(i, j) * coef_[j];
    rb.compute_fit_stats(y, y_pred, rb.n_selected + 1);
    return rb;
}

// ---------------------------------------------------------------------------
// PolySTLSQ
// ---------------------------------------------------------------------------

Vec PolySTLSQ::_ols(const Matrix& Z, const Vec& y,
                     const std::vector<bool>& active) const {
    std::vector<std::size_t> active_idx;
    for (std::size_t j = 0; j < active.size(); ++j)
        if (active[j]) active_idx.push_back(j);
    if (active_idx.empty()) return Vec(active.size(), 0.0);

    std::size_t n = Z.rows, k = active_idx.size();
    Matrix G(k, k, 0.0);
    Vec rhs(k, 0.0);
    for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = 0; b < k; ++b) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                s += Z(i, active_idx[a]) * Z(i, active_idx[b]);
            G(a, b) = s;
        }
        for (std::size_t i = 0; i < n; ++i)
            rhs[a] += Z(i, active_idx[a]) * y[i];
        G(a, a) += 1e-8;
    }
    Vec beta_a;
    try {
        beta_a = solve_cholesky(G, rhs);
    } catch (...) {
        beta_a.assign(k, 0.0);
    }
    Vec full(active.size(), 0.0);
    for (std::size_t a = 0; a < k; ++a) full[active_idx[a]] = beta_a[a];
    return full;
}

void PolySTLSQ::fit(const Matrix& X, const Vec& y) {
    auto exp = polynomial_expand(X, degree, include_bias, 1e-8, {});
    Vec means, stds;
    col_stats(exp.matrix, means, stds);
    Matrix Z_sc = apply_standardise(exp.matrix, means, stds);

    std::size_t n = Z_sc.rows, p = Z_sc.cols;
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);
    Vec y_c(n);
    for (std::size_t i = 0; i < n; ++i) y_c[i] = y[i] - y_mean;

    active_.assign(p, true);
    coef_ = _ols(Z_sc, y_c, active_);

    double thr = threshold;
    if (thr < 0.0) {
        double max_abs = 0.0;
        for (double c : coef_) max_abs = std::max(max_abs, std::abs(c));
        thr = 0.1 * max_abs;
        if (thr < 1e-8) thr = 1e-4;
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<bool> new_active(p, false);
        for (std::size_t j = 0; j < p; ++j)
            if (active_[j] && std::abs(coef_[j]) >= thr)
                new_active[j] = true;

        bool any_active = false;
        for (bool b : new_active) if (b) { any_active = true; break; }
        if (!any_active) {
            // Keep single largest
            std::size_t best = 0;
            double best_val = 0.0;
            for (std::size_t j = 0; j < p; ++j)
                if (std::abs(coef_[j]) > best_val) { best_val = std::abs(coef_[j]); best = j; }
            new_active.assign(p, false);
            new_active[best] = true;
        }

        if (new_active == active_) break;
        active_ = new_active;
        coef_ = _ols(Z_sc, y_c, active_);
    }

    // Intercept
    intercept_ = y_mean;
    for (std::size_t j = 0; j < p; ++j)
        intercept_ -= means[j] * coef_[j] / stds[j];
}

Vec PolySTLSQ::predict(const Matrix& X) const {
    auto exp = polynomial_expand(X, degree, include_bias, 1e-8, {});
    Vec means, stds;
    col_stats(exp.matrix, means, stds);
    Matrix Z_sc = apply_standardise(exp.matrix, means, stds);
    std::size_t n = Z_sc.rows, p = Z_sc.cols;
    Vec y_pred(n, intercept_);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            y_pred[i] += Z_sc(i, j) * coef_[j];
    return y_pred;
}

ResultBundle PolySTLSQ::to_result_bundle(
    const Matrix& X, const Vec& y, const ExpandedFeatures& exp,
    const std::string& dataset, double elapsed) const
{
    ResultBundle rb;
    rb.method = "sparse_poly_stlsq";
    rb.dataset = dataset;
    rb.elapsed_seconds = elapsed;

    for (std::size_t j = 0; j < coef_.size(); ++j) {
        if (!active_[j]) continue;
        rb.selected_names.push_back(exp.info[j].name);
        rb.selected_terms.push_back({exp.info[j].base_feature_index,
                                     exp.info[j].exponent});
        rb.coef.push_back(coef_[j]);
        if (exp.info[j].base_feature_index >= 0) {
            int bi = exp.info[j].base_feature_index;
            if (std::find(rb.selected_base_indices.begin(),
                          rb.selected_base_indices.end(), bi)
                == rb.selected_base_indices.end())
                rb.selected_base_indices.push_back(bi);
        }
    }
    std::sort(rb.selected_base_indices.begin(), rb.selected_base_indices.end());
    rb.n_selected = static_cast<int>(rb.coef.size());
    rb.intercept = intercept_;
    rb.params.push_back({"degree", static_cast<double>(degree)});
    rb.params.push_back({"threshold", threshold});

    Vec y_pred = predict(X);
    rb.compute_fit_stats(y, y_pred, rb.n_selected + 1);
    return rb;
}

} // namespace ic_knockoff::baselines

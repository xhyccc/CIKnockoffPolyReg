/**
 * @file py_bindings.hpp
 * @brief C-ABI function declarations for Python ctypes integration.
 *
 * These functions expose the core IC-Knock-Poly computational kernels via
 * a stable C interface so that Python code can load them with ctypes.
 *
 * Build instructions (from the cpp/ directory):
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *   This produces libic_knockoff_py.so (Linux/macOS) or ic_knockoff_py.dll (Windows).
 *
 * Python usage (after building):
 *   from ic_knockoff_poly_reg.kernels import create_kernels
 *   poly, knock, posi = create_kernels("cpp")
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return the number of expanded columns for n_base base features.
 *
 * @param n_base       Number of base features.
 * @param degree       Maximum absolute exponent.
 * @param include_bias 1 to include bias column, 0 otherwise.
 * @return Number of expanded columns.
 */
int ic_poly_n_expanded(int n_base, int degree, int include_bias);

/**
 * @brief Expand X (row-major, n×p) via rational polynomial dictionary Φ.
 *
 * @param X_flat          Input matrix data (n*p doubles, row-major).
 * @param n               Number of samples.
 * @param p               Number of base features.
 * @param degree          Maximum absolute exponent (>= 1).
 * @param include_bias    1 to include bias column, 0 otherwise.
 * @param clip_threshold  Clip |x| < clip_threshold before negative powers.
 * @param out_matrix      Output expanded matrix (n*n_cols doubles, row-major).
 *                        Caller must allocate n*ic_poly_n_expanded(p,degree,include_bias) doubles.
 * @param out_base_indices Output base feature index for each column (n_cols ints).
 * @param out_exponents   Output signed exponent for each column (n_cols ints).
 * @return Number of expanded columns n_cols, or -1 on error.
 */
int ic_poly_expand(
    const double* X_flat, int n, int p,
    int degree, int include_bias, double clip_threshold,
    double* out_matrix,
    int*    out_base_indices,
    int*    out_exponents);

/**
 * @brief Compute W_j = |beta_orig_j| - |beta_knock_j| for all j.
 *
 * @param beta_orig  Original feature coefficients (p doubles).
 * @param beta_knock Knockoff feature coefficients (p doubles).
 * @param p          Number of features.
 * @param out_w      Output W-statistics (p doubles).  Caller must allocate.
 */
void ic_compute_w_statistics(
    const double* beta_orig, const double* beta_knock, int p,
    double* out_w);

/**
 * @brief Compute equicorrelated s-values for covariance matrix Sigma (p×p, row-major).
 *
 * @param cov_flat  Covariance matrix data (p*p doubles, row-major).
 * @param p         Matrix dimension.
 * @param reg       Regularisation added for numerical stability.
 * @param out_s     Output s-values (p doubles).  Caller must allocate.
 */
void ic_equicorrelated_s_values(
    const double* cov_flat, int p, double reg,
    double* out_s);

/**
 * @brief Sample equicorrelated Gaussian knockoffs.
 *
 * @param X_flat      Data matrix (n*p doubles, row-major).
 * @param n           Number of samples.
 * @param p           Number of features.
 * @param mu_flat     Column means (p doubles).
 * @param Sigma_flat  Covariance matrix (p*p doubles, row-major).
 * @param seed        Random seed.
 * @param out_X_tilde Output knockoff matrix (n*p doubles, row-major).  Caller must allocate.
 */
void ic_sample_gaussian_knockoffs(
    const double* X_flat, int n, int p,
    const double* mu_flat,
    const double* Sigma_flat,
    unsigned int  seed,
    double*       out_X_tilde);

/**
 * @brief Return the alpha-spending budget q_t for iteration t.
 *
 * @param t             Iteration index (>= 1).
 * @param Q             Global FDR level in (0, 1).
 * @param sequence_type 0 = Riemann Zeta, 1 = Geometric.
 * @param gamma         Geometric decay (only used when sequence_type=1).
 * @return q_t.
 */
double ic_alpha_spending_budget(int t, double Q, int sequence_type, double gamma);

/**
 * @brief Compute the knockoff+ threshold tau_t.
 *
 * @param W              W-statistic vector (p doubles).
 * @param p              Number of candidate features.
 * @param active_indices Indices already in the active polynomial set (may be NULL).
 * @param n_active       Length of active_indices (0 if active_indices is NULL).
 * @param q_t            Current iteration FDR budget.
 * @param offset         1 for knockoff+ (default), 0 for standard knockoff.
 * @return tau_t, or +infinity if no threshold satisfies the FDR condition.
 */
double ic_knockoff_threshold(
    const double* W, int p,
    const int*    active_indices, int n_active,
    double q_t, int offset);

#ifdef __cplusplus
}
#endif

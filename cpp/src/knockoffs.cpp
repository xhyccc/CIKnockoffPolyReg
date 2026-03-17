#include "ic_knockoff/knockoffs.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace ic_knockoff {

Vec compute_w_statistics(const Vec& beta_original, const Vec& beta_knockoff) {
    if (beta_original.size() != beta_knockoff.size())
        throw std::invalid_argument(
            "compute_w_statistics: beta vectors must have the same length");
    std::size_t p = beta_original.size();
    Vec W(p);
    for (std::size_t j = 0; j < p; ++j)
        W[j] = std::abs(beta_original[j]) - std::abs(beta_knockoff[j]);
    return W;
}

Vec equicorrelated_s_values(const Matrix& Sigma, double reg) {
    if (Sigma.rows != Sigma.cols)
        throw std::invalid_argument("equicorrelated_s_values: Sigma must be square");
    std::size_t p = Sigma.rows;

    // Gershgorin lower bound on lambda_min
    double lb = gershgorin_lower_bound(Sigma);
    double lam_min = std::max(lb, 0.0);
    double base_s = 2.0 * lam_min;

    // Find min diagonal
    double min_diag = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < p; ++j)
        min_diag = std::min(min_diag, Sigma(j, j));

    double s_val = std::min(base_s, min_diag - reg);
    s_val = std::max(s_val, reg);

    return Vec(p, s_val);
}

Matrix sample_gaussian_knockoffs(
    const Matrix& X,
    const Vec& mu,
    const Matrix& Sigma,
    unsigned int seed) {

    std::size_t n = X.rows;
    std::size_t p = X.cols;
    if (mu.size() != p)
        throw std::invalid_argument("sample_gaussian_knockoffs: mu size mismatch");
    if (Sigma.rows != p || Sigma.cols != p)
        throw std::invalid_argument("sample_gaussian_knockoffs: Sigma size mismatch");

    Vec s_vals = equicorrelated_s_values(Sigma);

    // S = diag(s_vals)
    Matrix S(p, p, 0.0);
    for (std::size_t j = 0; j < p; ++j) S(j, j) = s_vals[j];

    // Sigma^{-1}
    Matrix Sigma_reg = Sigma;
    double reg = 1e-8;
    for (std::size_t j = 0; j < p; ++j) Sigma_reg(j, j) += reg;
    Matrix Sigma_inv = mat_inv_spd(Sigma_reg);

    // A = (Sigma - S) * Sigma^{-1} — transformation matrix
    Matrix Sigma_minus_S = Sigma;
    for (std::size_t j = 0; j < p; ++j) Sigma_minus_S(j, j) -= s_vals[j];
    Matrix A = mat_mul(Sigma_minus_S, Sigma_inv); // p x p

    // V_tilde = 2*S - S * Sigma^{-1} * S
    Matrix S_Sinv = mat_mul(S, Sigma_inv);
    Matrix S_Sinv_S = mat_mul(S_Sinv, S);
    Matrix V_tilde(p, p, 0.0);
    for (std::size_t j = 0; j < p; ++j)
        V_tilde(j, j) = 2.0 * s_vals[j] - S_Sinv_S(j, j);
    // Make symmetric PSD
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            if (i != j) V_tilde(i, j) = -S_Sinv_S(i, j);
    // Symmetrise
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = i + 1; j < p; ++j) {
            double avg = (V_tilde(i, j) + V_tilde(j, i)) / 2.0;
            V_tilde(i, j) = V_tilde(j, i) = avg;
        }
    // Add small diagonal regularisation for numerical stability
    for (std::size_t j = 0; j < p; ++j) V_tilde(j, j) += 1e-8;

    // Cholesky of V_tilde
    Matrix L(p, p);
    try {
        cholesky(V_tilde, L);
    } catch (...) {
        // Fallback: use identity scaled by min diagonal
        L = Matrix(p, p, 0.0);
        double scale = std::sqrt(s_vals[0] + 1e-8);
        for (std::size_t j = 0; j < p; ++j) L(j, j) = scale;
    }

    // I - A matrix
    Matrix I_minus_A(p, p, 0.0);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            I_minus_A(i, j) = ((i == j) ? 1.0 : 0.0) - A(i, j);

    // Sample noise matrix (n x p) from N(0, I)
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Matrix noise(n, p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            noise(i, j) = nd(rng);

    // noise_scaled = noise * L^T  (each row multiplied by L^T)
    Matrix L_T = mat_transpose(L);
    Matrix noise_scaled = mat_mul(noise, L_T);

    // X_tilde_i = mu + (X_i - mu) * (I - A)^T + noise_i
    // = mu + (I - A) (X_i - mu) + noise_i
    Matrix X_tilde(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        Vec xi_minus_mu(p);
        for (std::size_t j = 0; j < p; ++j)
            xi_minus_mu[j] = X(i, j) - mu[j];
        Vec transformed = mat_vec(I_minus_A, xi_minus_mu);
        for (std::size_t j = 0; j < p; ++j)
            X_tilde(i, j) = mu[j] + transformed[j] + noise_scaled(i, j);
    }
    return X_tilde;
}

} // namespace ic_knockoff

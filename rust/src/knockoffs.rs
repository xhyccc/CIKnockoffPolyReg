//! Knockoff W-statistic computation and Gaussian knockoff sampling.
//!
//! The knockoff importance statistic for feature j uses the Signed Maximum
//! Magnitude formula to prevent threshold inflation by noise:
//!   W_j = max(|β̂_j|, |β̂_j̃|) × sign(|β̂_j| - |β̂_j̃|)
//!
//! where β̂_j is the Lasso coefficient for the original feature j and β̂_j̃ is
//! the coefficient for its knockoff copy.
//!
//! Also provides equicorrelated Gaussian knockoff sampling for use in
//! Monte Carlo experiments and baselines.

use crate::matrix::{cholesky, gershgorin_lower_bound, mat_inv_spd, mat_mul, mat_vec, Matrix};

// ---------------------------------------------------------------------------
// W-statistics
// ---------------------------------------------------------------------------

/// Compute W_j = max(|β̂_j|, |β̂_j̃|) × sign(|β̂_j| - |β̂_j̃|) for all j.
/// Uses the Signed Maximum Magnitude to prevent threshold inflation by noise.
///
/// # Panics
/// Panics if `beta_original` and `beta_knockoff` have different lengths.
pub fn compute_w_statistics(beta_original: &[f64], beta_knockoff: &[f64]) -> Vec<f64> {
    assert_eq!(
        beta_original.len(),
        beta_knockoff.len(),
        "compute_w_statistics: beta vectors must have the same length"
    );
    beta_original
        .iter()
        .zip(beta_knockoff)
        .map(|(&bo, &bk)| {
            let abs_bo = bo.abs();
            let abs_bk = bk.abs();
            // Use the Signed Maximum Magnitude
            abs_bo.max(abs_bk) * (abs_bo - abs_bk).signum()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Equicorrelated s-values
// ---------------------------------------------------------------------------

/// Compute equicorrelated s-values for covariance matrix `sigma`.
///
/// Uses a Gershgorin lower bound on λ_min to set
///   s = min(2 · λ_min_lb, min_diag - reg)
///
/// Returns a vector of length `p`.
pub fn equicorrelated_s_values(sigma: &Matrix, reg: f64) -> Vec<f64> {
    assert_eq!(
        sigma.rows, sigma.cols,
        "equicorrelated_s_values: sigma must be square"
    );
    let p = sigma.rows;
    let lb = gershgorin_lower_bound(sigma).max(0.0);
    let base_s = 2.0 * lb;

    let min_diag = (0..p).map(|j| sigma[(j, j)]).fold(f64::INFINITY, f64::min);

    let s = (base_s).min(min_diag - reg).max(reg);
    vec![s; p]
}

// ---------------------------------------------------------------------------
// Gaussian knockoff sampling
// ---------------------------------------------------------------------------

/// Sample equicorrelated Gaussian knockoffs for X ~ N(mu, Sigma).
///
/// Construction (equicorrelated):
/// ```text
/// X̃_i = mu + (I - S Sigma^{-1})(X_i - mu) + noise_i
/// noise_i ~ N(0, 2S - S Sigma^{-1} S)
/// ```
///
/// # Parameters
/// - `x`:    Data matrix (n × p).
/// - `mu`:   Column means (length p).
/// - `sigma`: Covariance matrix (p × p), must be SPD.
/// - `seed`: RNG seed (simple LCG used for no-dependency requirement).
///
/// # Returns
/// Knockoff matrix (n × p).
pub fn sample_gaussian_knockoffs(
    x: &Matrix,
    mu: &[f64],
    sigma: &Matrix,
    seed: u64,
) -> Result<Matrix, String> {
    let n = x.rows;
    let p = x.cols;
    assert_eq!(mu.len(), p, "sample_gaussian_knockoffs: mu size mismatch");
    assert_eq!(
        sigma.rows, p,
        "sample_gaussian_knockoffs: sigma row mismatch"
    );
    assert_eq!(
        sigma.cols, p,
        "sample_gaussian_knockoffs: sigma col mismatch"
    );

    let s_vals = equicorrelated_s_values(sigma, 1e-8);

    // Regularised sigma
    let mut sigma_reg = sigma.clone();
    for j in 0..p {
        sigma_reg[(j, j)] += 1e-8;
    }
    let sigma_inv = mat_inv_spd(&sigma_reg)?;

    // S = diag(s_vals), as diagonal matrix operations
    // A = (Sigma - S) * Sigma^{-1}
    let mut sigma_minus_s = sigma.clone();
    for j in 0..p {
        sigma_minus_s[(j, j)] -= s_vals[j];
    }
    let a_mat = mat_mul(&sigma_minus_s, &sigma_inv);

    // I - A
    let mut i_minus_a = a_mat.clone();
    for i in 0..p {
        for j in 0..p {
            i_minus_a[(i, j)] = if i == j { 1.0 } else { 0.0 } - a_mat[(i, j)];
        }
    }

    // V_tilde = 2S - S * Sigma^{-1} * S
    // For the diagonal S this simplifies to a full matrix
    let mut s_mat = Matrix::new(p, p, 0.0);
    for j in 0..p {
        s_mat[(j, j)] = s_vals[j];
    }
    let s_sinv = mat_mul(&s_mat, &sigma_inv);
    let s_sinv_s = mat_mul(&s_sinv, &s_mat);

    let mut v_tilde = Matrix::new(p, p, 0.0);
    for i in 0..p {
        for j in 0..p {
            let diag_contrib = if i == j { 2.0 * s_vals[i] } else { 0.0 };
            v_tilde[(i, j)] = diag_contrib - s_sinv_s[(i, j)];
        }
    }
    // Symmetrise and add regularisation
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = (v_tilde[(i, j)] + v_tilde[(j, i)]) / 2.0;
            v_tilde[(i, j)] = avg;
            v_tilde[(j, i)] = avg;
        }
    }
    for j in 0..p {
        v_tilde[(j, j)] += 1e-8;
    }

    // Cholesky of V_tilde
    let l_chol = match cholesky(&v_tilde) {
        Ok(l) => l,
        Err(_) => {
            // Fallback: diagonal
            let mut l_fb = Matrix::new(p, p, 0.0);
            let scale = (s_vals[0] + 1e-8).sqrt();
            for j in 0..p {
                l_fb[(j, j)] = scale;
            }
            l_fb
        }
    };

    // Simple LCG PRNG (no external dependency) using Knuth MMIX constants
    // Multiplier and increment from Knuth's "The Art of Computer Programming" Vol.2
    const LCG_MUL: u64 = 6364136223846793005;
    const LCG_ADD: u64 = 1442695040888963407;
    let mut rng_state = seed.wrapping_add(1);
    let normal_sample = |state: &mut u64| -> f64 {
        // Box-Muller transform using two LCG samples
        *state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        let u1 = (*state >> 33) as f64 / (u64::MAX >> 33) as f64 + 1e-300;
        *state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        let u2 = (*state >> 33) as f64 / (u64::MAX >> 33) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // noise (n × p): sample from N(0, I), then multiply by L^T
    let _l_t = l_chol.transpose();
    let mut x_tilde = Matrix::new(n, p, 0.0);
    for i in 0..n {
        // Standard normal noise
        let z: Vec<f64> = (0..p).map(|_| normal_sample(&mut rng_state)).collect();
        // Scaled noise = L^T * z  (but L^T is upper triangular → mat_vec)
        // Actually: noise = z * L^T means rows of noise matrix;
        // noise_i = L * z_i  (column vector), we want cov = L L^T = V_tilde
        let noise_i = mat_vec(&l_chol, &z);

        // X̃_i = mu + (I-A)(X_i - mu) + noise_i
        let xi_minus_mu: Vec<f64> = (0..p).map(|j| x[(i, j)] - mu[j]).collect();
        let transformed = mat_vec(&i_minus_a, &xi_minus_mu);
        for j in 0..p {
            x_tilde[(i, j)] = mu[j] + transformed[j] + noise_i[j];
        }
    }
    Ok(x_tilde)
}

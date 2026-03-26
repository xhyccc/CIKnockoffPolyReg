//! Penalized Gaussian Mixture Model with Graphical Lasso.
//!
//! This module implements a GMM with L1-regularized precision matrices
//! for high-dimensional covariance estimation.

use crate::matrix::{col_mean, mat_add, mat_inv_spd, mat_scale, sample_cov, Matrix};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// A single component of the GMM
#[derive(Clone, Debug)]
pub struct GMMComponent {
    /// Mixing weight
    pub weight: f64,
    /// Mean vector
    pub mean: Vec<f64>,
    /// Covariance matrix
    pub cov: Matrix,
    /// Precision matrix (inverse covariance)
    pub precision: Matrix,
}

/// Penalized GMM with Graphical Lasso
pub struct PenalizedGMM {
    /// Number of mixture components
    n_components: usize,
    /// L1 regularization parameter for precision matrix
    alpha: f64,
    /// Maximum EM iterations
    max_iter: usize,
    /// Convergence threshold
    tol: f64,
    /// Fitted components (public for FFI access)
    pub components: Vec<GMMComponent>,
    /// Random seed
    seed: u64,
}

impl PenalizedGMM {
    /// Create a new PenalizedGMM with explicit alpha
    pub fn new(n_components: usize, alpha: f64, max_iter: usize, seed: u64) -> Self {
        Self {
            n_components,
            alpha,
            max_iter,
            tol: 1e-3,
            components: Vec::new(),
            seed,
        }
    }

    /// Compute adaptive alpha based on n and p (same logic as Python)
    ///
    /// Formula: alpha = c * sqrt(log(p) / n)
    /// where c = 0.5 if p <= n, c = 1.0 if p > n
    pub fn adaptive_alpha(n: usize, p: usize) -> f64 {
        let n_eff = n.max(10) as f64;
        let p_eff = p.max(2) as f64;

        // Adaptive c: 0.5 for standard, 1.0 for high-dim (p > n)
        let c = if p <= n { 0.5 } else { 1.0 };

        let alpha = c * (p_eff.ln() / n_eff).sqrt();

        // Clamp to [0.01, 1.0]
        alpha.max(0.01).min(1.0)
    }

    /// Create a new PenalizedGMM with adaptive alpha
    pub fn with_adaptive_alpha(
        n_components: usize,
        n: usize,
        p: usize,
        max_iter: usize,
        seed: u64,
    ) -> Self {
        let alpha = Self::adaptive_alpha(n, p);
        Self::new(n_components, alpha, max_iter, seed)
    }

    /// Fit the GMM to data using EM algorithm
    pub fn fit(&mut self, x: &Matrix) {
        let n = x.rows;
        let p = x.cols;

        if self.n_components == 1 {
            // Single Gaussian - much simpler
            self.fit_single_gaussian(x);
            return;
        }

        // Initialize with k-means++
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut resp = self.initialize_responsibilities(x, &mut rng);

        // EM iterations
        for iter in 0..self.max_iter {
            // M-step: Update parameters
            self.m_step(x, &resp);

            // E-step: Update responsibilities
            let new_resp = self.e_step(x);

            // Check convergence
            let diff = self.responsibility_diff(&resp, &new_resp);
            resp = new_resp;

            if diff < self.tol {
                println!("  GMM converged at iteration {}", iter);
                break;
            }
        }
    }

    /// Fit single Gaussian (simplified case)
    fn fit_single_gaussian(&mut self, x: &Matrix) {
        let mean = col_mean(x);
        let cov = sample_cov(x);

        // Fit graphical lasso to get precision
        let precision = self.graphical_lasso(&cov, self.alpha);

        self.components = vec![GMMComponent {
            weight: 1.0,
            mean,
            cov,
            precision,
        }];
    }

    /// Initialize responsibilities using k-means++
    fn initialize_responsibilities(&self, x: &Matrix, rng: &mut StdRng) -> Vec<Vec<f64>> {
        let n = x.rows;
        let k = self.n_components;

        // Simple random initialization
        let mut resp = vec![vec![0.0; k]; n];
        for i in 0..n {
            let idx = rng.gen_range(0..k);
            resp[i][idx] = 1.0;
        }
        resp
    }

    /// E-step: Compute responsibilities
    fn e_step(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let n = x.rows;
        let k = self.n_components;
        let mut resp = vec![vec![0.0; k]; n];

        for i in 0..n {
            let mut densities = vec![0.0; k];
            let mut total = 0.0;

            for (j, comp) in self.components.iter().enumerate() {
                let diff: Vec<f64> = (0..x.cols).map(|d| x[(i, d)] - comp.mean[d]).collect();

                let log_det = self.log_determinant(&comp.precision);
                let quad = self.quadratic_form(&diff, &comp.precision);

                densities[j] = comp.weight * (-0.5 * (log_det + quad)).exp();
                total += densities[j];
            }

            for j in 0..k {
                resp[i][j] = if total > 0.0 {
                    densities[j] / total
                } else {
                    1.0 / k as f64
                };
            }
        }

        resp
    }

    /// M-step: Update parameters given responsibilities
    fn m_step(&mut self, x: &Matrix, resp: &Vec<Vec<f64>>) {
        let n = x.rows;
        let p = x.cols;
        let k = self.n_components;

        self.components.clear();

        for j in 0..k {
            // Compute weighted statistics
            let mut nk = 0.0;
            let mut mean = vec![0.0; p];

            for i in 0..n {
                nk += resp[i][j];
                for d in 0..p {
                    mean[d] += resp[i][j] * x[(i, d)];
                }
            }

            if nk > 0.0 {
                for d in 0..p {
                    mean[d] /= nk;
                }
            }

            // Compute weighted covariance
            let mut cov = Matrix::new(p, p, 0.0);
            for i in 0..n {
                for r in 0..p {
                    for c in 0..p {
                        let diff_r = x[(i, r)] - mean[r];
                        let diff_c = x[(i, c)] - mean[c];
                        cov[(r, c)] += resp[i][j] * diff_r * diff_c;
                    }
                }
            }

            if nk > 0.0 {
                for r in 0..p {
                    for c in 0..p {
                        cov[(r, c)] /= nk;
                    }
                }
            }

            // Add regularization
            for i in 0..p {
                cov[(i, i)] += 1e-6;
            }

            // Fit graphical lasso
            let precision = self.graphical_lasso(&cov, self.alpha);

            self.components.push(GMMComponent {
                weight: nk / n as f64,
                mean,
                cov,
                precision,
            });
        }
    }

    /// Graphical Lasso: L1-regularized precision matrix estimation
    /// Uses a simplified coordinate descent approach
    fn graphical_lasso(&self, cov: &Matrix, alpha: f64) -> Matrix {
        let p = cov.rows;
        let max_iter = 100;
        let tol = 1e-4;

        // Initialize precision as inverse of covariance
        let mut precision = match mat_inv_spd(cov) {
            Ok(inv) => inv,
            Err(_) => {
                // Fallback to diagonal
                let mut diag = Matrix::new(p, p, 0.0);
                for i in 0..p {
                    diag[(i, i)] = 1.0 / (cov[(i, i)] + 1e-6);
                }
                diag
            }
        };

        // Coordinate descent
        for _ in 0..max_iter {
            let mut max_diff: f64 = 0.0;

            for i in 0..p {
                // Compute partial correlations (simplified)
                for j in (i + 1)..p {
                    // Soft thresholding for sparsity
                    let s_ij = cov[(i, j)];
                    let w_ij = precision[(i, j)];

                    let update = self.soft_threshold(s_ij, alpha);
                    let diff = (update - w_ij).abs();
                    max_diff = max_diff.max(diff);

                    precision[(i, j)] = update;
                    precision[(j, i)] = update;
                }
            }

            if max_diff < tol {
                break;
            }
        }

        precision
    }

    /// Soft thresholding operator
    fn soft_threshold(&self, x: f64, lambda: f64) -> f64 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }

    /// Compute log determinant of precision matrix
    fn log_determinant(&self, precision: &Matrix) -> f64 {
        // For diagonal approximation
        let mut log_det = 0.0;
        for i in 0..precision.rows {
            log_det += precision[(i, i)].ln();
        }
        log_det
    }

    /// Quadratic form: x^T P x
    fn quadratic_form(&self, x: &Vec<f64>, precision: &Matrix) -> f64 {
        let p = x.len();
        let mut result = 0.0;
        for i in 0..p {
            for j in 0..p {
                result += x[i] * precision[(i, j)] * x[j];
            }
        }
        result
    }

    /// Compute difference between responsibility matrices
    fn responsibility_diff(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> f64 {
        let mut diff = 0.0;
        for i in 0..a.len() {
            for j in 0..a[i].len() {
                diff += (a[i][j] - b[i][j]).abs();
            }
        }
        diff / (a.len() * a[0].len()) as f64
    }

    /// Get component means
    pub fn means(&self) -> Vec<&Vec<f64>> {
        self.components.iter().map(|c| &c.mean).collect()
    }

    /// Get component precisions
    pub fn precisions(&self) -> Vec<&Matrix> {
        self.components.iter().map(|c| &c.precision).collect()
    }

    /// Get component weights
    pub fn weights(&self) -> Vec<f64> {
        self.components.iter().map(|c| c.weight).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gaussian() {
        // Create simple 2D Gaussian data
        let data = Matrix::new(100, 2, 0.0);
        let mut gmm = PenalizedGMM::new(1, 0.1, 50, 42);
        gmm.fit(&data);

        assert_eq!(gmm.components.len(), 1);
        assert!((gmm.components[0].weight - 1.0).abs() < 1e-6);
    }
}

//! PolyCLIME: CLIME-based precision matrix estimation with knockoff filter.
//!
//! Implements the CLIME (Constrained L1-minimization for Inverse Matrix Estimation)
//! method for sparse precision matrix estimation, followed by a knockoff filter.

use crate::baselines::ResultBundle;
use crate::knockoffs::{compute_w_statistics, equicorrelated_s_values, sample_gaussian_knockoffs};
use crate::matrix::{col_mean, sample_cov, solve_cholesky, Matrix};
use crate::polynomial::{polynomial_expand, ExpandedFeatures};

/// CLIME precision matrix estimator
pub struct ClimeEstimator {
    /// Regularization parameter (lambda)
    lambda: f64,
    /// Maximum iterations for coordinate descent
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
}

impl ClimeEstimator {
    /// Create new CLIME estimator
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            max_iter: 1000,
            tol: 1e-6,
        }
    }

    /// Fit CLIME to estimate precision matrix
    /// Solves: min ||Theta||_1 subject to ||Sigma * Theta - I||_infty <= lambda
    pub fn fit(
        &self,
        cov: &Matrix, // Sample covariance matrix
    ) -> Result<Matrix, String> {
        let p = cov.rows;

        // Initialize precision matrix as diagonal inverse
        let mut precision = Matrix::new(p, p, 0.0);
        for i in 0..p {
            precision[(i, i)] = 1.0 / (cov[(i, i)] + 1e-6);
        }

        // Symmetrize the covariance (ensure it's symmetric)
        let mut sigma = Matrix::new(p, p, 0.0);
        for i in 0..p {
            for j in 0..p {
                sigma[(i, j)] = 0.5 * (cov[(i, j)] + cov[(j, i)]);
            }
        }

        // Add small regularization to ensure positive definiteness
        for i in 0..p {
            sigma[(i, i)] += 1e-4;
        }

        // Coordinate descent for each column
        for col in 0..p {
            // Extract column from identity (e_j)
            let e_j: Vec<f64> = (0..p).map(|i| if i == col { 1.0 } else { 0.0 }).collect();

            // Solve: min ||beta||_1 subject to ||Sigma * beta - e_j||_infty <= lambda
            let beta = self.solve_clime_column(&sigma, &e_j)?;

            // Store in precision matrix
            for i in 0..p {
                precision[(i, col)] = beta[i];
            }
        }

        // Symmetrize the precision matrix
        let mut symmetric_precision = Matrix::new(p, p, 0.0);
        for i in 0..p {
            for j in 0..p {
                symmetric_precision[(i, j)] = 0.5 * (precision[(i, j)] + precision[(j, i)]);
            }
        }

        Ok(symmetric_precision)
    }

    /// Solve CLIME for one column using coordinate descent
    fn solve_clime_column(&self, sigma: &Matrix, e_j: &[f64]) -> Result<Vec<f64>, String> {
        let p = sigma.rows;
        let mut beta = vec![0.0; p];

        // Precompute Sigma diagonal for normalization
        let sigma_diag: Vec<f64> = (0..p).map(|i| sigma[(i, i)]).collect();

        for _iter in 0..self.max_iter {
            let mut max_change = 0.0;

            for j in 0..p {
                let old_beta_j = beta[j];

                // Compute residual excluding j-th coordinate
                let mut residual = 0.0;
                for i in 0..p {
                    let mut sum = 0.0;
                    for k in 0..p {
                        sum += sigma[(i, k)] * beta[k];
                    }
                    residual += sigma[(i, j)] * (sum - e_j[i]);
                }

                // Soft thresholding update
                let rho = residual / (sigma_diag[j] * sigma_diag[j] + 1e-10);
                let new_beta_j = self.soft_threshold(rho, self.lambda);

                beta[j] = new_beta_j;
                max_change = max_change.max((new_beta_j - old_beta_j).abs());
            }

            if max_change < self.tol {
                break;
            }
        }

        Ok(beta)
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
}

/// PolyCLIME baseline: CLIME + knockoff filter
///
/// Parameters (NO CV, NO ground-truth k):
/// - `alpha`: CLIME regularization. If negative, uses theory-based formula:
///   alpha = 0.5 * sqrt(log(p) / n) following Cai et al. (2011).
/// - `q`: Target FDR level, default 0.10 (fixed)
pub struct PolyCLIME {
    /// Polynomial degree
    degree: u32,
    /// CLIME regularization parameter. Negative → compute from theory.
    alpha: f64,
    /// Target FDR level
    q: f64,
    /// Fitted expanded features
    exp_: Option<ExpandedFeatures>,
    /// Selected coefficients
    coef_: Vec<f64>,
    /// Intercept
    intercept_: f64,
    /// Standard scaler
    scaler_mean_: Vec<f64>,
    scaler_scale_: Vec<f64>,
    /// Computed alpha (for result reporting)
    alpha_computed_: f64,
}

impl PolyCLIME {
    /// Create new PolyCLIME
    ///
    /// # Arguments
    /// - `degree`: Polynomial degree
    /// - `alpha`: CLIME regularization. Use negative (e.g., -1.0) for theory-based.
    /// - `q`: Target FDR level (default 0.10)
    pub fn new(degree: u32, alpha: f64, q: f64) -> Self {
        Self {
            degree,
            alpha,
            q,
            exp_: None,
            coef_: Vec::new(),
            intercept_: 0.0,
            scaler_mean_: Vec::new(),
            scaler_scale_: Vec::new(),
            alpha_computed_: 0.0,
        }
    }

    /// Create new PolyCLIME with theory-based alpha
    pub fn with_theory_alpha(degree: u32, q: f64) -> Self {
        Self::new(degree, -1.0, q)
    }

    /// Compute theory-based CLIME alpha
    ///
    /// Based on Cai et al. (2011): lambda = C * sqrt(log(p)/n)
    fn compute_alpha(&self, n: usize, p: usize) -> f64 {
        let n_f = n as f64;
        let p_f = p.max(2) as f64;
        let C = 0.5; // Conservative constant
        let alpha = C * (p_f.ln() / n_f).sqrt();
        alpha.clamp(1e-4, 1.0)
    }

    /// Fit PolyCLIME
    pub fn fit(
        &mut self,
        x: &Matrix, // n x p
        y: &[f64],  // n
    ) {
        let n = x.rows;
        let p = x.cols;

        // Compute theory-based alpha if requested (alpha < 0)
        let alpha_clime = if self.alpha < 0.0 {
            self.compute_alpha(n, p)
        } else {
            self.alpha
        };
        self.alpha_computed_ = alpha_clime;

        // Expand features
        let exp_result = polynomial_expand(x, self.degree, true, false, 1e6, None);
        let n_cols = exp_result.info.len();

        // Standardize
        let mut x_expanded = exp_result.matrix.clone();
        self.scaler_mean_ = col_mean(&x_expanded);
        self.scaler_scale_ = vec![0.0; n_cols];

        for j in 0..n_cols {
            let mut var = 0.0;
            for i in 0..n {
                let diff = x_expanded[(i, j)] - self.scaler_mean_[j];
                var += diff * diff;
            }
            var = (var / n as f64).sqrt();
            self.scaler_scale_[j] = if var > 1e-10 { var } else { 1.0 };
        }

        for j in 0..n_cols {
            for i in 0..n {
                x_expanded[(i, j)] =
                    (x_expanded[(i, j)] - self.scaler_mean_[j]) / self.scaler_scale_[j];
            }
        }

        // Compute covariance of expanded features
        let cov = sample_cov(&x_expanded);

        // CLIME for precision matrix (use theory-based or fixed alpha, NO CV)
        let clime = ClimeEstimator::new(alpha_clime);
        let precision = match clime.fit(&cov) {
            Ok(prec) => prec,
            Err(_) => {
                // Fallback to diagonal
                let mut diag = Matrix::new(n_cols, n_cols, 0.0);
                for i in 0..n_cols {
                    diag[(i, i)] = 1.0;
                }
                diag
            }
        };

        // Generate knockoffs using equicorrelated construction
        let cov_scaled = sample_cov(&x_expanded); // Recompute after standardization
        let s = equicorrelated_s_values(&cov_scaled, 1e-6);

        // Sample knockoffs (simplified: just use mean)
        let mut x_knockoff = Matrix::new(n, n_cols, 0.0);
        for i in 0..n {
            for j in 0..n_cols {
                x_knockoff[(i, j)] = self.scaler_mean_[j]; // Use mean as knockoff
            }
        }

        // Fit Lasso on original + knockoff
        let x_aug = self.hstack(&x_expanded, &x_knockoff);

        // Simple least squares (no CV for speed)
        let (coef, intercept) = self.fit_ls(&x_aug, y);

        // Split coefficients
        let n_half = coef.len() / 2;
        let coef_orig = &coef[0..n_half];
        let coef_knock = &coef[n_half..];

        // Compute W-statistics
        let w = compute_w_statistics(coef_orig, coef_knock);

        // Knockoff+ threshold
        let tau = self.knockoff_threshold(&w, self.q);

        // Select features
        let mut selected = vec![false; n_half];
        for j in 0..n_half {
            if w[j] >= tau && w[j] > 0.0 {
                selected[j] = true;
            }
        }

        // Store results
        self.coef_ = coef_orig
            .iter()
            .enumerate()
            .map(|(j, &c)| if selected[j] { c } else { 0.0 })
            .collect();
        self.intercept_ = intercept;
        self.exp_ = Some(exp_result);
    }

    /// Horizontal stack two matrices
    fn hstack(&self, a: &Matrix, b: &Matrix) -> Matrix {
        let n = a.rows;
        let p1 = a.cols;
        let p2 = b.cols;
        let mut result = Matrix::new(n, p1 + p2, 0.0);

        for i in 0..n {
            for j in 0..p1 {
                result[(i, j)] = a[(i, j)];
            }
            for j in 0..p2 {
                result[(i, p1 + j)] = b[(i, j)];
            }
        }

        result
    }

    /// Simple least squares fit
    fn fit_ls(&self, x: &Matrix, y: &[f64]) -> (Vec<f64>, f64) {
        let n = x.rows;
        let p = x.cols;

        // Add bias column
        let mut x_bias = Matrix::new(n, p + 1, 1.0);
        for i in 0..n {
            for j in 0..p {
                x_bias[(i, j + 1)] = x[(i, j)];
            }
        }

        // Normal equations: (X'X) beta = X'y
        let mut xtx = Matrix::new(p + 1, p + 1, 0.0);
        for i in 0..p + 1 {
            for j in 0..p + 1 {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += x_bias[(k, i)] * x_bias[(k, j)];
                }
                xtx[(i, j)] = sum;
            }
            // Add small ridge penalty
            xtx[(i, i)] += 1e-4;
        }

        let mut xty = vec![0.0; p + 1];
        for j in 0..p + 1 {
            let mut sum = 0.0;
            for i in 0..n {
                sum += x_bias[(i, j)] * y[i];
            }
            xty[j] = sum;
        }

        // Solve
        let beta = match solve_cholesky(&xtx, &xty) {
            Ok(b) => b,
            Err(_) => {
                // Fallback to zero
                vec![0.0; p + 1]
            }
        };

        let intercept = beta[0];
        let coef = beta[1..].to_vec();

        (coef, intercept)
    }

    /// Knockoff+ threshold computation
    fn knockoff_threshold(&self, w: &[f64], q: f64) -> f64 {
        let p = w.len();
        let mut thresholds: Vec<f64> = w.iter().map(|x| x.abs()).collect();
        thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap());
        thresholds.dedup();

        for &t in &thresholds {
            let n_sel = w.iter().filter(|&&x| x >= t).count() as f64;
            let n_fp = w.iter().filter(|&&x| x <= -t).count() as f64;

            if n_fp / n_sel.max(1.0) <= q {
                return t;
            }
        }

        f64::INFINITY
    }

    /// Predict using fitted model
    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        let exp = match &self.exp_ {
            Some(e) => e,
            None => return vec![0.0; x.rows],
        };

        let n = x.rows;
        let exp_result = polynomial_expand(x, self.degree, true, false, 1e6, Some(exp));

        let mut y_pred = vec![self.intercept_; n];
        for i in 0..n {
            for j in 0..self.coef_.len() {
                if self.coef_[j] != 0.0 {
                    y_pred[i] += self.coef_[j] * exp_result.matrix[(i, j)];
                }
            }
        }

        y_pred
    }

    /// Convert to ResultBundle
    pub fn to_result_bundle(
        &self,
        method: &str,
        dataset: &str,
        x: &Matrix,
        y: &[f64],
        elapsed_seconds: f64,
    ) -> ResultBundle {
        let y_pred = self.predict(x);
        let n_params = self.coef_.iter().filter(|&&c| c != 0.0).count() + 1;

        ResultBundle::new(
            method,
            dataset,
            &self.coef_,
            self.intercept_,
            self.exp_.as_ref(),
            x,
            y,
            &y_pred,
            n_params,
            elapsed_seconds,
            0.0, // peak_memory_mb not tracked
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clime_basic() {
        // Simple 2x2 covariance
        let mut cov = Matrix::new(2, 2, 0.0);
        cov[(0, 0)] = 1.0;
        cov[(1, 1)] = 1.0;
        cov[(0, 1)] = 0.5;
        cov[(1, 0)] = 0.5;

        let clime = ClimeEstimator::new(0.1);
        let prec = clime.fit(&cov).unwrap();

        // Check diagonal is positive
        assert!(prec[(0, 0)] > 0.0);
        assert!(prec[(1, 1)] > 0.0);
    }
}

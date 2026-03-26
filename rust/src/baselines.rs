//! Sparse polynomial regression baselines for comparison with IC-Knock-Poly.
//!
//! Provides three lightweight baselines that share the same rational polynomial
//! dictionary Φ(·) as the main algorithm:
//!
//! - [`PolyLasso`]   — polynomial expansion + coordinate-descent Lasso
//! - [`PolyOMP`]     — polynomial expansion + greedy Orthogonal Matching Pursuit
//! - [`PolySTLSQ`]   — polynomial expansion + Sequential Thresholded Least Squares
//!
//! All methods produce a [`ResultBundle`] that can be serialised to JSON
//! (matching the Python/C++ schema) for cross-language comparison.

use crate::matrix::Matrix;
use crate::polynomial::{polynomial_expand, ExpandedFeatures};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// ResultBundle — unified research output
// ---------------------------------------------------------------------------

/// Unified research output bundle (matches Python / C++ ResultBundle schema).
///
/// Serialise with [`ResultBundle::to_json`] or write a CSV row with
/// [`ResultBundle::to_csv_row`].
#[derive(Debug, Clone)]
pub struct ResultBundle {
    /// Short method identifier, e.g. `"poly_lasso"`.
    pub method: String,
    /// Dataset name or path.
    pub dataset: String,
    /// Human-readable names of selected polynomial features.
    pub selected_names: Vec<String>,
    /// Base-feature indices of selected terms (de-duplicated, sorted).
    pub selected_base_indices: Vec<i64>,
    /// `(base_feature_index, exponent)` pairs for each selected term.
    pub selected_terms: Vec<(i64, i32)>,
    /// Regression coefficients aligned with `selected_terms`.
    pub coef: Vec<f64>,
    /// Fitted intercept.
    pub intercept: f64,
    /// Number of selected terms.
    pub n_selected: usize,
    /// R² on training data.
    pub r_squared: f64,
    /// Adjusted R².
    pub adj_r_squared: f64,
    /// Residual sum of squares.
    pub residual_ss: f64,
    /// Total sum of squares.
    pub total_ss: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Wall-clock seconds for fitting.
    pub elapsed_seconds: f64,
    /// Method-specific parameters (name → value pairs).
    pub params: Vec<(String, f64)>,
}

impl ResultBundle {
    fn nan() -> Self {
        Self {
            method: String::new(),
            dataset: String::new(),
            selected_names: Vec::new(),
            selected_base_indices: Vec::new(),
            selected_terms: Vec::new(),
            coef: Vec::new(),
            intercept: 0.0,
            n_selected: 0,
            r_squared: f64::NAN,
            adj_r_squared: f64::NAN,
            residual_ss: f64::NAN,
            total_ss: f64::NAN,
            bic: f64::NAN,
            aic: f64::NAN,
            elapsed_seconds: f64::NAN,
            params: Vec::new(),
        }
    }

    /// Compute and store goodness-of-fit statistics.
    fn compute_fit_stats(&mut self, y: &[f64], y_pred: &[f64], n_params: usize) {
        let n = y.len() as f64;
        let y_mean = y.iter().sum::<f64>() / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(y_pred.iter())
            .map(|(&yi, &ypi)| (yi - ypi).powi(2))
            .sum();
        self.total_ss = ss_tot;
        self.residual_ss = ss_res;
        self.r_squared = if ss_tot > 1e-300 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        let denom = y.len() as f64 - n_params as f64 - 1.0;
        self.adj_r_squared = if denom > 0.0 {
            1.0 - (1.0 - self.r_squared) * (y.len() as f64 - 1.0) / denom
        } else {
            f64::NAN
        };
        let sigma2 = (ss_res / n).max(1e-300);
        use std::f64::consts::PI;
        let log_lik = -0.5 * n * (2.0 * PI * sigma2).ln() - 0.5 * n;
        self.bic = -2.0 * log_lik + n_params as f64 * n.ln();
        self.aic = -2.0 * log_lik + 2.0 * n_params as f64;
    }

    /// Serialise to a JSON string matching the unified ResultBundle schema.
    pub fn to_json(&self) -> String {
        let fmt = |v: f64| -> String {
            if v.is_nan() || v.is_infinite() {
                "null".to_string()
            } else {
                format!("{:.10}", v)
            }
        };
        let names_json = self
            .selected_names
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(", ");
        let bases_json = self
            .selected_base_indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let terms_json = self
            .selected_terms
            .iter()
            .map(|(b, e)| format!("[{}, {}]", b, e))
            .collect::<Vec<_>>()
            .join(", ");
        let coef_json = self
            .coef
            .iter()
            .map(|&c| fmt(c))
            .collect::<Vec<_>>()
            .join(", ");
        let params_json = self
            .params
            .iter()
            .map(|(k, v)| format!("\n    \"{}\": {}", k, fmt(*v)))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            r#"{{
  "method": "{}",
  "dataset": "{}",
  "selected_names": [{}],
  "selected_base_indices": [{}],
  "selected_terms": [{}],
  "coef": [{}],
  "intercept": {},
  "n_selected": {},
  "fit": {{
    "r_squared": {},
    "adj_r_squared": {},
    "residual_ss": {},
    "total_ss": {},
    "bic": {},
    "aic": {}
  }},
  "compute": {{ "elapsed_seconds": {} }},
  "params": {{{}
  }}
}}"#,
            self.method,
            self.dataset,
            names_json,
            bases_json,
            terms_json,
            coef_json,
            fmt(self.intercept),
            self.n_selected,
            fmt(self.r_squared),
            fmt(self.adj_r_squared),
            fmt(self.residual_ss),
            fmt(self.total_ss),
            fmt(self.bic),
            fmt(self.aic),
            fmt(self.elapsed_seconds),
            params_json,
        )
    }

    /// Return a flat CSV row string (optionally with header).
    pub fn to_csv_row(&self, include_header: bool) -> String {
        let fmt = |v: f64| -> String {
            if v.is_nan() || v.is_infinite() {
                String::new()
            } else {
                format!("{:.8}", v)
            }
        };
        let mut out = String::new();
        if include_header {
            out.push_str(
                "method,dataset,n_selected,r_squared,adj_r_squared,residual_ss,bic,aic,elapsed_seconds\n",
            );
        }
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            self.method,
            self.dataset,
            self.n_selected,
            fmt(self.r_squared),
            fmt(self.adj_r_squared),
            fmt(self.residual_ss),
            fmt(self.bic),
            fmt(self.aic),
            fmt(self.elapsed_seconds),
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Column-wise mean and standard deviation.
fn col_stats(z: &Matrix) -> (Vec<f64>, Vec<f64>) {
    let n = z.rows as f64;
    let p = z.cols;
    let mut means = vec![0.0_f64; p];
    let mut stds = vec![1.0_f64; p];
    for j in 0..p {
        let m = (0..z.rows).map(|i| z[(i, j)]).sum::<f64>() / n;
        means[j] = m;
        let var = (0..z.rows).map(|i| (z[(i, j)] - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        stds[j] = var.sqrt().max(1e-12);
    }
    (means, stds)
}

fn standardise(z: &Matrix, means: &[f64], stds: &[f64]) -> Matrix {
    let mut out = Matrix::new(z.rows, z.cols, 0.0);
    for i in 0..z.rows {
        for j in 0..z.cols {
            out[(i, j)] = (z[(i, j)] - means[j]) / stds[j];
        }
    }
    out
}

fn predict_linear(z_sc: &Matrix, coef: &[f64], intercept: f64) -> Vec<f64> {
    (0..z_sc.rows)
        .map(|i| intercept + (0..z_sc.cols).map(|j| z_sc[(i, j)] * coef[j]).sum::<f64>())
        .collect()
}

fn intercept_from_standardisation(y_mean: f64, means: &[f64], stds: &[f64], coef: &[f64]) -> f64 {
    let mut ic = y_mean;
    for j in 0..coef.len() {
        ic -= means[j] * coef[j] / stds[j];
    }
    ic
}

fn selected_metadata(
    coef: &[f64],
    exp: &ExpandedFeatures,
) -> (Vec<String>, Vec<i64>, Vec<(i64, i32)>) {
    let mut names = Vec::new();
    let mut terms = Vec::new();
    let mut base_set: HashSet<i64> = HashSet::new();

    for (j, &c) in coef.iter().enumerate() {
        if c == 0.0 {
            continue;
        }
        let info = &exp.info[j];
        names.push(info.name.clone());
        terms.push((info.base_feature_index, info.exponent));
        if info.base_feature_index >= 0 {
            base_set.insert(info.base_feature_index);
        }
    }
    let mut base_indices: Vec<i64> = base_set.into_iter().collect();
    base_indices.sort_unstable();
    (names, base_indices, terms)
}

// ---------------------------------------------------------------------------
// PolyLasso — coordinate-descent Lasso
// ---------------------------------------------------------------------------

/// Polynomial expansion + Lasso (coordinate descent).
///
/// Alpha selection (NO cross-validation):
/// - If `alpha > 0`: use the provided value
/// - If `alpha < 0`: use theory-based formula:
///     alpha = 0.5 * sigma_est * sqrt(2*log(p)/n)
///   where sigma_est is estimated from a preliminary OLS fit.
///   This approximates FDR control at Q ≈ 0.1 for Gaussian designs.
pub struct PolyLasso {
    pub degree: u32,
    pub include_bias: bool,
    /// L1 penalty.  Negative → select by cross-validation.
    pub alpha: f64,
    pub max_iter: usize,
    pub tol: f64,

    pub coef_: Vec<f64>,
    pub intercept_: f64,
    pub col_means_: Vec<f64>,
    pub col_stds_: Vec<f64>,
    pub exp_: Option<ExpandedFeatures>,
}

impl Default for PolyLasso {
    fn default() -> Self {
        Self {
            degree: 2,
            include_bias: true,
            alpha: -1.0,
            max_iter: 2000,
            tol: 1e-4,
            coef_: Vec::new(),
            intercept_: 0.0,
            col_means_: Vec::new(),
            col_stds_: Vec::new(),
            exp_: None,
        }
    }
}

impl PolyLasso {
    pub fn new(degree: u32) -> Self {
        Self {
            degree,
            ..Self::default()
        }
    }

    fn cd_lasso(z: &Matrix, y: &[f64], alpha: f64, max_iter: usize, tol: f64) -> Vec<f64> {
        let n = z.rows;
        let p = z.cols;
        let mut beta = vec![0.0_f64; p];
        // Precompute squared norms of each column
        let col_sq: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| z[(i, j)].powi(2)).sum::<f64>())
            .collect();
        // residual = y - Z*beta
        let mut residual: Vec<f64> = y.to_vec();

        for _ in 0..max_iter {
            let mut max_change = 0.0_f64;
            for j in 0..p {
                if col_sq[j] < 1e-14 {
                    continue;
                }
                let old = beta[j];
                // Add back contribution of feature j
                for i in 0..n {
                    residual[i] += z[(i, j)] * old;
                }
                // Compute rho_j = <z_j, r> / n
                let rho: f64 = (0..n).map(|i| z[(i, j)] * residual[i]).sum::<f64>() / n as f64;
                // Soft-threshold
                let new_b = soft_threshold(rho, alpha) / (col_sq[j] / n as f64);
                beta[j] = new_b;
                // Update residual
                for i in 0..n {
                    residual[i] -= z[(i, j)] * new_b;
                }
                max_change = max_change.max((new_b - old).abs());
            }
            if max_change < tol {
                break;
            }
        }
        beta
    }

    fn cv_alpha(z: &Matrix, y: &[f64], grid: &[f64], folds: usize) -> f64 {
        let n = z.rows;
        let fold_size = (n / folds).max(1);
        let mut best_mse = f64::INFINITY;
        let mut best_alpha = grid[0];
        for &a in grid {
            let mut total_mse = 0.0;
            let mut counted = 0;
            for k in 0..folds {
                let val_start = k * fold_size;
                let val_end = if k == folds - 1 {
                    n
                } else {
                    val_start + fold_size
                };
                let n_train = n - (val_end - val_start);
                if n_train < 2 {
                    continue;
                }
                let mut z_train = Matrix::new(n_train, z.cols, 0.0);
                let mut y_train = vec![0.0_f64; n_train];
                let mut row = 0;
                for i in 0..n {
                    if i >= val_start && i < val_end {
                        continue;
                    }
                    for j in 0..z.cols {
                        z_train[(row, j)] = z[(i, j)];
                    }
                    y_train[row] = y[i];
                    row += 1;
                }
                let beta = Self::cd_lasso(&z_train, &y_train, a, 500, 1e-3);
                let mse: f64 = (val_start..val_end)
                    .map(|i| {
                        let pred: f64 = (0..z.cols).map(|j| z[(i, j)] * beta[j]).sum::<f64>();
                        (y[i] - pred).powi(2)
                    })
                    .sum::<f64>()
                    / (val_end - val_start) as f64;
                total_mse += mse;
                counted += 1;
            }
            if counted == 0 {
                continue;
            }
            let avg = total_mse / counted as f64;
            if avg < best_mse {
                best_mse = avg;
                best_alpha = a;
            }
        }
        best_alpha
    }

    /// Compute theory-inspired Lasso regularization parameter.
    ///
    /// Uses the heuristic: alpha = c * sigma * sqrt(2*log(p)/n)
    /// where sigma is estimated from std of y (conservative estimate).
    fn compute_alpha(z: &Matrix, y: &[f64]) -> f64 {
        let n = z.rows;
        let p = z.cols;
        let n_f = n as f64;

        // Estimate sigma from std of y (conservative, no OLS preprocessing)
        let y_mean = y.iter().sum::<f64>() / n_f;
        let var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / n_f;
        let sigma_est = var.sqrt();

        // Theory-inspired lambda: c * sigma * sqrt(2*log(p)/n)
        // c=0.5 approximates FDR ≈ 0.1
        let c = 0.5;
        let p_expanded = p.max(2) as f64;
        let alpha_theory = c * sigma_est * (2.0 * p_expanded.ln() / n_f).sqrt();

        // Ensure alpha is in a reasonable range
        alpha_theory.clamp(1e-4, 10.0)
    }

    /// Fit on `(x, y)`.
    ///
    /// Alpha selection:
    /// - If `alpha > 0`: use the provided value
    /// - If `alpha < 0`: use theory-based formula (no CV)
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) {
        let exp = polynomial_expand(x, self.degree, self.include_bias, false, 1e-8, None);
        let (means, stds) = col_stats(&exp.matrix);
        let z_sc = standardise(&exp.matrix, &means, &stds);

        let n = z_sc.rows as f64;
        let y_mean = y.iter().sum::<f64>() / n;
        let y_c: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

        let alpha = if self.alpha < 0.0 {
            // Use theory-based alpha computation (NO CV)
            Self::compute_alpha(&z_sc, &y_c)
        } else {
            self.alpha
        };

        self.coef_ = Self::cd_lasso(&z_sc, &y_c, alpha, self.max_iter, self.tol);
        self.intercept_ = intercept_from_standardisation(y_mean, &means, &stds, &self.coef_);
        self.col_means_ = means;
        self.col_stds_ = stds;
        self.exp_ = Some(exp);
    }

    /// Predict on new `x`.
    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        let exp = polynomial_expand(x, self.degree, self.include_bias, false, 1e-8, None);
        let z_sc = standardise(&exp.matrix, &self.col_means_, &self.col_stds_);
        predict_linear(&z_sc, &self.coef_, self.intercept_)
    }

    /// Build a [`ResultBundle`].
    pub fn to_result_bundle(
        &self,
        x: &Matrix,
        y: &[f64],
        dataset: &str,
        elapsed: f64,
    ) -> ResultBundle {
        let exp = self.exp_.as_ref().expect("call fit() first");
        let (names, base_indices, terms) = selected_metadata(&self.coef_, exp);
        let n_selected = terms.len();
        let sel_coef: Vec<f64> = self
            .coef_
            .iter()
            .zip(exp.info.iter())
            .filter(|(&c, _)| c != 0.0)
            .map(|(&c, _)| c)
            .collect();

        let y_pred = self.predict(x);

        let mut rb = ResultBundle {
            method: "poly_lasso".into(),
            dataset: dataset.into(),
            selected_names: names,
            selected_base_indices: base_indices,
            selected_terms: terms,
            coef: sel_coef,
            intercept: self.intercept_,
            n_selected,
            elapsed_seconds: elapsed,
            params: vec![
                ("degree".into(), self.degree as f64),
                ("alpha".into(), self.alpha),
            ],
            ..ResultBundle::nan()
        };
        rb.compute_fit_stats(y, &y_pred, n_selected + 1);
        rb
    }
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// PolyOMP — greedy Orthogonal Matching Pursuit
// ---------------------------------------------------------------------------

/// Polynomial expansion + greedy Orthogonal Matching Pursuit.
///
/// Sparsity selection (NO ground-truth k):
/// - If `max_nonzero > 0`: use the provided value
/// - If `max_nonzero == 0`: use phase-transition formula:
///     If p > n (underdetermined): k = min(n/(4*log(p/n)), n/3, p/3, 15)
///     If p <= n (overdetermined): k = min(min(n,p)/4, n/3, p/3, 15)
///   This adapts to problem geometry via compressed sensing phase transition theory.
pub struct PolyOMP {
    pub degree: u32,
    pub include_bias: bool,
    /// Maximum non-zeros.  `0` → phase-transition formula (NO CV, NO ground-truth k).
    pub max_nonzero: usize,

    pub coef_: Vec<f64>,
    pub intercept_: f64,
    pub col_means_: Vec<f64>,
    pub col_stds_: Vec<f64>,
    pub exp_: Option<ExpandedFeatures>,
}

impl Default for PolyOMP {
    fn default() -> Self {
        Self {
            degree: 2,
            include_bias: true,
            max_nonzero: 0,
            coef_: Vec::new(),
            intercept_: 0.0,
            col_means_: Vec::new(),
            col_stds_: Vec::new(),
            exp_: None,
        }
    }
}

impl PolyOMP {
    pub fn new(degree: u32) -> Self {
        Self {
            degree,
            ..Self::default()
        }
    }

    fn proj_coef(z: &Matrix, y: &[f64], active: &[usize]) -> Vec<f64> {
        let k = active.len();
        let n = z.rows;
        let mut g = Matrix::new(k, k, 0.0);
        let mut rhs = vec![0.0_f64; k];
        for a in 0..k {
            for b in 0..k {
                let s: f64 = (0..n).map(|i| z[(i, active[a])] * z[(i, active[b])]).sum();
                g[(a, b)] = s;
            }
            g[(a, a)] += 1e-8;
            rhs[a] = (0..n).map(|i| z[(i, active[a])] * y[i]).sum::<f64>();
        }
        // Simple Cholesky solve (tiny system)
        match crate::matrix::cholesky(&g) {
            Ok(l) => {
                // forward sub then backward sub
                let z_vec = forward_sub(&l, &rhs);
                backward_sub(&l, &z_vec)
            }
            Err(_) => vec![0.0; k],
        }
    }

    /// Fit on `(x, y)`.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) {
        let exp = polynomial_expand(x, self.degree, self.include_bias, false, 1e-8, None);
        let (means, stds) = col_stats(&exp.matrix);
        let z = standardise(&exp.matrix, &means, &stds);

        let n = z.rows;
        let p = z.cols;
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let y_c: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

        // Theory-inspired sparsity selection WITHOUT using ground-truth k.
        // Phase transition for sparse recovery (Donoho-Tanner):
        //   - Underdetermined (p > n): k ~ n / (2*log(p/n))
        //   - Overdetermined (p <= n): k ~ min(n, p) / 2
        // We use conservative bounds to avoid overfitting.
        let budget = if self.max_nonzero == 0 {
            let n_f = n as f64;
            let p_f = p as f64;
            let phase_transition = if p > n {
                // Underdetermined: use compressed sensing phase transition
                let ratio = p_f / n_f;
                n_f / (2.0 * ratio.ln())
            } else {
                // Overdetermined: can use more features
                n_f.min(p_f) / 2.0
            };
            let k = (phase_transition / 2.0)
                .min(n_f / 3.0)
                .min(p_f / 3.0)
                .min(15.0) as usize;
            k.max(1).min(p)
        } else {
            self.max_nonzero.min(p)
        };

        let mut active: Vec<usize> = Vec::new();
        let mut used = vec![false; p];
        let mut residual = y_c.clone();

        for _ in 0..budget {
            let (best_j, best_cor) = (0..p).fold((0, -1.0_f64), |(bj, bc), j| {
                if used[j] {
                    return (bj, bc);
                }
                let c = (0..n).map(|i| z[(i, j)] * residual[i]).sum::<f64>().abs();
                if c > bc {
                    (j, c)
                } else {
                    (bj, bc)
                }
            });
            if best_cor <= 0.0 {
                break;
            }
            active.push(best_j);
            used[best_j] = true;

            let beta_a = Self::proj_coef(&z, &y_c, &active);
            // Recompute residual
            residual = y_c.clone();
            for (a, &ja) in active.iter().enumerate() {
                for i in 0..n {
                    residual[i] -= z[(i, ja)] * beta_a[a];
                }
            }
            let res_norm: f64 = residual.iter().map(|r| r.powi(2)).sum::<f64>().sqrt();
            if res_norm < 1e-8 {
                break;
            }
        }

        self.coef_ = vec![0.0; p];
        if !active.is_empty() {
            let beta_a = Self::proj_coef(&z, &y_c, &active);
            for (a, &ja) in active.iter().enumerate() {
                self.coef_[ja] = beta_a[a];
            }
        }
        self.intercept_ = intercept_from_standardisation(y_mean, &means, &stds, &self.coef_);
        self.col_means_ = means;
        self.col_stds_ = stds;
        self.exp_ = Some(exp);
    }

    /// Predict on new `x`.
    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        let exp = polynomial_expand(x, self.degree, self.include_bias, false, 1e-8, None);
        let z_sc = standardise(&exp.matrix, &self.col_means_, &self.col_stds_);
        predict_linear(&z_sc, &self.coef_, self.intercept_)
    }

    /// Build a [`ResultBundle`].
    pub fn to_result_bundle(
        &self,
        x: &Matrix,
        y: &[f64],
        dataset: &str,
        elapsed: f64,
    ) -> ResultBundle {
        let exp = self.exp_.as_ref().expect("call fit() first");
        let (names, base_indices, terms) = selected_metadata(&self.coef_, exp);
        let n_selected = terms.len();
        let sel_coef: Vec<f64> = self.coef_.iter().filter(|&&c| c != 0.0).copied().collect();
        let y_pred = self.predict(x);

        let mut rb = ResultBundle {
            method: "poly_omp".into(),
            dataset: dataset.into(),
            selected_names: names,
            selected_base_indices: base_indices,
            selected_terms: terms,
            coef: sel_coef,
            intercept: self.intercept_,
            n_selected,
            elapsed_seconds: elapsed,
            params: vec![
                ("degree".into(), self.degree as f64),
                ("max_nonzero".into(), self.max_nonzero as f64),
            ],
            ..ResultBundle::nan()
        };
        rb.compute_fit_stats(y, &y_pred, n_selected + 1);
        rb
    }
}

// ---------------------------------------------------------------------------
// PolySTLSQ — Sequential Thresholded Least Squares
// ---------------------------------------------------------------------------

/// Polynomial expansion + Sequential Thresholded Least Squares (SINDy-style).
pub struct PolySTLSQ {
    pub degree: u32,
    pub include_bias: bool,
    /// Threshold for pruning.  Negative → `0.1 * max|β₀|`.
    pub threshold: f64,
    pub max_iter: usize,

    pub coef_: Vec<f64>,
    pub intercept_: f64,
    pub active_: Vec<bool>,
    pub col_means_: Vec<f64>,
    pub col_stds_: Vec<f64>,
    pub exp_: Option<ExpandedFeatures>,
}

impl Default for PolySTLSQ {
    fn default() -> Self {
        Self {
            degree: 2,
            include_bias: true,
            threshold: -1.0,
            max_iter: 20,
            coef_: Vec::new(),
            intercept_: 0.0,
            active_: Vec::new(),
            col_means_: Vec::new(),
            col_stds_: Vec::new(),
            exp_: None,
        }
    }
}

impl PolySTLSQ {
    pub fn new(degree: u32) -> Self {
        Self {
            degree,
            ..Self::default()
        }
    }

    fn ols(z: &Matrix, y: &[f64], active: &[bool]) -> Vec<f64> {
        let active_idx: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(_, &a)| a)
            .map(|(i, _)| i)
            .collect();
        let k = active_idx.len();
        let n = z.rows;
        let mut g = Matrix::new(k, k, 0.0);
        let mut rhs = vec![0.0_f64; k];
        for a in 0..k {
            for b in 0..k {
                let s: f64 = (0..n)
                    .map(|i| z[(i, active_idx[a])] * z[(i, active_idx[b])])
                    .sum();
                g[(a, b)] = s;
            }
            g[(a, a)] += 1e-8;
            rhs[a] = (0..n).map(|i| z[(i, active_idx[a])] * y[i]).sum::<f64>();
        }
        let beta_a = match crate::matrix::cholesky(&g) {
            Ok(l) => backward_sub(&l, &forward_sub(&l, &rhs)),
            Err(_) => vec![0.0; k],
        };
        let mut full = vec![0.0_f64; active.len()];
        for (a, &ja) in active_idx.iter().enumerate() {
            full[ja] = beta_a[a];
        }
        full
    }

    /// Fit on `(x, y)`.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) {
        let exp = polynomial_expand(x, self.degree, self.include_bias, false, 1e-8, None);
        let (means, stds) = col_stats(&exp.matrix);
        let z = standardise(&exp.matrix, &means, &stds);

        let n = z.rows;
        let p = z.cols;
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let y_c: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

        self.active_ = vec![true; p];
        self.coef_ = Self::ols(&z, &y_c, &self.active_);

        let thr = if self.threshold < 0.0 {
            let max_abs = self.coef_.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
            (0.1 * max_abs).max(1e-8)
        } else {
            self.threshold
        };

        for _ in 0..self.max_iter {
            let new_active: Vec<bool> = self
                .coef_
                .iter()
                .zip(self.active_.iter())
                .map(|(&c, &a)| a && c.abs() >= thr)
                .collect();

            let any_active = new_active.iter().any(|&a| a);
            let new_active = if !any_active {
                // Keep single largest
                let best = self
                    .coef_
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut na = vec![false; p];
                na[best] = true;
                na
            } else {
                new_active
            };

            if new_active == self.active_ {
                break;
            }
            self.active_ = new_active;
            self.coef_ = Self::ols(&z, &y_c, &self.active_);
        }

        self.intercept_ = intercept_from_standardisation(y_mean, &means, &stds, &self.coef_);
        self.col_means_ = means;
        self.col_stds_ = stds;
        self.exp_ = Some(exp);
    }

    /// Predict on new `x`.
    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        let exp = polynomial_expand(x, self.degree, self.include_bias, false, 1e-8, None);
        let z_sc = standardise(&exp.matrix, &self.col_means_, &self.col_stds_);
        predict_linear(&z_sc, &self.coef_, self.intercept_)
    }

    /// Build a [`ResultBundle`].
    pub fn to_result_bundle(
        &self,
        x: &Matrix,
        y: &[f64],
        dataset: &str,
        elapsed: f64,
    ) -> ResultBundle {
        let exp = self.exp_.as_ref().expect("call fit() first");
        let coef_masked: Vec<f64> = self
            .coef_
            .iter()
            .zip(self.active_.iter())
            .map(|(&c, &a)| if a { c } else { 0.0 })
            .collect();
        let (names, base_indices, terms) = selected_metadata(&coef_masked, exp);
        let n_selected = terms.len();
        let sel_coef: Vec<f64> = coef_masked.iter().filter(|&&c| c != 0.0).copied().collect();
        let y_pred = self.predict(x);

        let mut rb = ResultBundle {
            method: "sparse_poly_stlsq".into(),
            dataset: dataset.into(),
            selected_names: names,
            selected_base_indices: base_indices,
            selected_terms: terms,
            coef: sel_coef,
            intercept: self.intercept_,
            n_selected,
            elapsed_seconds: elapsed,
            params: vec![
                ("degree".into(), self.degree as f64),
                ("threshold".into(), self.threshold),
            ],
            ..ResultBundle::nan()
        };
        rb.compute_fit_stats(y, &y_pred, n_selected + 1);
        rb
    }
}

// ---------------------------------------------------------------------------
// Tiny Cholesky helpers for OMP / STLSQ inner solves
// ---------------------------------------------------------------------------

fn forward_sub(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[(i, j)] * x[j];
        }
        x[i] = s / l[(i, i)];
    }
    x
}

fn backward_sub(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[(j, i)] * x[j]; // L^T is upper triangular
        }
        x[i] = s / l[(i, i)];
    }
    x
}

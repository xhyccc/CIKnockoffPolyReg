//! Rational polynomial dictionary expansion Φ(X) with feature interactions.
//!
//! For each base feature `x_j`, the dictionary produces:
//!   - Positive powers: x_j^1, ..., x_j^degree
//!   - Negative powers: x_j^{-1}, ..., x_j^{-degree}
//!   - Interaction terms: x_i^a * x_j^b where |a|+|b| <= degree
//!   - Optional bias column (constant 1)

use crate::matrix::Matrix;

/// Metadata for a single expanded feature column.
#[derive(Debug, Clone)]
pub struct ExpandedFeatureInfo {
    /// Original base feature index (-1 for bias, -2 for interaction).
    pub base_feature_index: i64,
    /// Signed exponent (0 for bias, total degree for interactions).
    pub exponent: i32,
    /// Human-readable name (e.g. `"x2^(-1)"`, `"x1*x2"`).
    pub name: String,
    /// For interaction terms: indices of involved base features.
    pub interaction_indices: Option<Vec<usize>>,
    /// For interaction terms: individual exponents for each involved base feature.
    pub interaction_exponents: Option<Vec<i32>>,
}

/// Result of polynomial dictionary expansion.
#[derive(Debug, Clone)]
pub struct ExpandedFeatures {
    /// Expanded data matrix (n_samples × n_expanded_features).
    pub matrix: Matrix,
    /// Metadata for each column.
    pub info: Vec<ExpandedFeatureInfo>,
}

/// Number of columns produced by [`polynomial_expand`] for `n_base` features with interactions.
pub fn n_expanded_features(
    n_base: usize,
    degree: u32,
    include_bias: bool,
    include_interactions: bool,
) -> usize {
    let monomials = n_base * 2 * degree as usize;
    let interactions = if include_interactions && n_base >= 2 {
        // For each pair, count valid (a,b) where |a|+|b| <= degree, a,b != 0
        // Number of valid combinations per pair: 4 * sum_{k=2}^{degree} (k-1) = 2*degree*(degree-1)
        let n_pairs = n_base * (n_base - 1) / 2;
        let valid_per_pair: usize = (2..=degree).map(|k| 4_usize * (k as usize - 1)).sum();
        n_pairs * valid_per_pair
    } else {
        0
    };
    monomials + interactions + if include_bias { 1 } else { 0 }
}

/// Expand base feature matrix `X` (n × p) via rational polynomial dictionary Φ with interactions.
///
/// # Parameters
/// - `x`:                  Input matrix (n_samples × n_base_features).
/// - `degree`:             Maximum absolute exponent (≥ 1).
/// - `include_bias`:       Append a constant-1 column when `true`.
/// - `include_interactions`: Include interaction terms when `true`.
/// - `clip_threshold`:     Values with `|x| < clip_threshold` are clamped to
///                         `±clip_threshold` before computing negative powers.
/// - `base_names`:         Optional names for each base feature.  Uses `"x0"`,
///                         `"x1"`, … when `None`.
///
/// # Panics
/// Panics if `degree == 0` or `base_names.len() != x.cols`.
pub fn polynomial_expand(
    x: &Matrix,
    degree: u32,
    include_bias: bool,
    include_interactions: bool,
    clip_threshold: f64,
    base_names: Option<&[String]>,
) -> ExpandedFeatures {
    assert!(degree >= 1, "polynomial_expand: degree must be >= 1");
    let n = x.rows;
    let p = x.cols;

    let default_names: Vec<String>;
    let names: &[String] = match base_names {
        Some(ns) => {
            assert_eq!(
                ns.len(),
                p,
                "polynomial_expand: base_names.len() must equal x.cols"
            );
            ns
        }
        None => {
            default_names = (0..p).map(|j| format!("x{j}")).collect();
            &default_names
        }
    };

    let n_out = n_expanded_features(p, degree, include_bias, include_interactions);
    let mut out = Matrix::new(n, n_out, 0.0);
    let mut info: Vec<ExpandedFeatureInfo> = Vec::with_capacity(n_out);

    let mut col = 0usize;

    // Helper to safely clip values
    let safe_value = |v: f64| -> f64 {
        if v.abs() < clip_threshold {
            if v >= 0.0 {
                clip_threshold
            } else {
                -clip_threshold
            }
        } else {
            v
        }
    };

    // 1. Individual monomials
    for j in 0..p {
        // Positive powers: d = 1 to degree
        for d in 1..=degree {
            for i in 0..n {
                out[(i, col)] = x[(i, j)].powi(d as i32);
            }
            let feat_name = if d == 1 {
                names[j].clone()
            } else {
                format!("{}^{}", names[j], d)
            };
            info.push(ExpandedFeatureInfo {
                base_feature_index: j as i64,
                exponent: d as i32,
                name: feat_name,
                interaction_indices: None,
                interaction_exponents: None,
            });
            col += 1;
        }
        // Negative powers: d = 1 to degree
        for d in 1..=degree {
            for i in 0..n {
                let xij = safe_value(x[(i, j)]);
                out[(i, col)] = xij.powi(-(d as i32));
            }
            let feat_name = format!("{}^(-{})", names[j], d);
            info.push(ExpandedFeatureInfo {
                base_feature_index: j as i64,
                exponent: -(d as i32),
                name: feat_name,
                interaction_indices: None,
                interaction_exponents: None,
            });
            col += 1;
        }
    }

    // 2. Interaction terms: x_i^a * x_j^b where |a|+|b| <= degree, a,b != 0
    if include_interactions && p >= 2 {
        for i in 0..p {
            for j in (i + 1)..p {
                // Generate all combinations of exponents
                for exp_i in -(degree as i32)..=(degree as i32) {
                    if exp_i == 0 {
                        continue;
                    }
                    for exp_j in -(degree as i32)..=(degree as i32) {
                        if exp_j == 0 {
                            continue;
                        }
                        if exp_i.abs() + exp_j.abs() <= degree as i32 {
                            // Compute x_i^exp_i * x_j^exp_j
                            for row in 0..n {
                                let xi = safe_value(x[(row, i)]);
                                let xj = safe_value(x[(row, j)]);
                                out[(row, col)] = xi.powi(exp_i) * xj.powi(exp_j);
                            }

                            // Format name
                            let term_i = if exp_i == 1 {
                                names[i].clone()
                            } else if exp_i < 0 {
                                format!("{}^({})", names[i], exp_i)
                            } else {
                                format!("{}^{}", names[i], exp_i)
                            };
                            let term_j = if exp_j == 1 {
                                names[j].clone()
                            } else if exp_j < 0 {
                                format!("{}^({})", names[j], exp_j)
                            } else {
                                format!("{}^{}", names[j], exp_j)
                            };

                            info.push(ExpandedFeatureInfo {
                                base_feature_index: -2,              // -2 indicates interaction
                                exponent: exp_i.abs() + exp_j.abs(), // total degree
                                name: format!("{}*{}", term_i, term_j),
                                interaction_indices: Some(vec![i, j]),
                                interaction_exponents: Some(vec![exp_i, exp_j]),
                            });
                            col += 1;
                        }
                    }
                }
            }
        }
    }

    // 3. Bias column
    if include_bias {
        for i in 0..n {
            out[(i, col)] = 1.0;
        }
        info.push(ExpandedFeatureInfo {
            base_feature_index: -1,
            exponent: 0,
            name: "1".to_string(),
            interaction_indices: None,
            interaction_exponents: None,
        });
    }

    ExpandedFeatures { matrix: out, info }
}

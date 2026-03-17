//! Rational polynomial dictionary expansion Φ(X).
//!
//! For each base feature `x_j`, the dictionary produces:
//!   - Positive powers: x_j^1, ..., x_j^degree
//!   - Negative powers: x_j^{-1}, ..., x_j^{-degree}
//!   - Optional bias column (constant 1)
//!
//! This implements the dictionary Φ(·) = (·, 1/·, 1)^d from the paper.

use crate::matrix::Matrix;

/// Metadata for a single expanded feature column.
#[derive(Debug, Clone)]
pub struct ExpandedFeatureInfo {
    /// Original base feature index (-1 for bias column).
    pub base_feature_index: i64,
    /// Signed exponent (0 for bias).
    pub exponent: i32,
    /// Human-readable name (e.g. `"x2^(-1)"`).
    pub name: String,
}

/// Result of polynomial dictionary expansion.
#[derive(Debug, Clone)]
pub struct ExpandedFeatures {
    /// Expanded data matrix (n_samples × n_expanded_features).
    pub matrix: Matrix,
    /// Metadata for each column.
    pub info: Vec<ExpandedFeatureInfo>,
}

/// Number of columns produced by [`polynomial_expand`] for `n_base` features.
pub fn n_expanded_features(n_base: usize, degree: u32, include_bias: bool) -> usize {
    n_base * 2 * degree as usize + if include_bias { 1 } else { 0 }
}

/// Expand base feature matrix `X` (n × p) via rational polynomial dictionary Φ.
///
/// # Parameters
/// - `x`:              Input matrix (n_samples × n_base_features).
/// - `degree`:         Maximum absolute exponent (≥ 1).
/// - `include_bias`:   Append a constant-1 column when `true`.
/// - `clip_threshold`: Values with `|x| < clip_threshold` are clamped to
///                     `±clip_threshold` before computing negative powers.
/// - `base_names`:     Optional names for each base feature.  Uses `"x0"`,
///                     `"x1"`, … when `None`.
///
/// # Panics
/// Panics if `degree == 0` or `base_names.len() != x.cols`.
pub fn polynomial_expand(
    x: &Matrix,
    degree: u32,
    include_bias: bool,
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

    let n_out = n_expanded_features(p, degree, include_bias);
    let mut out = Matrix::new(n, n_out, 0.0);
    let mut info: Vec<ExpandedFeatureInfo> = Vec::with_capacity(n_out);

    let mut col = 0usize;
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
            });
            col += 1;
        }
        // Negative powers: d = 1 to degree
        for d in 1..=degree {
            for i in 0..n {
                let mut xij = x[(i, j)];
                if xij.abs() < clip_threshold {
                    xij = if xij >= 0.0 { clip_threshold } else { -clip_threshold };
                }
                out[(i, col)] = xij.powi(-(d as i32));
            }
            let feat_name = format!("{}^(-{})", names[j], d);
            info.push(ExpandedFeatureInfo {
                base_feature_index: j as i64,
                exponent: -(d as i32),
                name: feat_name,
            });
            col += 1;
        }
    }

    // Bias column
    if include_bias {
        for i in 0..n {
            out[(i, col)] = 1.0;
        }
        info.push(ExpandedFeatureInfo {
            base_feature_index: -1,
            exponent: 0,
            name: "1".to_string(),
        });
    }

    ExpandedFeatures { matrix: out, info }
}

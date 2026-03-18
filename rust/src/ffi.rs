//! C-ABI bindings for Python ctypes integration.
//!
//! Exposes the core IC-Knock-Poly Rust kernels through a stable C interface
//! so that Python code can load the compiled shared library with `ctypes`.
//!
//! # Building
//!
//! ```bash
//! cd rust/
//! cargo build --release
//! # Produces target/release/libic_knockoff_poly_reg.so  (Linux/macOS)
//! #         or  target/release/ic_knockoff_poly_reg.dll (Windows)
//! ```
//!
//! # Python usage
//!
//! ```python
//! from ic_knockoff_poly_reg.kernels import create_kernels
//! poly, knock, posi = create_kernels("rust")
//! ```

use std::collections::HashSet;
use std::slice;

use crate::knockoffs::{compute_w_statistics, equicorrelated_s_values, sample_gaussian_knockoffs};
use crate::matrix::Matrix;
use crate::polynomial::{n_expanded_features, polynomial_expand};
use crate::posi::{alpha_spending_budget, knockoff_threshold, SpendingSequence};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Decode a sequence-type code into the [`SpendingSequence`] enum.
/// 0 = RiemannZeta (default), 1 = Geometric.
fn seq_from_code(code: i32) -> SpendingSequence {
    if code == 1 {
        SpendingSequence::Geometric
    } else {
        SpendingSequence::RiemannZeta
    }
}

/// Build a [`Matrix`] from a raw row-major C pointer.
///
/// # Safety
/// `ptr` must point to at least `rows * cols` valid `f64` values.
unsafe fn matrix_from_raw(ptr: *const f64, rows: usize, cols: usize) -> Matrix {
    let s = slice::from_raw_parts(ptr, rows * cols);
    let mut m = Matrix::new(rows, cols, 0.0);
    m.data.copy_from_slice(s);
    m
}

// ---------------------------------------------------------------------------
// Polynomial expansion
// ---------------------------------------------------------------------------

/// Return the number of expanded columns for `n_base` base features.
///
/// # Arguments
/// - `n_base`       – Number of base features.
/// - `degree`       – Maximum absolute exponent.
/// - `include_bias` – 1 to include a bias column, 0 otherwise.
#[no_mangle]
pub extern "C" fn ic_poly_n_expanded(n_base: i32, degree: i32, include_bias: i32) -> i32 {
    n_expanded_features(n_base as usize, degree as u32, include_bias != 0) as i32
}

/// Expand `X` (row-major, n×p) via rational polynomial dictionary Φ.
///
/// Fills `out_matrix` (n × n_cols, row-major), `out_base_indices` and
/// `out_exponents` (both of length n_cols).  Returns n_cols on success or
/// -1 on error.
///
/// Callers must pre-allocate:
/// - `out_matrix`:       `n * ic_poly_n_expanded(p, degree, include_bias)` doubles
/// - `out_base_indices`: `ic_poly_n_expanded(p, degree, include_bias)` ints
/// - `out_exponents`:    `ic_poly_n_expanded(p, degree, include_bias)` ints
///
/// # Safety
/// All pointer arguments must be non-null and point to sufficient memory.
#[no_mangle]
pub unsafe extern "C" fn ic_poly_expand(
    x_flat: *const f64,
    n: i32,
    p: i32,
    degree: i32,
    include_bias: i32,
    clip_threshold: f64,
    out_matrix: *mut f64,
    out_base_indices: *mut i32,
    out_exponents: *mut i32,
) -> i32 {
    if x_flat.is_null() || out_matrix.is_null() || out_base_indices.is_null() || out_exponents.is_null() {
        return -1;
    }
    let n = n as usize;
    let p = p as usize;
    let degree = degree as u32;

    let x = matrix_from_raw(x_flat, n, p);
    let result = polynomial_expand(&x, degree, include_bias != 0, clip_threshold, None);
    let n_cols = result.info.len();

    // Copy expanded matrix (row-major)
    let out_mat = slice::from_raw_parts_mut(out_matrix, n * n_cols);
    for i in 0..n {
        for j in 0..n_cols {
            out_mat[i * n_cols + j] = result.matrix[(i, j)];
        }
    }

    // Copy column metadata
    let out_bi = slice::from_raw_parts_mut(out_base_indices, n_cols);
    let out_ex = slice::from_raw_parts_mut(out_exponents, n_cols);
    for j in 0..n_cols {
        out_bi[j] = result.info[j].base_feature_index as i32;
        out_ex[j] = result.info[j].exponent;
    }

    n_cols as i32
}

// ---------------------------------------------------------------------------
// Knockoff W-statistics
// ---------------------------------------------------------------------------

/// Compute W_j = |beta_orig_j| − |beta_knock_j| for all j.
///
/// # Safety
/// `beta_orig`, `beta_knock`, and `out_w` must each point to `p` valid doubles.
#[no_mangle]
pub unsafe extern "C" fn ic_compute_w_statistics(
    beta_orig: *const f64,
    beta_knock: *const f64,
    p: i32,
    out_w: *mut f64,
) {
    let p = p as usize;
    let orig = slice::from_raw_parts(beta_orig, p);
    let knock = slice::from_raw_parts(beta_knock, p);
    let out = slice::from_raw_parts_mut(out_w, p);
    let w = compute_w_statistics(orig, knock);
    out.copy_from_slice(&w);
}

// ---------------------------------------------------------------------------
// Equicorrelated s-values
// ---------------------------------------------------------------------------

/// Compute equicorrelated s-values for covariance matrix Sigma (p×p, row-major).
///
/// # Safety
/// `cov_flat` must point to `p*p` valid doubles; `out_s` must point to `p` doubles.
#[no_mangle]
pub unsafe extern "C" fn ic_equicorrelated_s_values(
    cov_flat: *const f64,
    p: i32,
    reg: f64,
    out_s: *mut f64,
) {
    let p = p as usize;
    let sigma = matrix_from_raw(cov_flat, p, p);
    let s = equicorrelated_s_values(&sigma, reg);
    let out = slice::from_raw_parts_mut(out_s, p);
    out.copy_from_slice(&s);
}

// ---------------------------------------------------------------------------
// Gaussian knockoff sampling
// ---------------------------------------------------------------------------

/// Sample equicorrelated Gaussian knockoffs.
///
/// Returns 0 on success, -1 on failure (e.g. non-positive-definite covariance).
///
/// # Safety
/// All pointer arguments must be non-null and point to sufficient memory:
/// - `x_flat`:    `n*p` doubles
/// - `mu_flat`:   `p` doubles
/// - `sigma_flat`: `p*p` doubles
/// - `out_x_tilde`: `n*p` doubles (caller-allocated)
#[no_mangle]
pub unsafe extern "C" fn ic_sample_gaussian_knockoffs(
    x_flat: *const f64,
    n: i32,
    p: i32,
    mu_flat: *const f64,
    sigma_flat: *const f64,
    seed: u64,
    out_x_tilde: *mut f64,
) -> i32 {
    let n = n as usize;
    let p = p as usize;

    let x = matrix_from_raw(x_flat, n, p);
    let mu = slice::from_raw_parts(mu_flat, p);
    let sigma = matrix_from_raw(sigma_flat, p, p);
    let out = slice::from_raw_parts_mut(out_x_tilde, n * p);

    match sample_gaussian_knockoffs(&x, mu, &sigma, seed) {
        Ok(x_tilde) => {
            out.copy_from_slice(&x_tilde.data);
            0
        }
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------------
// Alpha-spending budget
// ---------------------------------------------------------------------------

/// Return the alpha-spending budget q_t for iteration `t`.
///
/// `sequence_type`: 0 = Riemann Zeta (default), 1 = Geometric.
#[no_mangle]
pub extern "C" fn ic_alpha_spending_budget(
    t: i32,
    q: f64,
    sequence_type: i32,
    gamma: f64,
) -> f64 {
    let seq = seq_from_code(sequence_type);
    alpha_spending_budget(t as usize, q, seq, gamma)
}

// ---------------------------------------------------------------------------
// Knockoff+ threshold
// ---------------------------------------------------------------------------

/// Compute the knockoff+ threshold τ_t.
///
/// Returns +∞ if no threshold satisfies the FDR condition.
///
/// `active_indices` may be NULL when `n_active == 0`.
///
/// # Safety
/// `w_ptr` must point to `p` valid doubles.
/// If `n_active > 0`, `active_indices` must point to `n_active` valid ints.
#[no_mangle]
pub unsafe extern "C" fn ic_knockoff_threshold(
    w_ptr: *const f64,
    p: i32,
    active_indices: *const i32,
    n_active: i32,
    q_t: f64,
    offset: i32,
) -> f64 {
    let p = p as usize;
    let w = slice::from_raw_parts(w_ptr, p);

    let mut active: HashSet<usize> = HashSet::new();
    if n_active > 0 && !active_indices.is_null() {
        let act = slice::from_raw_parts(active_indices, n_active as usize);
        for &idx in act {
            active.insert(idx as usize);
        }
    }

    knockoff_threshold(w, q_t, &active, offset as i64)
}

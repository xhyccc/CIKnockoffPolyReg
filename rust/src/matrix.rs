//! Dense matrix operations for IC-Knock-Poly.
//!
//! Provides a simple row-major `Matrix` type with:
//! - Basic arithmetic (add, scale, multiply)
//! - Cholesky decomposition and SPD inversion
//! - Column mean and sample covariance
//! - Block extraction
//! - Gershgorin eigenvalue lower bound

/// Dense matrix in row-major layout.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a new matrix filled with `fill`.
    pub fn new(rows: usize, cols: usize, fill: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![fill; rows * cols],
        }
    }

    /// Create an identity matrix of size n×n.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n, 0.0);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut out = Self::new(self.cols, self.rows, 0.0);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

/// Element-wise addition: C = A + B.
pub fn mat_add(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.rows, b.rows, "mat_add: row mismatch");
    assert_eq!(a.cols, b.cols, "mat_add: col mismatch");
    let data = a.data.iter().zip(&b.data).map(|(&x, &y)| x + y).collect();
    Matrix {
        rows: a.rows,
        cols: a.cols,
        data,
    }
}

/// Scale: B = alpha * A.
pub fn mat_scale(a: &Matrix, alpha: f64) -> Matrix {
    let data = a.data.iter().map(|&x| alpha * x).collect();
    Matrix {
        rows: a.rows,
        cols: a.cols,
        data,
    }
}

/// Matrix multiplication: C = A * B.
pub fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "mat_mul: incompatible shapes");
    let mut c = Matrix::new(a.rows, b.cols, 0.0);
    for i in 0..a.rows {
        for k in 0..a.cols {
            let aik = a[(i, k)];
            for j in 0..b.cols {
                c[(i, j)] += aik * b[(k, j)];
            }
        }
    }
    c
}

/// Matrix-vector product: y = A * x.
pub fn mat_vec(a: &Matrix, x: &[f64]) -> Vec<f64> {
    assert_eq!(a.cols, x.len(), "mat_vec: incompatible shapes");
    let mut y = vec![0.0; a.rows];
    for i in 0..a.rows {
        for j in 0..a.cols {
            y[i] += a[(i, j)] * x[j];
        }
    }
    y
}

// ---------------------------------------------------------------------------
// Cholesky decomposition
// ---------------------------------------------------------------------------

/// Cholesky decomposition: returns lower-triangular L such that A = L * L^T.
///
/// # Errors
/// Returns `Err` if A is not positive-definite.
pub fn cholesky(a: &Matrix) -> Result<Matrix, String> {
    let n = a.rows;
    assert_eq!(a.cols, n, "cholesky: matrix must be square");
    let mut l = Matrix::new(n, n, 0.0);
    for j in 0..n {
        let mut sum_sq = 0.0_f64;
        for k in 0..j {
            sum_sq += l[(j, k)] * l[(j, k)];
        }
        let diag = a[(j, j)] - sum_sq;
        if diag <= 0.0 {
            return Err(format!(
                "cholesky: matrix not positive-definite (diag={diag:.6e} at j={j})"
            ));
        }
        l[(j, j)] = diag.sqrt();
        for i in (j + 1)..n {
            let mut s = 0.0_f64;
            for k in 0..j {
                s += l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = (a[(i, j)] - s) / l[(j, j)];
        }
    }
    Ok(l)
}

/// Forward substitution: solve L x = b (L lower triangular).
pub fn forward_substitution(l: &Matrix, b: &[f64]) -> Vec<f64> {
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

/// Back substitution: solve L^T x = b.
pub fn backward_substitution(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = vec![0.0_f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[(j, i)] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = s / l[(i, i)];
    }
    x
}

/// Solve A x = b for symmetric positive-definite A via Cholesky.
pub fn solve_cholesky(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, String> {
    let l = cholesky(a)?;
    let y = forward_substitution(&l, b);
    Ok(backward_substitution(&l, &y))
}

/// Invert a symmetric positive-definite matrix via Cholesky.
pub fn mat_inv_spd(a: &Matrix) -> Result<Matrix, String> {
    let n = a.rows;
    let l = cholesky(a)?;
    let mut inv = Matrix::new(n, n, 0.0);
    let mut e = vec![0.0_f64; n];
    for j in 0..n {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[j] = 1.0;
        let y = forward_substitution(&l, &e);
        let x = backward_substitution(&l, &y);
        for i in 0..n {
            inv[(i, j)] = x[i];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Block extraction
// ---------------------------------------------------------------------------

/// Extract a sub-matrix using row and column index slices.
pub fn extract_block(a: &Matrix, row_idx: &[usize], col_idx: &[usize]) -> Matrix {
    let mut out = Matrix::new(row_idx.len(), col_idx.len(), 0.0);
    for (i, &ri) in row_idx.iter().enumerate() {
        for (j, &ci) in col_idx.iter().enumerate() {
            out[(i, j)] = a[(ri, ci)];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Column-wise mean of X (n_samples × n_features).
pub fn col_mean(x: &Matrix) -> Vec<f64> {
    let n = x.rows as f64;
    let mut mu = vec![0.0_f64; x.cols];
    for i in 0..x.rows {
        for j in 0..x.cols {
            mu[j] += x[(i, j)];
        }
    }
    mu.iter_mut().for_each(|v| *v /= n);
    mu
}

/// Sample covariance matrix (n_features × n_features).
///
/// # Panics
/// Panics if `x.rows < 2`.
pub fn sample_cov(x: &Matrix) -> Matrix {
    assert!(x.rows >= 2, "sample_cov: need at least 2 samples");
    let n = x.rows;
    let p = x.cols;
    let mu = col_mean(x);
    let mut cov = Matrix::new(p, p, 0.0);
    let inv_nm1 = 1.0 / (n as f64 - 1.0);
    for i in 0..n {
        for j in 0..p {
            for k in 0..p {
                cov[(j, k)] += (x[(i, j)] - mu[j]) * (x[(i, k)] - mu[k]) * inv_nm1;
            }
        }
    }
    cov
}

/// Gershgorin lower bound on the smallest eigenvalue of A.
pub fn gershgorin_lower_bound(a: &Matrix) -> f64 {
    let n = a.rows;
    let mut min_lb = f64::INFINITY;
    for i in 0..n {
        let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| a[(i, j)].abs()).sum();
        let lb = a[(i, i)] - row_sum;
        if lb < min_lb {
            min_lb = lb;
        }
    }
    min_lb
}

//! Integration tests for the IC-Knock-Poly Rust library.

use ic_knockoff_poly_reg::knockoffs::{compute_w_statistics, sample_gaussian_knockoffs};
use ic_knockoff_poly_reg::matrix::{
    cholesky, col_mean, extract_block, forward_substitution, backward_substitution,
    gershgorin_lower_bound, mat_add, mat_inv_spd, mat_mul, mat_scale, mat_vec,
    sample_cov, solve_cholesky, Matrix,
};
use ic_knockoff_poly_reg::polynomial::{n_expanded_features, polynomial_expand};
use ic_knockoff_poly_reg::posi::{
    alpha_spending_budget, alpha_spending_budgets, knockoff_threshold, SpendingSequence,
};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

#[test]
fn test_mat_add_basic() {
    let mut a = Matrix::new(2, 2, 0.0);
    let mut b = Matrix::new(2, 2, 0.0);
    a[(0, 0)] = 1.0; a[(0, 1)] = 2.0; a[(1, 0)] = 3.0; a[(1, 1)] = 4.0;
    b[(0, 0)] = 5.0; b[(0, 1)] = 6.0; b[(1, 0)] = 7.0; b[(1, 1)] = 8.0;
    let c = mat_add(&a, &b);
    assert!((c[(0, 0)] - 6.0).abs() < 1e-12);
    assert!((c[(1, 1)] - 12.0).abs() < 1e-12);
}

#[test]
fn test_mat_scale() {
    let mut a = Matrix::new(2, 2, 0.0);
    a[(0, 0)] = 2.0; a[(1, 1)] = 3.0;
    let b = mat_scale(&a, 2.0);
    assert!((b[(0, 0)] - 4.0).abs() < 1e-12);
    assert!((b[(1, 1)] - 6.0).abs() < 1e-12);
}

#[test]
fn test_mat_mul_basic() {
    let mut a = Matrix::new(2, 3, 0.0);
    a[(0, 0)] = 1.0; a[(0, 1)] = 2.0; a[(0, 2)] = 3.0;
    a[(1, 0)] = 4.0; a[(1, 1)] = 5.0; a[(1, 2)] = 6.0;
    let mut b = Matrix::new(3, 2, 0.0);
    b[(0, 0)] = 7.0;  b[(0, 1)] = 8.0;
    b[(1, 0)] = 9.0;  b[(1, 1)] = 10.0;
    b[(2, 0)] = 11.0; b[(2, 1)] = 12.0;
    let c = mat_mul(&a, &b);
    assert!((c[(0, 0)] - 58.0).abs() < 1e-12);
    assert!((c[(0, 1)] - 64.0).abs() < 1e-12);
    assert!((c[(1, 0)] - 139.0).abs() < 1e-12);
    assert!((c[(1, 1)] - 154.0).abs() < 1e-12);
}

#[test]
fn test_cholesky_factor() {
    // A = [[4,2],[2,3]]
    let mut a = Matrix::new(2, 2, 0.0);
    a[(0, 0)] = 4.0; a[(0, 1)] = 2.0;
    a[(1, 0)] = 2.0; a[(1, 1)] = 3.0;
    let l = cholesky(&a).expect("cholesky should succeed");
    let llt = mat_mul(&l, &l.transpose());
    assert!((llt[(0, 0)] - 4.0).abs() < 1e-12);
    assert!((llt[(0, 1)] - 2.0).abs() < 1e-12);
    assert!((llt[(1, 0)] - 2.0).abs() < 1e-12);
    assert!((llt[(1, 1)] - 3.0).abs() < 1e-12);
}

#[test]
fn test_cholesky_not_spd_fails() {
    let mut a = Matrix::new(2, 2, 0.0);
    a[(0, 0)] = -1.0; a[(1, 1)] = 1.0;
    assert!(cholesky(&a).is_err());
}

#[test]
fn test_solve_cholesky_recovers_rhs() {
    // A x = b  →  check A x̂ ≈ b
    let mut a = Matrix::new(3, 3, 0.0);
    a[(0, 0)] = 4.0; a[(0, 1)] = 2.0; a[(0, 2)] = 0.0;
    a[(1, 0)] = 2.0; a[(1, 1)] = 5.0; a[(1, 2)] = 1.0;
    a[(2, 0)] = 0.0; a[(2, 1)] = 1.0; a[(2, 2)] = 3.0;
    let b = vec![1.0, 2.0, 3.0];
    let x = solve_cholesky(&a, &b).expect("solve should succeed");
    let ax = mat_vec(&a, &x);
    for i in 0..3 {
        assert!((ax[i] - b[i]).abs() < 1e-10, "residual too large at {i}");
    }
}

#[test]
fn test_mat_inv_spd_product_is_identity() {
    let mut a = Matrix::new(2, 2, 0.0);
    a[(0, 0)] = 4.0; a[(0, 1)] = 2.0;
    a[(1, 0)] = 2.0; a[(1, 1)] = 3.0;
    let ainv = mat_inv_spd(&a).expect("inversion should succeed");
    let i_check = mat_mul(&ainv, &a);
    assert!((i_check[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((i_check[(0, 1)] - 0.0).abs() < 1e-10);
    assert!((i_check[(1, 0)] - 0.0).abs() < 1e-10);
    assert!((i_check[(1, 1)] - 1.0).abs() < 1e-10);
}

#[test]
fn test_col_mean() {
    let mut x = Matrix::new(3, 2, 0.0);
    x[(0, 0)] = 1.0; x[(0, 1)] = 4.0;
    x[(1, 0)] = 2.0; x[(1, 1)] = 5.0;
    x[(2, 0)] = 3.0; x[(2, 1)] = 6.0;
    let mu = col_mean(&x);
    assert!((mu[0] - 2.0).abs() < 1e-12);
    assert!((mu[1] - 5.0).abs() < 1e-12);
}

#[test]
fn test_sample_cov_shape_and_symmetry() {
    let mut x = Matrix::new(5, 3, 0.0);
    let vals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    for i in 0..5 {
        for j in 0..3 {
            x[(i, j)] = vals[i * 3 + j];
        }
    }
    let c = sample_cov(&x);
    assert_eq!(c.rows, 3);
    assert_eq!(c.cols, 3);
    for j in 0..3 {
        assert!(c[(j, j)] >= 0.0);
    }
    assert!((c[(0, 1)] - c[(1, 0)]).abs() < 1e-12);
    assert!((c[(0, 2)] - c[(2, 0)]).abs() < 1e-12);
}

#[test]
fn test_extract_block() {
    let mut a = Matrix::new(4, 4, 0.0);
    for i in 0..4 {
        for j in 0..4 {
            a[(i, j)] = (i * 4 + j) as f64;
        }
    }
    let b = extract_block(&a, &[1, 3], &[0, 2]);
    assert_eq!(b.rows, 2);
    assert_eq!(b.cols, 2);
    assert!((b[(0, 0)] - a[(1, 0)]).abs() < 1e-12);
    assert!((b[(0, 1)] - a[(1, 2)]).abs() < 1e-12);
    assert!((b[(1, 0)] - a[(3, 0)]).abs() < 1e-12);
    assert!((b[(1, 1)] - a[(3, 2)]).abs() < 1e-12);
}

#[test]
fn test_gershgorin_lower_bound() {
    // A = [[5, -1], [-1, 5]] → Gershgorin lb = 5 - 1 = 4 for each row
    let mut a = Matrix::new(2, 2, 0.0);
    a[(0, 0)] = 5.0; a[(0, 1)] = -1.0;
    a[(1, 0)] = -1.0; a[(1, 1)] = 5.0;
    let lb = gershgorin_lower_bound(&a);
    assert!((lb - 4.0).abs() < 1e-12);
}

#[test]
fn test_matrix_identity() {
    let i = Matrix::identity(3);
    for r in 0..3 {
        for c in 0..3 {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!((i[(r, c)] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn test_matrix_transpose() {
    let mut a = Matrix::new(2, 3, 0.0);
    a[(0, 0)] = 1.0; a[(0, 1)] = 2.0; a[(0, 2)] = 3.0;
    a[(1, 0)] = 4.0; a[(1, 1)] = 5.0; a[(1, 2)] = 6.0;
    let at = a.transpose();
    assert_eq!(at.rows, 3);
    assert_eq!(at.cols, 2);
    assert!((at[(2, 0)] - 3.0).abs() < 1e-12);
    assert!((at[(1, 1)] - 5.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Polynomial expansion
// ---------------------------------------------------------------------------

#[test]
fn test_poly_expansion_shape() {
    let x = Matrix::new(10, 3, 1.0);
    let result = polynomial_expand(&x, 2, true, 1e-8, None);
    let expected = n_expanded_features(3, 2, true);
    assert_eq!(result.matrix.cols, expected); // 3*2*2 + 1 = 13
    assert_eq!(result.matrix.rows, 10);
    assert_eq!(result.info.len(), expected);
}

#[test]
fn test_poly_positive_powers() {
    let mut x = Matrix::new(1, 1, 0.0);
    x[(0, 0)] = 3.0;
    let result = polynomial_expand(&x, 2, false, 1e-8, None);
    // Columns: x^1=3, x^2=9, x^(-1)=1/3, x^(-2)=1/9
    assert!((result.matrix[(0, 0)] - 3.0).abs() < 1e-12);
    assert!((result.matrix[(0, 1)] - 9.0).abs() < 1e-12);
    assert!((result.matrix[(0, 2)] - 1.0 / 3.0).abs() < 1e-12);
    assert!((result.matrix[(0, 3)] - 1.0 / 9.0).abs() < 1e-12);
}

#[test]
fn test_poly_bias_is_ones() {
    let x = Matrix::new(5, 2, 2.0);
    let result = polynomial_expand(&x, 1, true, 1e-8, None);
    let last_col = result.matrix.cols - 1;
    assert_eq!(result.info[last_col].name, "1");
    for i in 0..5 {
        assert!((result.matrix[(i, last_col)] - 1.0).abs() < 1e-12);
    }
}

#[test]
fn test_poly_near_zero_clipping_finite() {
    let mut x = Matrix::new(1, 1, 0.0);
    x[(0, 0)] = 0.0;
    let result = polynomial_expand(&x, 1, false, 1e-8, None);
    // Negative power must be finite (clipped, not infinite)
    assert!(result.matrix[(0, 1)].is_finite());
}

#[test]
#[should_panic]
fn test_poly_degree_zero_panics() {
    let x = Matrix::new(1, 1, 1.0);
    polynomial_expand(&x, 0, false, 1e-8, None);
}

#[test]
#[should_panic]
fn test_poly_name_mismatch_panics() {
    let x = Matrix::new(1, 2, 1.0);
    let names = vec!["only_one".to_string()];
    polynomial_expand(&x, 1, false, 1e-8, Some(&names));
}

#[test]
fn test_n_expanded_features() {
    assert_eq!(n_expanded_features(4, 2, true), 4 * 2 * 2 + 1);
    assert_eq!(n_expanded_features(3, 1, false), 3 * 2);
}

#[test]
fn test_poly_custom_names() {
    let x = Matrix::new(1, 2, 1.0);
    let names = vec!["foo".to_string(), "bar".to_string()];
    let result = polynomial_expand(&x, 1, false, 1e-8, Some(&names));
    assert!(result.info[0].name.starts_with("foo"));
    assert!(result.info[2].name.starts_with("bar"));
}

// ---------------------------------------------------------------------------
// Knockoff statistics
// ---------------------------------------------------------------------------

#[test]
fn test_w_statistics_basic() {
    let bo = vec![0.5, 0.0, 1.0, 0.2];
    let bk = vec![0.1, 0.3, 0.5, 0.2];
    let w = compute_w_statistics(&bo, &bk);
    assert!((w[0] - (0.5 - 0.1)).abs() < 1e-12);
    assert!((w[1] - (0.0 - 0.3)).abs() < 1e-12);
    assert!((w[2] - (1.0 - 0.5)).abs() < 1e-12);
    assert!((w[3] - 0.0).abs() < 1e-12);
}

#[test]
#[should_panic]
fn test_w_statistics_length_mismatch_panics() {
    compute_w_statistics(&[1.0, 2.0], &[1.0]);
}

#[test]
fn test_gaussian_knockoffs_shape_and_finite() {
    let n = 20;
    let p = 3;
    let mut x = Matrix::new(n, p, 0.0);
    for i in 0..n {
        for j in 0..p {
            x[(i, j)] = (i * p + j) as f64 * 0.1 + 1.0;
        }
    }
    let mu = col_mean(&x);
    let mut sigma = sample_cov(&x);
    for j in 0..p {
        sigma[(j, j)] += 0.5;
    }
    let xt = sample_gaussian_knockoffs(&x, &mu, &sigma, 42).expect("should succeed");
    assert_eq!(xt.rows, n);
    assert_eq!(xt.cols, p);
    assert!(xt.data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_gaussian_knockoffs_not_equal_to_original() {
    let n = 15;
    let p = 2;
    let mut x = Matrix::new(n, p, 0.0);
    for i in 0..n {
        for j in 0..p {
            x[(i, j)] = (i + j + 1) as f64;
        }
    }
    let mu = col_mean(&x);
    let mut sigma = sample_cov(&x);
    for j in 0..p {
        sigma[(j, j)] += 1.0;
    }
    let xt = sample_gaussian_knockoffs(&x, &mu, &sigma, 7).expect("should succeed");
    let identical: bool = x.data.iter().zip(&xt.data).all(|(a, b)| (a - b).abs() < 1e-12);
    assert!(!identical, "knockoffs should differ from original");
}

// ---------------------------------------------------------------------------
// PoSI alpha-spending
// ---------------------------------------------------------------------------

#[test]
fn test_riemann_zeta_budget_t1() {
    let q1 = alpha_spending_budget(1, 0.10, SpendingSequence::RiemannZeta, 0.5);
    let expected = 0.10 * 6.0 / (std::f64::consts::PI * std::f64::consts::PI);
    assert!((q1 - expected).abs() < 1e-10);
}

#[test]
fn test_riemann_zeta_sum_leq_q() {
    let total: f64 = (1..=1000)
        .map(|t| alpha_spending_budget(t, 0.10, SpendingSequence::RiemannZeta, 0.5))
        .sum();
    assert!(total <= 0.10 + 1e-6);
}

#[test]
fn test_geometric_budget_t1() {
    let q1 = alpha_spending_budget(1, 0.10, SpendingSequence::Geometric, 0.5);
    assert!((q1 - 0.05).abs() < 1e-10);
}

#[test]
fn test_geometric_sum_leq_q() {
    let total: f64 = (1..=200)
        .map(|t| alpha_spending_budget(t, 0.10, SpendingSequence::Geometric, 0.5))
        .sum();
    assert!(total <= 0.10 + 1e-6);
}

#[test]
fn test_budgets_array_length() {
    let budgets = alpha_spending_budgets(5, 0.10, SpendingSequence::RiemannZeta, 0.5);
    assert_eq!(budgets.len(), 5);
}

#[test]
fn test_riemann_zeta_decays() {
    let qs: Vec<f64> = (1..=5)
        .map(|t| alpha_spending_budget(t, 0.10, SpendingSequence::RiemannZeta, 0.5))
        .collect();
    for i in 0..4 {
        assert!(qs[i] > qs[i + 1]);
    }
}

#[test]
fn test_geometric_decays() {
    let qs: Vec<f64> = (1..=5)
        .map(|t| alpha_spending_budget(t, 0.10, SpendingSequence::Geometric, 0.5))
        .collect();
    for i in 0..4 {
        assert!(qs[i] > qs[i + 1]);
    }
}

#[test]
#[should_panic]
fn test_budget_t_zero_panics() {
    alpha_spending_budget(0, 0.10, SpendingSequence::RiemannZeta, 0.5);
}

#[test]
#[should_panic]
fn test_budget_invalid_q_panics() {
    alpha_spending_budget(1, 0.0, SpendingSequence::RiemannZeta, 0.5);
}

#[test]
fn test_knockoff_threshold_inf_on_zeros() {
    let w = vec![0.0, 0.0, 0.0];
    let tau = knockoff_threshold(&w, 0.10, &HashSet::new(), 1);
    assert!(tau.is_infinite());
}

#[test]
fn test_knockoff_threshold_empty() {
    let tau = knockoff_threshold(&[], 0.10, &HashSet::new(), 1);
    assert!(tau.is_infinite());
}

#[test]
fn test_knockoff_threshold_basic_selection() {
    let w = vec![3.0, 2.0, 1.5, 1.0];
    let tau = knockoff_threshold(&w, 0.5, &HashSet::new(), 1);
    assert!(tau <= 3.0);
    assert!(tau >= 0.0);
}

#[test]
fn test_knockoff_threshold_active_set_excluded() {
    let w = vec![5.0, 4.0, 3.0];
    let active: HashSet<usize> = vec![0].into_iter().collect();
    let tau = knockoff_threshold(&w, 0.10, &active, 1);
    assert!(tau >= 0.0 || tau.is_infinite());
}

#[test]
fn test_knockoff_threshold_no_selection_with_neg() {
    // 1 positive (2.0), 1 negative (-1.0) → ratio = (1+1)/max(1,2)=1 > any q<1 at tau=1
    // At tau=2: ratio = (1+0)/max(1,1)=1 > 0.20 → no selection
    let w = vec![2.0, -1.0];
    let tau = knockoff_threshold(&w, 0.20, &HashSet::new(), 1);
    assert!(tau.is_infinite());
}

#[test]
fn test_knockoff_threshold_offset_zero_vs_one() {
    let w = vec![2.0, 1.5, -0.5, -1.0];
    let tau_plus = knockoff_threshold(&w, 0.5, &HashSet::new(), 1);
    let tau_plain = knockoff_threshold(&w, 0.5, &HashSet::new(), 0);
    // knockoff+ is conservative: tau+ >= tau_plain or both infinite
    assert!(tau_plus >= tau_plain || tau_plus.is_infinite());
}

#[test]
fn test_forward_backward_substitution() {
    // L = [[2,0],[1,3]], b = [4, 7]
    // Forward: 2x0=4 → x0=2; 1*2+3*x1=7 → x1=5/3
    let mut l = Matrix::new(2, 2, 0.0);
    l[(0, 0)] = 2.0; l[(1, 0)] = 1.0; l[(1, 1)] = 3.0;
    let b = vec![4.0, 7.0];
    let y = forward_substitution(&l, &b);
    assert!((y[0] - 2.0).abs() < 1e-12);
    assert!((y[1] - 5.0 / 3.0).abs() < 1e-12);

    // Back substitution: L^T x = y
    // L^T = [[2,1],[0,3]]
    // 3*x1 = y1 → x1 = y1/3; 2*x0+x1=y0 → x0=(y0-x1)/2
    let x = backward_substitution(&l, &y);
    // verify L^T x = y: [[2,1],[0,3]] * x = y
    let y_check = vec![2.0 * x[0] + 1.0 * x[1], 3.0 * x[1]];
    assert!((y_check[0] - y[0]).abs() < 1e-10);
    assert!((y_check[1] - y[1]).abs() < 1e-10);
}

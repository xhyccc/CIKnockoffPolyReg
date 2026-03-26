//! Demo binary for IC-Knock-Poly Rust library.
//!
//! Runs a simple end-to-end demonstration of the PoSI threshold computation
//! and polynomial dictionary expansion on synthetic data.

use ic_knockoff_poly_reg::knockoffs::{compute_w_statistics, sample_gaussian_knockoffs};
use ic_knockoff_poly_reg::matrix::{col_mean, sample_cov};
use ic_knockoff_poly_reg::polynomial::polynomial_expand;
use ic_knockoff_poly_reg::posi::{alpha_spending_budget, knockoff_threshold, SpendingSequence};

fn main() {
    println!("=== IC-Knock-Poly Rust Demo ===\n");

    // ------------------------------------------------------------------
    // 1. Alpha-spending budgets
    // ------------------------------------------------------------------
    println!("1. Alpha-spending budgets (Q=0.10, Riemann Zeta):");
    for t in 1..=5 {
        let q = alpha_spending_budget(t, 0.10, SpendingSequence::RiemannZeta, 0.5);
        println!("   t={t}: q_{t} = {q:.6}");
    }

    println!("\n2. Alpha-spending budgets (Q=0.10, Geometric gamma=0.5):");
    for t in 1..=5 {
        let q = alpha_spending_budget(t, 0.10, SpendingSequence::Geometric, 0.5);
        println!("   t={t}: q_{t} = {q:.6}");
    }

    // ------------------------------------------------------------------
    // 2. Polynomial expansion
    // ------------------------------------------------------------------
    println!("\n3. Polynomial dictionary expansion (degree=2, 1 sample, 2 features):");
    let mut x = ic_knockoff_poly_reg::matrix::Matrix::new(1, 2, 0.0);
    x[(0, 0)] = 3.0;
    x[(0, 1)] = 2.0;
    let result = polynomial_expand(&x, 2, true, false, 1e-8, None);
    for (col, info) in result.info.iter().enumerate() {
        println!("   col {:2}: {:10} = {:.4}", col, info.name, result.matrix[(0, col)]);
    }

    // ------------------------------------------------------------------
    // 3. W-statistics and knockoff threshold
    // ------------------------------------------------------------------
    println!("\n4. W-statistics and knockoff+ threshold:");
    let beta_orig = vec![0.8, 0.0, 1.2, 0.1, 0.05];
    let beta_knock = vec![0.2, 0.1, 0.3, 0.05, 0.08];
    let w = compute_w_statistics(&beta_orig, &beta_knock);
    println!("   W = {:?}", w);
    let q1 = alpha_spending_budget(1, 0.10, SpendingSequence::RiemannZeta, 0.5);
    let tau = knockoff_threshold(&w, q1, &std::collections::HashSet::new(), 1);
    if tau.is_infinite() {
        println!("   tau_1 = inf (no selections)");
    } else {
        println!("   tau_1 = {tau:.4} (select features with W >= {tau:.4})");
        let selected: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|(_, &wj)| wj >= tau)
            .map(|(j, _)| j)
            .collect();
        println!("   Selected features: {selected:?}");
    }

    // ------------------------------------------------------------------
    // 4. Gaussian knockoff sampling
    // ------------------------------------------------------------------
    println!("\n5. Gaussian knockoff sample (n=10, p=3):");
    let mut data = ic_knockoff_poly_reg::matrix::Matrix::new(10, 3, 0.0);
    let vals = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
        28.0, 29.0, 30.0,
    ];
    for i in 0..10 {
        for j in 0..3 {
            data[(i, j)] = vals[i * 3 + j] * 0.1;
        }
    }
    let mu = col_mean(&data);
    let mut sigma = sample_cov(&data);
    for j in 0..3 {
        sigma[(j, j)] += 0.5; // regularise
    }
    match sample_gaussian_knockoffs(&data, &mu, &sigma, 42) {
        Ok(xt) => {
            println!("   Knockoff matrix shape: {}×{}", xt.rows, xt.cols);
            println!("   First row of knockoffs: {:?}", &xt.data[..3]);
            let all_finite = xt.data.iter().all(|x| x.is_finite());
            println!("   All values finite: {all_finite}");
        }
        Err(e) => println!("   Error: {e}"),
    }

    println!("\nDemo complete.");
}

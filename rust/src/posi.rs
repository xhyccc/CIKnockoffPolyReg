//! PoSI α-spending sequences and knockoff+ threshold computation.
//!
//! Implements Section 3 of the IC-Knock-Poly paper.
//!
//! ## Alpha-spending sequences
//!
//! Two sequences satisfying Σ q_t ≤ Q are provided:
//!
//! - **Riemann Zeta** (default):  `q_t = Q · 6 / (π² · t²)`
//! - **Geometric**:               `q_t = Q · (1−γ) · γ^(t−1)`
//!
//! ## Knockoff+ threshold
//!
//! ```text
//! τ_t = min{ τ > 0 :
//!     (1 + |{j ∉ A_poly : W_j ≤ −τ}|)
//!     / max(1, |{j ∉ A_poly : W_j ≥ τ}|)  ≤ q_t }
//! ```

use std::collections::HashSet;

/// Available alpha-spending sequence types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpendingSequence {
    /// Riemann Zeta: q_t = Q · 6 / (π² · t²). Default choice.
    RiemannZeta,
    /// Geometric: q_t = Q · (1−γ) · γ^(t−1).
    Geometric,
}

/// Return the FDR budget q_t for iteration t (1-indexed).
///
/// # Parameters
/// - `t`:        Iteration index (≥ 1).
/// - `q`:        Global target FDR level in (0, 1).
/// - `sequence`: Spending sequence type.
/// - `gamma`:    Geometric decay rate (only used for [`SpendingSequence::Geometric`]).
///
/// # Panics
/// Panics if `t < 1`, `q ∉ (0,1)`, or (`sequence == Geometric` and `gamma ∉ (0,1)`).
pub fn alpha_spending_budget(
    t: usize,
    q: f64,
    sequence: SpendingSequence,
    gamma: f64,
) -> f64 {
    assert!(t >= 1, "alpha_spending_budget: t must be >= 1");
    assert!(q > 0.0 && q < 1.0, "alpha_spending_budget: Q must be in (0,1)");
    match sequence {
        SpendingSequence::RiemannZeta => {
            let t_f = t as f64;
            q * 6.0 / (std::f64::consts::PI * std::f64::consts::PI * t_f * t_f)
        }
        SpendingSequence::Geometric => {
            assert!(
                gamma > 0.0 && gamma < 1.0,
                "alpha_spending_budget: gamma must be in (0,1) for Geometric sequence"
            );
            q * (1.0 - gamma) * gamma.powi(t as i32 - 1)
        }
    }
}

/// Compute q_1, ..., q_{max_t} and return them as a `Vec<f64>`.
pub fn alpha_spending_budgets(
    max_t: usize,
    q: f64,
    sequence: SpendingSequence,
    gamma: f64,
) -> Vec<f64> {
    (1..=max_t)
        .map(|t| alpha_spending_budget(t, q, sequence, gamma))
        .collect()
}

/// Compute the knockoff+ threshold τ_t.
///
/// Returns `f64::INFINITY` if no τ satisfies the FDR condition (nothing selected).
///
/// # Parameters
/// - `w`:          W-statistic vector for all candidate features.
/// - `q_t`:        FDR budget for the current iteration.
/// - `active_poly`: Set of indices already in the active polynomial set
///                  (excluded from computation).
/// - `offset`:     1 for knockoff+ (default), 0 for standard knockoff.
pub fn knockoff_threshold(
    w: &[f64],
    q_t: f64,
    active_poly: &HashSet<usize>,
    offset: i64,
) -> f64 {
    // Candidate W values (excluding active set)
    let w_cand: Vec<f64> = w
        .iter()
        .enumerate()
        .filter(|(j, _)| !active_poly.contains(j))
        .map(|(_, &v)| v)
        .collect();

    if w_cand.is_empty() {
        return f64::INFINITY;
    }

    // Unique positive |W| values as threshold candidates, sorted ascending
    let mut candidates: Vec<f64> = w_cand
        .iter()
        .filter(|&&v| v != 0.0)
        .map(|&v| v.abs())
        .collect();
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    candidates.dedup();

    if candidates.is_empty() {
        return f64::INFINITY;
    }

    for &tau in &candidates {
        let n_neg = w_cand.iter().filter(|&&v| v <= -tau).count() as i64;
        let n_pos = w_cand.iter().filter(|&&v| v >= tau).count() as i64;
        let ratio = (offset + n_neg) as f64 / 1_i64.max(n_pos) as f64;
        if ratio <= q_t {
            return tau;
        }
    }
    f64::INFINITY
}

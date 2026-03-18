//! IC-Knock-Poly: Iterative Conditional Knockoffs for Sparse Rational
//! Polynomial Regression.
//!
//! This crate provides the core statistical routines for the IC-Knock-Poly
//! algorithm:
//!
//! - [`polynomial`]: Rational polynomial dictionary expansion Φ(·)
//! - [`knockoffs`]: Knockoff W-statistic computation and Gaussian knockoff sampling
//! - [`posi`]: PoSI α-spending sequences and knockoff+ threshold computation
//! - [`matrix`]: Dense matrix utilities (arithmetic, Cholesky, inversion)
//! - [`baselines`]: Sparse polynomial regression baselines for comparison

pub mod baselines;
pub mod knockoffs;
pub mod matrix;
pub mod polynomial;
pub mod posi;

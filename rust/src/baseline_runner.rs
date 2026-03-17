//! Comparison runner for IC-Knock-Poly Rust baselines.
//!
//! Loads a CSV dataset and runs all three polynomial regression baselines:
//! PolyLasso, PolyOMP, and PolySTLSQ.
//!
//! Usage:
//! ```
//! cargo run --bin baseline_runner -- data.csv [degree=2] [output_prefix]
//! ```
//!
//! The CSV must have a header row; the last column is treated as the response.
//! Results are written to:
//! - `<output_prefix>_results.json`
//! - `<output_prefix>_results.csv`
//!
//! If no output_prefix is given, results are only printed to stdout.

use ic_knockoff_poly_reg::baselines::{PolyLasso, PolyOMP, PolySTLSQ};
use ic_knockoff_poly_reg::matrix::Matrix;
use std::fs;
use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CSV loader
// ---------------------------------------------------------------------------

struct CsvData {
    x: Matrix,
    y: Vec<f64>,
    #[allow(dead_code)]
    source: String,
}

fn load_csv(path: &str) -> Result<CsvData, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Cannot read {path}: {e}"))?;
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());

    // First line: try parsing as numbers; if it fails, treat as header and skip
    let first = lines.next().ok_or("CSV is empty")?;
    let first_parsed: Vec<Result<f64, _>> =
        first.split(',').map(|t| t.trim().parse::<f64>()).collect();
    let has_header = first_parsed.iter().any(|r| r.is_err());

    let data_lines: Vec<&str> = if has_header {
        lines.collect()
    } else {
        // Re-add the first line as data
        std::iter::once(first).chain(lines).collect()
    };

    if data_lines.is_empty() {
        return Err("CSV has no data rows".into());
    }

    let rows: Vec<Vec<f64>> = data_lines
        .iter()
        .map(|line| {
            line.split(',')
                .map(|t| t.trim().parse::<f64>().unwrap_or(0.0))
                .collect()
        })
        .collect();

    let n = rows.len();
    let n_cols = rows[0].len();
    if n_cols < 2 {
        return Err("CSV must have at least 2 columns".into());
    }
    let p = n_cols - 1;

    let mut x = Matrix::new(n, p, 0.0);
    let mut y = vec![0.0_f64; n];
    for (i, row) in rows.iter().enumerate() {
        for j in 0..p {
            x[(i, j)] = row.get(j).copied().unwrap_or(0.0);
        }
        y[i] = row.get(p).copied().unwrap_or(0.0);
    }
    Ok(CsvData { x, y, source: path.to_string() })
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <data.csv> [degree=2] [output_prefix]",
            args[0]
        );
        std::process::exit(1);
    }

    let csv_path = &args[1];
    let degree: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let out_prefix: Option<&str> = args.get(3).map(|s| s.as_str());

    let data = match load_csv(csv_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading CSV: {e}");
            std::process::exit(1);
        }
    };

    println!("Dataset : {csv_path}");
    println!("Shape   : {} x {}", data.x.rows, data.x.cols);
    println!("Degree  : {degree}\n");

    let mut results = Vec::new();

    // ---- PolyLasso ----
    print!("Running PolyLasso ...  ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let mut lasso = PolyLasso::new(degree);
    lasso.fit(&data.x, &data.y);
    let elapsed = t0.elapsed().as_secs_f64();
    let rb = lasso.to_result_bundle(&data.x, &data.y, csv_path, elapsed);
    println!(
        "selected={}  R2={:.4}  t={:.2}s",
        rb.n_selected, rb.r_squared, elapsed
    );
    results.push(rb);

    // ---- PolyOMP ----
    print!("Running PolyOMP    ...  ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let mut omp = PolyOMP::new(degree);
    omp.fit(&data.x, &data.y);
    let elapsed = t0.elapsed().as_secs_f64();
    let rb = omp.to_result_bundle(&data.x, &data.y, csv_path, elapsed);
    println!(
        "selected={}  R2={:.4}  t={:.2}s",
        rb.n_selected, rb.r_squared, elapsed
    );
    results.push(rb);

    // ---- PolySTLSQ ----
    print!("Running PolySTLSQ  ...  ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let mut stlsq = PolySTLSQ::new(degree);
    stlsq.fit(&data.x, &data.y);
    let elapsed = t0.elapsed().as_secs_f64();
    let rb = stlsq.to_result_bundle(&data.x, &data.y, csv_path, elapsed);
    println!(
        "selected={}  R2={:.4}  t={:.2}s",
        rb.n_selected, rb.r_squared, elapsed
    );
    results.push(rb);

    // ---- Print table ----
    println!();
    println!(
        "{:<24}  {:>8}  {:>8}  {:>10}  {:>12}",
        "Method", "N_sel", "R2", "BIC", "Elapsed(s)"
    );
    println!("{}", "-".repeat(68));
    for r in &results {
        let fmt = |v: f64| {
            if v.is_nan() || v.is_infinite() {
                "     N/A".to_string()
            } else {
                format!("{:8.4}", v)
            }
        };
        println!(
            "{:<24}  {:>8}  {}  {}  {:>12.4}",
            r.method,
            r.n_selected,
            fmt(r.r_squared),
            fmt(r.bic),
            r.elapsed_seconds
        );
    }
    println!();

    // ---- Write files ----
    if let Some(prefix) = out_prefix {
        let json_path = format!("{prefix}_results.json");
        let csv_path_out = format!("{prefix}_results.csv");

        let json_arr = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i == 0 {
                    r.to_json()
                } else {
                    format!(",\n{}", r.to_json())
                }
            })
            .collect::<Vec<_>>()
            .join("");
        let json_str = format!("[\n{json_arr}\n]");

        fs::write(&json_path, json_str).ok();
        let mut csv_content = String::new();
        for (i, r) in results.iter().enumerate() {
            csv_content.push_str(&r.to_csv_row(i == 0));
        }
        fs::write(&csv_path_out, csv_content).ok();
        println!("Results written to {prefix}_results.json/.csv");
    }
}

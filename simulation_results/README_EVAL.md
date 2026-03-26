# IC-Knock-Poly Evaluation Scripts

This directory contains scripts for running and evaluating IC-Knock-Poly experiments.

## Scripts

### 1. `run_advantage_experiments.py`
Runs experiments designed to showcase IC-Knock-Poly's advantages:
- **High-dimensional**: p=20,30 with n=60,80 (n << dict_size)
- **High-degree**: d=3,4 (cubic, quartic)
- **High sparsity**: k=8,10,12
- **Challenging ratios**: n/dict_size < 1

**Usage:**
```bash
cd simulation_results
python run_advantage_experiments.py
```

**Output:**
- `advantage_results/advantage_experiments_*.json` - Raw results
- `advantage_results/summary_*.txt` - Text summary

### 2. `evaluate_experiments.py`
Analyzes experiment results and generates visualizations.

**Usage:**
```bash
# Analyze most recent results
python evaluate_experiments.py

# Analyze specific file
python evaluate_experiments.py advantage_results/advantage_experiments_20260323_135709.json
```

**Output:**
- Console output with tables and statistics
- `figures/01_overall_performance.png` - FDR/TPR comparison
- `figures/02_fdr_tpr_tradeoff.png` - Scatter plot

### 3. `run_large_scale_experiments.py`
Comprehensive experiments across parameter grid.

**Usage:**
```bash
python run_large_scale_experiments.py
```

**Configuration:**
- p: 10, 15
- n: 100, 200
- k: 3, 6
- degree: 2, 3
- noise: 0.1, 0.5
- trials: 3

## Quick Start

1. **Run advantage experiments** (shows IC-Knock-Poly strengths):
```bash
cd simulation_results
python run_advantage_experiments.py 2>&1 | tee advantage_run.log
```

2. **Evaluate results**:
```bash
python evaluate_experiments.py
```

3. **Check figures**:
```bash
ls advantage_results/figures/
```

## Profiling Metrics

The advantage experiments include fine-grained profiling:

- `time_per_sample_ms`: Time divided by n samples
- `time_per_feature_ms`: Time divided by dictionary size
- `time_per_term_ms`: Time divided by selected terms
- `memory_per_sample_kb`: Memory divided by n samples
- `memory_per_feature_kb`: Memory divided by dictionary size
- `n_to_dict_ratio`: n / dict_size (difficulty metric)

## Expected Runtime

- Advantage experiments: ~20-30 minutes (36 configs × 5 trials × 5 methods = 900 runs)
- Large-scale experiments: ~20 minutes (32 configs × 3 trials × 5 methods = 480 runs)

## Interpreting Results

### Key Metrics
- **FDR**: Lower is better (target = 0.2)
- **TPR**: Higher is better (1.0 = perfect)
- **Time**: Lower is better (check scaling)
- **Memory**: Lower is better

### Advantage Scenarios
IC-Knock-Poly should perform well when:
1. n/dict_ratio < 0.5 (hard estimation)
2. degree ≥ 3 (complex polynomials)
3. k ≥ 8 (many terms to find)
4. p ≥ 20 (high-dimensional)

## Troubleshooting

**Out of memory:**
- Reduce n or p in configurations
- Reduce N_TRIALS

**Too slow:**
- Ensure Rust library is built: `cd rust && cargo build --release`
- Check that backend="rust" is set

**Import errors:**
- Make sure you're in the `simulation_results` directory
- Verify PYTHONPATH includes `../python/src`

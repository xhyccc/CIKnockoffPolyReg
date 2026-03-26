# Large-Scale Experiments Guide

## 🚀 How to Run Outside This Session

### Step 1: Navigate to Repository
```bash
cd /Users/haoyi/Desktop/CIKnockoffPolyReg
```

### Step 2: Verify Rust Build
```bash
cd rust
cargo build --release
cd ..
```

### Step 3: Run Large-Scale Experiments
```bash
python simulation_results/run_large_scale_experiments.py
```

## 📊 Experiment Specifications

### Configuration Grid
- **p (dimension)**: 10, 15, 20
- **n (samples)**: 100, 200, 300
- **k (sparsity)**: 2, 4, 6, 8, 10 (polynomial terms)
- **d (degree)**: 1, 2, 3

### Dictionary Sizes
- **Degree 1**: 2p terms (e.g., p=20 → 40 terms)
- **Degree 2**: 4p terms (e.g., p=20 → 80 terms)
- **Degree 3**: 6p terms (e.g., p=20 → 120 terms)

### Sparsity Levels
Testing k up to 10 means:
- **Ultra-sparse**: k=2 (1.7% - 5% of dictionary)
- **Sparse**: k=6 (5% - 15% of dictionary)
- **Moderate**: k=10 (8% - 25% of dictionary)

## 📈 What Gets Measured

For each experiment:
1. **FDR** (False Discovery Rate)
2. **TPR** (True Positive Rate / Recall)
3. **F1 Score**
4. **Number of selections**
5. **Exact selected terms** [base, exponent]
6. **Execution time**
7. **Peak memory usage**
8. **R² score**
9. **BIC**

## 💾 Output Files

Results are saved with timestamps:
```
simulation_results/large_scale_results/
├── large_scale_experiments_YYYYMMDD_HHMMSS.json
└── summary_YYYYMMDD_HHMMSS.txt
```

### JSON Structure
```json
{
  "timestamp": "20240322_143052",
  "completed": 150,
  "total": 150,
  "total_time_minutes": 45.2,
  "system_info": {
    "total_memory_gb": 16.0,
    "available_memory_gb": 12.5,
    "python_version": "3.10.0"
  },
  "results": [
    {
      "method": "IC-Knock-Poly",
      "config": {"n": 100, "p": 15, "k": 6, "degree": 2, ...},
      "success": true,
      "fdr": 0.125,
      "tpr": 0.833,
      "f1": 0.741,
      "n_selected": 7,
      "n_true_positives": 5,
      "n_false_positives": 2,
      "selected_terms": [[0, 2], [1, -1], ...],
      "selected_names": ["x0^2", "x1^(-1)", ...],
      "true_terms": [[0, 2], [1, -1], [3, 1], ...],
      "r_squared": 0.987,
      "time_seconds": 2.34,
      "peak_memory_mb": 45.2,
      "memory_increase_mb": 12.5
    },
    ...
  ]
}
```

### Summary File Content
- Overall statistics by method
- Statistics by polynomial degree
- Statistics by sparsity (k)
- Key findings and conclusions

## ⏱️ Expected Runtime

- **~200-300 experiments** (configurations × 5 methods)
- **Estimated time**: 30-60 minutes
- **Memory**: Peak ~4-6 GB (well within 16GB limit)
- **Saves intermediate results** every 3 configurations

## 🎯 What to Look For

### Expected Results Pattern:

**Low k (k=2, ultra-sparse):**
- All methods perform well
- IC-Knock-Poly: TPR ~0.9-1.0, FDR ~0.0-0.1
- Others: TPR ~0.7-0.9, FDR ~0.0-0.2

**Medium k (k=6, sparse):**
- IC-Knock-Poly advantage emerges
- IC-Knock-Poly: TPR ~0.8-0.9, FDR ~0.1-0.2
- Others: TPR ~0.4-0.6, FDR ~0.1-0.3

**High k (k=10, moderate):**
- IC-Knock-Poly clearly superior
- IC-Knock-Poly: TPR ~0.6-0.8, FDR ~0.2-0.3
- Others: TPR ~0.3-0.5, FDR ~0.2-0.4

### Memory Usage Pattern:
- **IC-Knock-Poly**: 50-150 MB (GMM + iterations)
- **Poly-Knockoff/CLIME**: 20-50 MB (single Gaussian)
- **Poly-Lasso/OMP**: 10-30 MB (simple models)

## 🔍 Analyzing Results

After running, you can analyze results with Python:

```python
import json
import pandas as pd

# Load results
with open('simulation_results/large_scale_results/large_scale_experiments_*.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['results'])

# Filter successful runs
df_success = df[df['success'] == True]

# Compare methods
summary = df_success.groupby('method').agg({
    'tpr': 'mean',
    'fdr': 'mean',
    'time_seconds': 'mean',
    'peak_memory_mb': 'mean'
})
print(summary)
```

## 🚨 Troubleshooting

**If memory error occurs:**
- Reduce p (try p=15 max instead of 20)
- Reduce n (try n=200 max)
- Script auto-saves, so you won't lose progress

**If too slow:**
- Run with fewer k values (comment out k=8,10)
- Run specific degree only (comment out degree 1 or 3)

**To stop and resume:**
- Press Ctrl+C to stop gracefully
- Intermediate results saved every 3 configs
- Restart script - it will create new timestamped files

## 📧 Results Location

All results saved in:
```
/Users/haoyi/Desktop/CIKnockoffPolyReg/simulation_results/large_scale_results/
```

Share the JSON file for detailed analysis!

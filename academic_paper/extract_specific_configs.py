#!/usr/bin/env python3
"""Extract specific config results for paper tables."""

import sys
sys.path.insert(0, '/Users/haoyi/Desktop/CIKnockoffPolyReg/python/src')
sys.path.insert(0, '/Users/haoyi/Desktop/CIKnockoffPolyReg/python')

import json
import pandas as pd
import numpy as np

# Load data
with open('/Users/haoyi/Desktop/CIKnockoffPolyReg/simulation_results/large_scale_results/final_val_improved_20260328_204101.json', 'r') as f:
    data1 = json.load(f)

with open('/Users/haoyi/Desktop/CIKnockoffPolyReg/simulation_results/large_scale_results/final_val_v2_20260330_000800.json', 'r') as f:
    data2 = json.load(f)

def process_results(results):
    """Convert results to DataFrame."""
    rows = []
    for r in results:
        row = {
            'method': r['method'],
            'n': r['config']['n'],
            'p': r['config']['p'],
            'k': r['config']['k'],
            'degree': r['config']['degree'],
            'noise': r['config'].get('noise', 0.1),
            'correlation': r['config'].get('correlation', 0.0),
            'fdr': r['fdr'],
            'tpr': r['tpr'],
            'test_r2': r.get('test_r2', np.nan),
            'n_selected': r['n_selected'],
            'trial': r['config']['trial']
        }
        rows.append(row)
    return pd.DataFrame(rows)

df1 = process_results(data1['results'])
df2 = process_results(data2['results'])

# Print specific configs for Table 1 (Main Results)
print("=" * 80)
print("TABLE 1: Main Results - Specific Config")
print("=" * 80)
print("\nConfig: n=100, p=5, k=3, degree=3\n")

# Experiment 1 - specific config
mask1 = (df1['n'] == 100) & (df1['p'] == 5) & (df1['k'] == 3) & (df1['degree'] == 3)
df1_config = df1[mask1]

print("Experiment 1 (Independent Base Features)")
print("-" * 60)
for method in ['IC-Knock-Poly-Val', 'Poly-Knockoff-Val', 'Poly-Lasso-Val', 'Poly-CLIME-Val', 'Poly-OMP-Val']:
    mdata = df1_config[df1_config['method'] == method]
    if len(mdata) > 0:
        fdr = mdata['fdr'].mean()
        tpr = mdata['tpr'].mean()
        r2 = mdata['test_r2'].median()
        print(f"{method:25s} | FDR: {fdr:.3f} | TPR: {tpr:.3f} | R²: {r2:.3f} | n_trials: {len(mdata)}")

# Experiment 2 - specific config
mask2 = (df2['n'] == 100) & (df2['p'] == 5) & (df2['k'] == 3) & (df2['degree'] == 3) & (df2['noise'] == 3.0)
df2_config = df2[mask2]

print("\nExperiment 2 (Correlated Base Features with Label Noises)")
print("-" * 60)
for method in ['IC-Knock-Poly-Val', 'Poly-Knockoff-Val', 'Poly-Lasso-Val', 'Poly-CLIME-Val', 'Poly-OMP-Val', 'Poly-STLSQ-Val']:
    mdata = df2_config[df2_config['method'] == method]
    if len(mdata) > 0:
        fdr = mdata['fdr'].mean()
        tpr = mdata['tpr'].mean()
        r2 = mdata['test_r2'].median()
        print(f"{method:25s} | FDR: {fdr:.3f} | TPR: {tpr:.3f} | R²: {r2:.3f} | n_trials: {len(mdata)}")

# Print varying p (Table 4) - fixed config: n=100, k=3, degree=3, noise=3.0
print("\n" + "=" * 80)
print("TABLE 4: Varying p - Fixed Config: n=100, k=3, degree=3, noise=3.0")
print("=" * 80)
print()

for p in [1, 3, 5, 7, 9]:
    mask = (df2['n'] == 100) & (df2['p'] == p) & (df2['k'] == 3) & (df2['degree'] == 3) & (df2['noise'] == 3.0)
    df_p = df2[mask]
    
    print(f"p = {p}:")
    mdata = df_p[df_p['method'] == 'IC-Knock-Poly-Val']
    if len(mdata) > 0:
        fdr = mdata['fdr'].mean()
        tpr = mdata['tpr'].mean()
        r2 = mdata['test_r2'].median()
        selected = mdata['n_selected'].mean()
        print(f"  IC-Knock-Poly-Val: FDR={fdr:.3f}, TPR={tpr:.3f}, R²={r2:.3f}, Selected={selected:.1f} (n_trials={len(mdata)})")
    print()

# Print varying n (Table 5) - fixed config: p=5, k=3, degree=3, noise=3.0
print("=" * 80)
print("TABLE 5: Varying n - Fixed Config: p=5, k=3, degree=3, noise=3.0")
print("=" * 80)
print()

for n in [50, 75, 100]:
    mask = (df2['n'] == n) & (df2['p'] == 5) & (df2['k'] == 3) & (df2['degree'] == 3) & (df2['noise'] == 3.0)
    df_n = df2[mask]
    
    print(f"n = {n}:")
    mdata = df_n[df_n['method'] == 'IC-Knock-Poly-Val']
    if len(mdata) > 0:
        fdr = mdata['fdr'].mean()
        tpr = mdata['tpr'].mean()
        r2 = mdata['test_r2'].median()
        selected = mdata['n_selected'].mean()
        print(f"  IC-Knock-Poly-Val: FDR={fdr:.3f}, TPR={tpr:.3f}, R²={r2:.3f}, Selected={selected:.1f} (n_trials={len(mdata)})")
    print()

# Print Fixed-Q analysis for specific config
print("=" * 80)
print("TABLE: Fixed-Q Analysis - Fixed Config: n=100, p=5, k=3, degree=3, noise=3.0")
print("=" * 80)
print()

mask = (df2['n'] == 100) & (df2['p'] == 5) & (df2['k'] == 3) & (df2['degree'] == 3) & (df2['noise'] == 3.0)
df_q = df2[mask]

for method in ['IC-Knock-Poly-Q0.05', 'IC-Knock-Poly-Q0.1', 'IC-Knock-Poly-Q0.15', 'IC-Knock-Poly-Val']:
    mdata = df_q[df_q['method'] == method]
    if len(mdata) > 0:
        fdr = mdata['fdr'].mean()
        tpr = mdata['tpr'].mean()
        r2 = mdata['test_r2'].median()
        print(f"{method:30s} | FDR: {fdr:.3f} | TPR: {tpr:.3f} | R²: {r2:.3f} | n_trials: {len(mdata)}")

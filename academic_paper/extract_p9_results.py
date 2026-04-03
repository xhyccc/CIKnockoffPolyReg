#!/usr/bin/env python3
"""Extract p=9 results for Table 1."""

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

print("=" * 80)
print("TABLE 1: Main Results with p=5 and p=9 comparison")
print("=" * 80)
print("\nConfig: n=100, k=3, degree=3\n")

# Experiment 1 - p=5 vs p=9
for p in [5, 9]:
    mask1 = (df1['n'] == 100) & (df1['p'] == p) & (df1['k'] == 3) & (df1['degree'] == 3)
    df1_config = df1[mask1]
    
    print(f"Experiment 1 (Independent Base Features) - p={p}:")
    print("-" * 60)
    for method in ['IC-Knock-Poly-Val', 'Poly-Knockoff-Val', 'Poly-Lasso-Val', 'Poly-CLIME-Val', 'Poly-OMP-Val']:
        mdata = df1_config[df1_config['method'] == method]
        if len(mdata) > 0:
            fdr = mdata['fdr'].mean()
            tpr = mdata['tpr'].mean()
            r2 = mdata['test_r2'].median()
            print(f"{method:25s} | FDR: {fdr:.3f} | TPR: {tpr:.3f} | R²: {r2:.3f} | n_trials: {len(mdata)}")
    print()

# Experiment 2 - p=5 vs p=9 (noise=3.0)
for p in [5, 9]:
    mask2 = (df2['n'] == 100) & (df2['p'] == p) & (df2['k'] == 3) & (df2['degree'] == 3) & (df2['noise'] == 3.0)
    df2_config = df2[mask2]
    
    print(f"Experiment 2 (Correlated Base Features with Label Noises) - p={p}:")
    print("-" * 60)
    for method in ['IC-Knock-Poly-Val', 'Poly-Knockoff-Val', 'Poly-Lasso-Val', 'Poly-CLIME-Val', 'Poly-OMP-Val', 'Poly-STLSQ-Val']:
        mdata = df2_config[df2_config['method'] == method]
        if len(mdata) > 0:
            fdr = mdata['fdr'].mean()
            tpr = mdata['tpr'].mean()
            r2 = mdata['test_r2'].median()
            print(f"{method:25s} | FDR: {fdr:.3f} | TPR: {tpr:.3f} | R²: {r2:.3f} | n_trials: {len(mdata)}")
    print()

print("=" * 80)
print("Summary: p=5 vs p=9 comparison for IC-Knock-Poly-Val")
print("=" * 80)
print()

for exp_name, df, noise in [("Exp 1 (Independent)", df1, None), ("Exp 2 (Correlated)", df2, 3.0)]:
    print(f"{exp_name}:")
    for p in [5, 9]:
        if noise:
            mask = (df['n'] == 100) & (df['p'] == p) & (df['k'] == 3) & (df['degree'] == 3) & (df['noise'] == noise)
        else:
            mask = (df['n'] == 100) & (df['p'] == p) & (df['k'] == 3) & (df['degree'] == 3)
        
        mdata = df[mask & (df['method'] == 'IC-Knock-Poly-Val')]
        if len(mdata) > 0:
            fdr = mdata['fdr'].mean()
            tpr = mdata['tpr'].mean()
            r2 = mdata['test_r2'].median()
            print(f"  p={p}: FDR={fdr:.3f}, TPR={tpr:.3f}, R²={r2:.3f}")
    print()

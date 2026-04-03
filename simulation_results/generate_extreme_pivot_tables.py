#!/usr/bin/env python3
"""Generate pivot tables for extreme experiment results."""

import sys
sys.path.insert(0, '/Users/haoyi/Desktop/CIKnockoffPolyReg/python/src')
sys.path.insert(0, '/Users/haoyi/Desktop/CIKnockoffPolyReg/python')

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
results_file = '/Users/haoyi/Desktop/CIKnockoffPolyReg/simulation_results/large_scale_results/final_val_extreme_20260401_145550.json'
with open(results_file, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data['results'])} results")

# Convert to DataFrame
rows = []
for r in data['results']:
    if 'config' not in r:
        continue
    row = {
        'method': r['method'],
        'n': r['config']['n'],
        'p': r['config']['p'],
        'k': r['config']['k'],
        'degree': r['config']['degree'],
        'noise': r['config'].get('noise', 10.0),
        'trial': r['config']['trial'],
        'fdr': r.get('fdr', np.nan),
        'tpr': r.get('tpr', np.nan),
        'test_r2': r.get('test_r2', np.nan),
        'test_rmse': r.get('test_rmse', np.nan),
        'n_selected': r.get('n_selected', np.nan)
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(f"DataFrame shape: {df.shape}")
print(f"Methods: {df['method'].unique()}")
print(f"Noise levels: {df['noise'].unique()}")
print()

# Summary by method
print("="*80)
print("SUMMARY BY METHOD (Extreme Experiment)")
print("="*80)
summary = df.groupby('method').agg({
    'fdr': ['mean', 'std'],
    'tpr': ['mean', 'std'],
    'test_r2': ['mean', 'std', 'median'],
    'test_rmse': ['mean', 'median'],
    'n_selected': 'mean'
}).round(3)
print(summary)
print()

# Summary by noise level
print("="*80)
print("SUMMARY BY NOISE LEVEL")
print("="*80)
for noise in sorted(df['noise'].unique()):
    print(f"\nNoise = {noise}:")
    mask = df['noise'] == noise
    sub = df[mask]
    for method in ['IC-Knock-Poly-Val', 'Poly-Lasso-Val', 'Poly-Knockoff-Val']:
        mdata = sub[sub['method'] == method]
        if len(mdata) > 0:
            print(f"  {method:25s}: FDR={mdata['fdr'].mean():.3f}, TPR={mdata['tpr'].mean():.3f}, R²={mdata['test_r2'].median():.3f}")

# Representative config: n=100, p=5, k=3, degree=3
print()
print("="*80)
print("REPRESENTATIVE CONFIG: n=100, p=5, k=3, degree=3")
print("="*80)
mask = (df['n'] == 100) & (df['p'] == 5) & (df['k'] == 3) & (df['degree'] == 3)
df_rep = df[mask]

for noise in sorted(df_rep['noise'].unique()):
    print(f"\nNoise = {noise}:")
    sub = df_rep[df_rep['noise'] == noise]
    for method in ['IC-Knock-Poly-Val', 'Poly-Lasso-Val', 'Poly-Knockoff-Val', 'Poly-CLIME-Val', 'Poly-OMP-Val', 'Poly-STLSQ-Val']:
        mdata = sub[sub['method'] == method]
        if len(mdata) > 0:
            print(f"  {method:25s}: FDR={mdata['fdr'].mean():.3f}±{mdata['fdr'].std():.3f}, TPR={mdata['tpr'].mean():.3f}±{mdata['tpr'].std():.3f}")
            print(f"                           R²={mdata['test_r2'].median():.3f}, RMSE={mdata['test_rmse'].median():.3f}, #Sel={mdata['n_selected'].mean():.1f}")

# Create pivot tables and save to CSV
output_dir = Path('/Users/haoyi/Desktop/CIKnockoffPolyReg/simulation_results/large_scale_results/figure_extreme_20260402')
output_dir.mkdir(exist_ok=True)

# Pivot 1: Varying p (fixed n=100, k=3, degree=3, noise=10.0)
print()
print("="*80)
print("VARYING P (n=100, k=3, d=3, noise=10.0)")
print("="*80)
mask = (df['n'] == 100) & (df['k'] == 3) & (df['degree'] == 3) & (df['noise'] == 10.0)
df_p = df[mask]
pivot_p = df_p.pivot_table(
    index='p',
    columns='method',
    values=['fdr', 'tpr', 'test_r2', 'n_selected'],
    aggfunc={'fdr': 'mean', 'tpr': 'mean', 'test_r2': 'median', 'n_selected': 'mean'}
)
print(pivot_p)
pivot_p.to_csv(output_dir / 'controlled_varying_p_noise10.csv')

# Pivot 2: Varying noise (fixed n=100, p=5, k=3, degree=3)
print()
print("="*80)
print("VARYING NOISE (n=100, p=5, k=3, d=3)")
print("="*80)
mask = (df['n'] == 100) & (df['p'] == 5) & (df['k'] == 3) & (df['degree'] == 3)
df_noise = df[mask]
pivot_noise = df_noise.pivot_table(
    index='noise',
    columns='method',
    values=['fdr', 'tpr', 'test_r2', 'n_selected'],
    aggfunc={'fdr': 'mean', 'tpr': 'mean', 'test_r2': 'median', 'n_selected': 'mean'}
)
print(pivot_noise)
pivot_noise.to_csv(output_dir / 'controlled_varying_noise.csv')

print()
print(f"Pivot tables saved to: {output_dir}")

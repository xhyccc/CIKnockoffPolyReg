#!/usr/bin/env python3
"""Generate PDF figures for experiment results."""

import sys
sys.path.insert(0, '/Users/haoyi/Desktop/CIKnockoffPolyReg/python/src')
sys.path.insert(0, '/Users/haoyi/Desktop/CIKnockoffPolyReg/python')

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 11

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
            'fdr': r['fdr'],
            'tpr': r['tpr'],
            'test_r2': r.get('test_r2', np.nan),
            'n_selected': r['n_selected']
        }
        rows.append(row)
    return pd.DataFrame(rows)

df1 = process_results(data1['results'])
df2 = process_results(data2['results'])

# Create PDF pages
output_dir = '/Users/haoyi/Desktop/CIKnockoffPolyReg/academic_paper/figures'

# Figure 1: p scaling (Experiment 2)
with PdfPages(f'{output_dir}/fig_p_scaling.pdf') as pdf:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Filter for Exp 2, n=100, k=3, degree=3, noise=3.0
    mask = (df2['n'] == 100) & (df2['k'] == 3) & (df2['degree'] == 3) & (df2['noise'] == 3.0)
    df_plot = df2[mask]
    
    methods = ['IC-Knock-Poly-Val', 'Poly-Knockoff-Val', 'Poly-Lasso-Val']
    colors = {'IC-Knock-Poly-Val': '#1f77b4', 'Poly-Knockoff-Val': '#ff7f0e', 'Poly-Lasso-Val': '#2ca02c'}
    
    # Plot 1: FDR vs p
    ax = axes[0, 0]
    for method in methods:
        mdata = df_plot[df_plot['method'] == method].groupby('p')['fdr'].mean()
        ax.plot(mdata.index, mdata.values, marker='o', linewidth=2, 
                label=method.replace('-Val', ''), color=colors[method], markersize=8)
    ax.set_xlabel('Feature Dimension $p$', fontsize=12)
    ax.set_ylabel('FDR (False Discovery Rate)', fontsize=12)
    ax.set_title('(a) FDR vs Feature Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: TPR vs p
    ax = axes[0, 1]
    for method in methods:
        mdata = df_plot[df_plot['method'] == method].groupby('p')['tpr'].mean()
        ax.plot(mdata.index, mdata.values, marker='o', linewidth=2,
                label=method.replace('-Val', ''), color=colors[method], markersize=8)
    ax.set_xlabel('Feature Dimension $p$', fontsize=12)
    ax.set_ylabel('TPR (True Positive Rate)', fontsize=12)
    ax.set_title('(b) TPR vs Feature Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Test R2 vs p
    ax = axes[1, 0]
    for method in methods:
        mdata = df_plot[df_plot['method'] == method].groupby('p')['test_r2'].median()
        ax.plot(mdata.index, mdata.values, marker='o', linewidth=2,
                label=method.replace('-Val', ''), color=colors[method], markersize=8)
    ax.set_xlabel('Feature Dimension $p$', fontsize=12)
    ax.set_ylabel('Test $R^2$ (median)', fontsize=12)
    ax.set_title('(c) Prediction Accuracy vs Feature Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: N selected vs p
    ax = axes[1, 1]
    for method in methods:
        mdata = df_plot[df_plot['method'] == method].groupby('p')['n_selected'].mean()
        ax.plot(mdata.index, mdata.values, marker='o', linewidth=2,
                label=method.replace('-Val', ''), color=colors[method], markersize=8)
    ax.set_xlabel('Feature Dimension $p$', fontsize=12)
    ax.set_ylabel('Number of Selected Features', fontsize=12)
    ax.set_title('(d) Model Sparsity vs Feature Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(r'Performance Scaling with Feature Dimension (Exp 2: $\rho=0.8$, noise=3.0)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print('✓ Generated fig_p_scaling.pdf')

# Figure 2: n scaling (bar chart)
with PdfPages(f'{output_dir}/fig_n_scaling.pdf') as pdf:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Filter for Exp 2, p=5, k=3, degree=3, noise=3.0
    mask = (df2['p'] == 5) & (df2['k'] == 3) & (df2['degree'] == 3) & (df2['noise'] == 3.0)
    df_plot = df2[mask]
    
    # Get n values and methods
    n_values = sorted(df_plot['n'].unique())
    methods = ['IC-Knock-Poly-Val', 'Poly-Knockoff-Val', 'Poly-Lasso-Val']
    colors = {'IC-Knock-Poly-Val': '#1f77b4', 'Poly-Knockoff-Val': '#ff7f0e', 'Poly-Lasso-Val': '#2ca02c'}
    x = np.arange(len(n_values))
    width = 0.25
    
    # Plot 1: FDR bar chart
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        values = [df_plot[(df_plot['method'] == method) & (df_plot['n'] == n)]['fdr'].mean() for n in n_values]
        ax.bar(x + i*width, values, width, label=method.replace('-Val', ''), color=colors[method], alpha=0.8)
    ax.set_xlabel('Sample Size $n$', fontsize=12)
    ax.set_ylabel('FDR (False Discovery Rate)', fontsize=12)
    ax.set_title('(a) FDR vs Sample Size', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(n_values)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: TPR bar chart
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        values = [df_plot[(df_plot['method'] == method) & (df_plot['n'] == n)]['tpr'].mean() for n in n_values]
        ax.bar(x + i*width, values, width, label=method.replace('-Val', ''), color=colors[method], alpha=0.8)
    ax.set_xlabel('Sample Size $n$', fontsize=12)
    ax.set_ylabel('TPR (True Positive Rate)', fontsize=12)
    ax.set_title('(b) TPR vs Sample Size', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(n_values)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Test R2 bar chart
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        values = [df_plot[(df_plot['method'] == method) & (df_plot['n'] == n)]['test_r2'].median() for n in n_values]
        ax.bar(x + i*width, values, width, label=method.replace('-Val', ''), color=colors[method], alpha=0.8)
    ax.set_xlabel('Sample Size $n$', fontsize=12)
    ax.set_ylabel('Test $R^2$ (median)', fontsize=12)
    ax.set_title('(c) Prediction Accuracy vs Sample Size', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(n_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Key observations
    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.1, 0.8, 'Key Observations:', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.65, '• IC-Knock-Poly improves significantly', fontsize=11, transform=ax.transAxes)
    ax.text(0.1, 0.55, '  with more data (n=50→100)', fontsize=11, transform=ax.transAxes)
    ax.text(0.1, 0.40, '• Baselines show minimal improvement', fontsize=11, transform=ax.transAxes)
    ax.text(0.1, 0.30, '  even with more samples', fontsize=11, transform=ax.transAxes)
    ax.text(0.1, 0.15, '• Data efficiency is a key advantage', fontsize=11, transform=ax.transAxes)
    
    plt.suptitle(r'Performance Scaling with Sample Size (Exp 2: $\rho=0.8$, noise=3.0)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print('✓ Generated fig_n_scaling.pdf')

# Figure 3: Noise level comparison
with PdfPages(f'{output_dir}/fig_noise_comparison.pdf') as pdf:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Compare Exp 1 (low noise) vs Exp 2 (high noise)
    methods_short = ['IC-Knock-Poly-Val', 'Poly-Lasso-Val']
    exp_names = ['Exp 1: Low Noise\n($\\sigma \\in \\{0.1, 0.5\\}$)', 
                 'Exp 2: High Noise\n($\\sigma \\in \\{2.0, 3.0\\}$, $\\rho=0.8$)']
    
    # Plot 1: Test R2
    ax = axes[0]
    x = np.arange(len(exp_names))
    width = 0.35
    
    for i, method in enumerate(methods_short):
        values = []
        for df in [df1, df2]:
            mdata = df[df['method'] == method]['test_r2'].median()
            values.append(mdata)
        ax.bar(x + i*width, values, width, label=method.replace('-Val', ''), 
               color=colors[method], alpha=0.8)
    
    ax.set_ylabel('Test $R^2$ (median)', fontsize=12)
    ax.set_title('(a) Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(exp_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: FDR
    ax = axes[1]
    for i, method in enumerate(methods_short):
        values = []
        for df in [df1, df2]:
            mdata = df[df['method'] == method]['fdr'].mean()
            values.append(mdata)
        ax.bar(x + i*width, values, width, label=method.replace('-Val', ''),
               color=colors[method], alpha=0.8)
    
    ax.set_ylabel('FDR (mean)', fontsize=12)
    ax.set_title('(b) False Discovery Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(exp_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: TPR
    ax = axes[2]
    for i, method in enumerate(methods_short):
        values = []
        for df in [df1, df2]:
            mdata = df[df['method'] == method]['tpr'].mean()
            values.append(mdata)
        ax.bar(x + i*width, values, width, label=method.replace('-Val', ''),
               color=colors[method], alpha=0.8)
    
    ax.set_ylabel('TPR (mean)', fontsize=12)
    ax.set_title('(c) True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(exp_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Performance Comparison: Standard vs. Challenging Conditions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print('✓ Generated fig_noise_comparison.pdf')

print(f'\\nAll PDF figures saved to: {output_dir}/')
print('Files:')
print('  - fig_p_scaling.pdf')
print('  - fig_n_scaling.pdf')
print('  - fig_noise_comparison.pdf')

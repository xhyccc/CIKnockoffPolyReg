#!/usr/bin/env python3
"""Visualize large-scale experiment results."""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Find the most recent results file
results_dir = Path(__file__).parent / 'large_scale_results'
json_files = sorted(results_dir.glob('large_scale_experiments_*.json'))
if not json_files:
    print("No results files found!")
    sys.exit(1)

latest_file = json_files[-1]
print(f"Loading: {latest_file}")

with open(latest_file) as f:
    data = json.load(f)

results = data['results']
df = pd.DataFrame(results)

# Create output directory for figures
fig_dir = results_dir / 'figures'
fig_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150

# Extract config parameters
df['n'] = df['config'].apply(lambda x: x['n'])
df['p'] = df['config'].apply(lambda x: x['p'])
df['k'] = df['config'].apply(lambda x: x['k'])
df['degree'] = df['config'].apply(lambda x: x['degree'])
df['noise'] = df['config'].apply(lambda x: x['noise_std'])
df['dict_size'] = df['config'].apply(lambda x: x['dict_size'])

# Colors for methods
method_colors = {
    'IC-Knock-Poly': '#1f77b4',
    'Poly-Knockoff': '#ff7f0e', 
    'Poly-CLIME': '#2ca02c',
    'Poly-Lasso': '#d62728',
    'Poly-OMP': '#9467bd'
}

print(f"\nTotal experiments: {len(df)}")
print(f"Methods: {df['method'].unique()}")
print(f"Configurations: {len(df.groupby(['n', 'p', 'k', 'degree', 'noise']))}")

# Figure 1: Overall FDR vs TPR
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# FDR comparison
ax = axes[0]
method_stats = df.groupby('method').agg({
    'fdr': ['mean', 'std'],
    'tpr': ['mean', 'std']
}).reset_index()
method_stats.columns = ['method', 'fdr_mean', 'fdr_std', 'tpr_mean', 'tpr_std']
method_stats = method_stats.sort_values('fdr_mean')

bars = ax.barh(method_stats['method'], method_stats['fdr_mean'], 
               xerr=method_stats['fdr_std'], capsize=5,
               color=[method_colors[m] for m in method_stats['method']],
               alpha=0.8, edgecolor='black', linewidth=1)
ax.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Target FDR=0.2')
ax.set_xlabel('False Discovery Rate (FDR)', fontsize=12)
ax.set_title('Average FDR by Method', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# TPR comparison
ax = axes[1]
method_stats = method_stats.sort_values('tpr_mean', ascending=False)
bars = ax.barh(method_stats['method'], method_stats['tpr_mean'],
               xerr=method_stats['tpr_std'], capsize=5,
               color=[method_colors[m] for m in method_stats['method']],
               alpha=0.8, edgecolor='black', linewidth=1)
ax.set_xlabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('Average TPR by Method', fontsize=13, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / '01_overall_performance.png', bbox_inches='tight', dpi=300)
plt.savefig(fig_dir / '01_overall_performance.pdf', bbox_inches='tight')
print("Saved: 01_overall_performance.png")

# Figure 2: Performance vs Noise Level
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for method in df['method'].unique():
    subset = df[df['method'] == method]
    noise_stats = subset.groupby('noise').agg({
        'fdr': 'mean',
        'tpr': 'mean'
    }).reset_index()
    
    axes[0].plot(noise_stats['noise'], noise_stats['fdr'], 
                marker='o', linewidth=2, markersize=8,
                color=method_colors[method], label=method)
    axes[1].plot(noise_stats['noise'], noise_stats['tpr'],
                marker='o', linewidth=2, markersize=8,
                color=method_colors[method], label=method)

axes[0].axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Noise Level (σ)', fontsize=12)
axes[0].set_ylabel('FDR', fontsize=12)
axes[0].set_title('FDR vs Noise Level', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Noise Level (σ)', fontsize=12)
axes[1].set_ylabel('TPR', fontsize=12)
axes[1].set_title('TPR vs Noise Level', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / '02_noise_sensitivity.png', bbox_inches='tight', dpi=300)
plt.savefig(fig_dir / '02_noise_sensitivity.pdf', bbox_inches='tight')
print("Saved: 02_noise_sensitivity.png")

# Figure 3: Performance by Polynomial Degree
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

degree_stats = df.groupby(['method', 'degree']).agg({
    'fdr': 'mean',
    'tpr': 'mean'
}).reset_index()

for method in df['method'].unique():
    subset = degree_stats[degree_stats['method'] == method]
    axes[0].plot(subset['degree'], subset['fdr'],
                marker='o', linewidth=2, markersize=10,
                color=method_colors[method], label=method)
    axes[1].plot(subset['degree'], subset['tpr'],
                marker='o', linewidth=2, markersize=10,
                color=method_colors[method], label=method)

axes[0].axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Polynomial Degree', fontsize=12)
axes[0].set_ylabel('FDR', fontsize=12)
axes[0].set_title('FDR vs Polynomial Degree', fontsize=13, fontweight='bold')
axes[0].set_xticks([2, 3])
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Polynomial Degree', fontsize=12)
axes[1].set_ylabel('TPR', fontsize=12)
axes[1].set_title('TPR vs Polynomial Degree', fontsize=13, fontweight='bold')
axes[1].set_xticks([2, 3])
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / '03_degree_comparison.png', bbox_inches='tight', dpi=300)
plt.savefig(fig_dir / '03_degree_comparison.pdf', bbox_inches='tight')
print("Saved: 03_degree_comparison.png")

# Figure 4: FDR-TPR Tradeoff
fig, ax = plt.subplots(figsize=(10, 8))

for method in df['method'].unique():
    subset = df[df['method'] == method]
    ax.scatter(subset['fdr'], subset['tpr'], 
              alpha=0.4, s=50,
              color=method_colors[method], label=method)
    
    # Add mean point
    mean_fdr = subset['fdr'].mean()
    mean_tpr = subset['tpr'].mean()
    ax.scatter(mean_fdr, mean_tpr, 
              s=200, marker='*', 
              color=method_colors[method],
              edgecolor='black', linewidth=2,
              zorder=10)

ax.axvline(x=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target FDR=0.2')
ax.set_xlabel('False Discovery Rate (FDR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('FDR-TPR Tradeoff by Method', fontsize=13, fontweight='bold')
ax.set_xlim(-0.05, 1)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / '04_fdr_tpr_tradeoff.png', bbox_inches='tight', dpi=300)
plt.savefig(fig_dir / '04_fdr_tpr_tradeoff.pdf', bbox_inches='tight')
print("Saved: 04_fdr_tpr_tradeoff.png")

# Figure 5: Computational Efficiency
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

time_stats = df.groupby('method')['time_seconds'].mean().sort_values()
axes[0].barh(time_stats.index, time_stats.values,
            color=[method_colors[m] for m in time_stats.index],
            alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Average Time (seconds)', fontsize=12)
axes[0].set_title('Computational Time by Method', fontsize=13, fontweight='bold')
axes[0].set_xscale('log')
axes[0].grid(axis='x', alpha=0.3)

# Add time labels
for i, (method, time) in enumerate(time_stats.items()):
    axes[0].text(time * 1.1, i, f'{time:.2f}s', va='center', fontsize=9)

memory_stats = df.groupby('method')['peak_memory_mb'].mean().sort_values()
axes[1].barh(memory_stats.index, memory_stats.values,
            color=[method_colors[m] for m in memory_stats.index],
            alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Average Peak Memory (MB)', fontsize=12)
axes[1].set_title('Memory Usage by Method', fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / '05_computational_efficiency.png', bbox_inches='tight', dpi=300)
plt.savefig(fig_dir / '05_computational_efficiency.pdf', bbox_inches='tight')
print("Saved: 05_computational_efficiency.png")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
summary = df.groupby('method').agg({
    'fdr': ['mean', 'std', 'median'],
    'tpr': ['mean', 'std', 'median'],
    'time_seconds': ['mean', 'std'],
    'peak_memory_mb': 'mean'
}).round(3)

print(summary)

print(f"\nFigures saved to: {fig_dir}")
print("Done!")

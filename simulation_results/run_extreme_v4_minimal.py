#!/usr/bin/env python3
"""Experiment V4: EXTREME CHALLENGES - Minimal but Hard

Only extreme configurations:
- n=50, p=9 (small sample, high dim)
- n=100, p=5 (moderate)
- noise=8.0 (highest)
- k=5 (many true terms)
- degree=3 (high order)
- 3 trials each

Complex label noise + extreme correlation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import json
import time
from datetime import datetime

from simulations.data_generator_v2 import generate_simulation_v2
from ic_knockoff_poly_reg import ICKnockoffPolyReg
from ic_knockoff_poly_reg.evaluation import normalize_polynomial_term
from baselines.poly_knockoff import PolyKnockoff
from baselines.poly_clime import PolyCLIME
from baselines.poly_lasso import PolyLasso
from baselines.poly_omp import PolyOMP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'large_scale_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(OUTPUT_DIR, f'final_val_extreme_v4_minimal_{TIMESTAMP}.json')
N_TRIALS = 3
N_VAL = 100

# EXTREME parameters
NOISE_STD = 8.0  # Only highest
INTRA_BLOCK_CORR = 0.99
INTER_BLOCK_CORR = 0.3
OUTLIER_RATIO = 0.10
NONLINEAR_NOISE_EXPONENT = 1.5


def add_complex_noise(y_true, X, noise_std, seed=None):
    """Add extreme complex noise."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(y_true)
    noise = np.random.normal(0, noise_std, n)
    
    # Non-linear heteroscedasticity
    x_norms = np.linalg.norm(X, axis=1)
    hetero_factors = (x_norms / np.mean(x_norms)) ** NONLINEAR_NOISE_EXPONENT
    noise = noise * hetero_factors
    
    # Outliers
    n_outliers = int(n * OUTLIER_RATIO)
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    noise[outlier_idx] = np.random.standard_t(df=2, size=n_outliers) * noise_std * 3
    
    # Systematic bias
    bias = 0.3 * noise_std * np.sign(X[:, 0]) * np.abs(X[:, 0]) ** 0.5
    
    return y_true + noise + bias


def generate_extreme_data(n_labeled, p, k, degree, random_state=None):
    """Generate extreme data."""
    dataset = generate_simulation_v2(
        n_labeled=n_labeled, p=p, k=k, degree=degree,
        noise_std=1.0,
        correlation=INTRA_BLOCK_CORR,
        n_test=200,
        random_state=random_state
    )
    
    # Replace with complex noise
    dataset.y = add_complex_noise(dataset.y, dataset.X, NOISE_STD, random_state)
    dataset.y_test = add_complex_noise(dataset.y_test, dataset.X_test, NOISE_STD, 
                                       random_state + 1 if random_state else None)
    
    return dataset


def compute_metrics(y_true, y_pred):
    ss = np.sum((y_true - y_pred) ** 2)
    st = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss/st if st > 0 else 0
    rmse = np.sqrt(ss / len(y_true))
    return float(r2), float(rmse)


def run(model, dataset):
    """Run model and evaluate."""
    r = {'success': False}
    try:
        model.fit(dataset.X, dataset.y)
        y_pred = model.predict(dataset.X_test)
        test_r2, test_rmse = compute_metrics(dataset.y_test, y_pred)
        
        # Selection
        sel = []
        if hasattr(model, 'result_') and model.result_:
            sel = model.result_.selected_terms
        elif hasattr(model, '_sel_terms'):
            sel = model._sel_terms
        
        # FDR/TPR
        ss_set = set(normalize_polynomial_term(t) for t in sel)
        ts_set = set(normalize_polynomial_term(t) for t in dataset.true_poly_terms)
        tp = len(ss_set & ts_set)
        fp = len(ss_set - ts_set)
        
        r.update({
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'n_selected': len(sel),
            'fdr': fp / len(ss_set) if ss_set else 0.0,
            'tpr': tp / len(ts_set) if ts_set else 0.0,
            'selected_terms': [list(t) for t in sel],
            'true_terms': [list(t) for t in dataset.true_poly_terms],
            'success': True
        })
    except Exception as e:
        r['error'] = str(e)
    return r


def main():
    print("="*80)
    print("EXTREME V4 - MINIMAL BUT HARD")
    print("="*80)
    print(f"Noise: {NOISE_STD}, Correlation: {INTRA_BLOCK_CORR}")
    print(f"Complex: non-linear hetero + 10% outliers + systematic bias")
    print("="*80)
    
    # Minimal extreme configs
    configs = [
        {'n': 50, 'p': 9, 'k': 5, 'degree': 3},   # Extreme: small n, high p
        {'n': 100, 'p': 5, 'k': 5, 'degree': 3},  # Moderate extreme
    ]
    
    results = []
    
    for cfg in configs:
        n, p, k, d = cfg['n'], cfg['p'], cfg['k'], cfg['degree']
        print(f"\n{'='*80}")
        print(f"Config: n={n}, p={p}, k={k}, degree={d}, noise={NOISE_STD}")
        print(f"{'='*80}")
        
        for trial in range(N_TRIALS):
            print(f"\nTrial {trial}:")
            rs = trial + n * 1000 + p * 100
            
            # Generate data
            dataset = generate_extreme_data(n, p, k, d, rs)
            print(f"  Data: X={dataset.X.shape}, true_terms={len(dataset.true_poly_terms)}")
            
            # IC-Knock-Poly-Val
            print("  IC-Knock-Poly-Val...", end=' ')
            try:
                # Test multiple Q values
                best_q, best_val_r2 = 0.05, -float('inf')
                for q in [0.05, 0.10, 0.15]:
                    m = ICKnockoffPolyReg(degree=d, Q=q, random_state=rs, backend="rust")
                    m.fit(dataset.X, dataset.y)
                    val_idx = np.random.choice(len(dataset.X), min(50, len(dataset.X)), replace=False)
                    val_r2 = compute_metrics(dataset.y[val_idx], m.predict(dataset.X[val_idx]))[0]
                    if val_r2 > best_val_r2:
                        best_val_r2, best_q = val_r2, q
                
                m = ICKnockoffPolyReg(degree=d, Q=best_q, random_state=rs, backend="rust")
                r = run(m, dataset)
                r.update({'method': 'IC-Knock-Poly-Val', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}, 'best_Q': best_q})
                results.append(r)
                print(f"R²={r['test_r2']:.3f}, TPR={r['tpr']:.3f}, FDR={r['fdr']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Poly-Knockoff-Val
            print("  Poly-Knockoff-Val...", end=' ')
            try:
                m = PolyKnockoff(degree=d, Q=0.10, random_state=rs)
                r = run(m, dataset)
                r.update({'method': 'Poly-Knockoff-Val', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}})
                results.append(r)
                print(f"R²={r['test_r2']:.3f}, TPR={r['tpr']:.3f}, FDR={r['fdr']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Poly-Lasso
            print("  Poly-Lasso...", end=' ')
            try:
                m = PolyLasso(degree=d, alpha=0.1, max_iter=5000)
                r = run(m, dataset)
                r.update({'method': 'Poly-Lasso', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}})
                results.append(r)
                print(f"R²={r['test_r2']:.3f}, TPR={r['tpr']:.3f}, FDR={r['fdr']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Done! {len(results)} results saved to {RESULTS_FILE}")
    print("="*80)


if __name__ == '__main__':
    main()

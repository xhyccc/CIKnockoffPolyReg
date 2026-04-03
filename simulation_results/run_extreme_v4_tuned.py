#!/usr/bin/env python3
"""Experiment V4: EXTREME CHALLENGES - TUNED BASELINES

Tune regularization for all baselines:
- Lasso: alpha ∈ [0.001, 0.01, 0.1, 1.0, 10.0]
- CLIME: alpha ∈ [0.1, 0.5, 1.0, 2.0]
- OMP: k ∈ [1, 2, 3, 5, 10]
- STLSQ: threshold ∈ [0.01, 0.05, 0.1, 0.2]

Use validation set to select best hyperparameter.
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
from baselines.improved_stlsq import ImprovedSparsePolySTLSQ

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'large_scale_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(OUTPUT_DIR, f'final_val_extreme_v4_tuned_{TIMESTAMP}.json')
N_TRIALS = 3
N_VAL = 100

# EXTREME parameters
NOISE_STD = 8.0
CORRELATION = 0.99
OUTLIER_RATIO = 0.10
NONLINEAR_NOISE_EXPONENT = 1.5

# Hyperparameter grids
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
CLIME_ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
OMP_KS = [1, 2, 3, 5, 7, 10]
STLSQ_THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.2]


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
    if n_outliers > 0:
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
        correlation=CORRELATION,
        n_test=200,
        random_state=random_state
    )
    
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


def evaluate_model(model, dataset):
    """Evaluate model."""
    y_pred = model.predict(dataset.X_test)
    test_r2, test_rmse = compute_metrics(dataset.y_test, y_pred)
    
    # Get selected terms
    sel = []
    if hasattr(model, 'result_') and model.result_:
        sel = model.result_.selected_terms
    elif hasattr(model, '_sel_terms'):
        sel = model._sel_terms
    elif hasattr(model, 'selected_indices'):
        indices = model.selected_indices
        for idx in indices:
            base_idx = model._base_feature_indices[idx]
            exp = model._power_exponents[idx]
            interaction = model._interaction_indices[idx] if idx < len(model._interaction_indices) else None
            if interaction:
                sel.append([base_idx, exp, interaction])
            else:
                sel.append([base_idx, exp])
    
    # Compute FDR/TPR
    ss_set = set(normalize_polynomial_term(t) for t in sel)
    ts_set = set(normalize_polynomial_term(t) for t in dataset.true_poly_terms)
    tp = len(ss_set & ts_set)
    fp = len(ss_set - ts_set)
    
    return {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'n_selected': len(sel),
        'fdr': fp / len(ss_set) if ss_set else 0.0,
        'tpr': tp / len(ts_set) if ts_set else 0.0,
        'selected_terms': [list(t) for t in sel],
        'true_terms': [list(t) for t in dataset.true_poly_terms]
    }


def tune_and_fit(ModelClass, X_train, y_train, X_val, y_val, param_grid, fixed_params):
    """Tune hyperparameter using validation set."""
    best_param = None
    best_val_r2 = -float('inf')
    best_model = None
    
    for param_val in param_grid:
        try:
            kwargs = {**fixed_params, 'random_state': 42}
            # Determine parameter name based on model
            if ModelClass == PolyLasso:
                kwargs['alpha'] = param_val
            elif ModelClass == PolyCLIME:
                kwargs['alpha'] = param_val
            elif ModelClass == PolyOMP:
                kwargs['n_nonzero_coefs'] = param_val
            elif ModelClass == ImprovedSparsePolySTLSQ:
                kwargs['relative_threshold'] = param_val
            
            model = ModelClass(**kwargs)
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            val_r2 = compute_metrics(y_val, y_pred_val)[0]
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_param = param_val
                best_model = model
                
        except Exception as e:
            continue
    
    return best_model, best_param, best_val_r2


def main():
    print("="*80)
    print("EXTREME V4 - TUNED BASELINES")
    print("="*80)
    print(f"Noise: {NOISE_STD}, Correlation: {CORRELATION}")
    print("Complex: non-linear hetero + 10% outliers + systematic bias")
    print("\nTuning grids:")
    print(f"  Lasso alpha: {LASSO_ALPHAS}")
    print(f"  CLIME alpha: {CLIME_ALPHAS}")
    print(f"  OMP k: {OMP_KS}")
    print(f"  STLSQ threshold: {STLSQ_THRESHOLDS}")
    print("="*80)
    
    configs = [
        {'n': 50, 'p': 9, 'k': 5, 'degree': 3},
        {'n': 100, 'p': 5, 'k': 5, 'degree': 3},
    ]
    
    results = []
    total_start = time.time()
    
    for cfg in configs:
        n, p, k, d = cfg['n'], cfg['p'], cfg['k'], cfg['degree']
        print(f"\n{'='*80}")
        print(f"Config: n={n}, p={p}, k={k}, degree={d}, noise={NOISE_STD}")
        print(f"{'='*80}")
        
        for trial in range(N_TRIALS):
            print(f"\n  Trial {trial}:")
            rs = trial + n * 1000 + p * 100
            
            # Generate data
            dataset = generate_extreme_data(n, p, k, d, rs)
            
            # Split train/val
            val_size = min(N_VAL, len(dataset.X) // 3)
            val_idx = np.random.choice(len(dataset.X), val_size, replace=False)
            train_idx = np.setdiff1d(np.arange(len(dataset.X)), val_idx)
            
            X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
            X_val, y_val = dataset.X[val_idx], dataset.y[val_idx]
            
            print(f"    Train: {X_train.shape}, Val: {X_val.shape}")
            
            # IC-Knock-Poly-Val
            print("    IC-Knock-Poly-Val...", end=' ', flush=True)
            try:
                best_q, best_val_r2 = 0.05, -float('inf')
                for q in [0.05, 0.10, 0.15]:
                    m = ICKnockoffPolyReg(degree=d, Q=q, random_state=rs, backend="rust")
                    m.fit(X_train, y_train)
                    val_r2 = compute_metrics(y_val, m.predict(X_val))[0]
                    if val_r2 > best_val_r2:
                        best_val_r2, best_q = val_r2, q
                
                m = ICKnockoffPolyReg(degree=d, Q=best_q, random_state=rs, backend="rust")
                m.fit(dataset.X, dataset.y)
                r = evaluate_model(m, dataset)
                r.update({'method': 'IC-Knock-Poly-Val', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}, 'best_Q': best_q, 'best_val_r2': best_val_r2})
                results.append(r)
                print(f"Q={best_q}, R²={r['test_r2']:.3f}, TPR={r['tpr']:.2f}, FDR={r['fdr']:.2f}, n_sel={r['n_selected']}")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Poly-Knockoff-Val
            print("    Poly-Knockoff-Val...", end=' ', flush=True)
            try:
                m = PolyKnockoff(degree=d, Q=0.10, random_state=rs)
                m.fit(dataset.X, dataset.y)
                r = evaluate_model(m, dataset)
                r.update({'method': 'Poly-Knockoff-Val', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}})
                results.append(r)
                print(f"R²={r['test_r2']:.3f}, TPR={r['tpr']:.2f}, FDR={r['fdr']:.2f}, n_sel={r['n_selected']}")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Poly-Lasso (TUNED)
            print("    Poly-Lasso (tuned)...", end=' ', flush=True)
            try:
                m, best_alpha, val_r2 = tune_and_fit(
                    PolyLasso, X_train, y_train, X_val, y_val,
                    LASSO_ALPHAS, {'degree': d, 'max_iter': 10000}
                )
                if m:
                    r = evaluate_model(m, dataset)
                    r.update({'method': 'Poly-Lasso-Tuned', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}, 'best_alpha': best_alpha, 'best_val_r2': val_r2})
                    results.append(r)
                    print(f"α={best_alpha}, R²={r['test_r2']:.3f}, TPR={r['tpr']:.2f}, FDR={r['fdr']:.2f}, n_sel={r['n_selected']}")
                else:
                    print("All alphas failed")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Poly-CLIME (TUNED)
            print("    Poly-CLIME (tuned)...", end=' ', flush=True)
            try:
                m, best_alpha, val_r2 = tune_and_fit(
                    PolyCLIME, X_train, y_train, X_val, y_val,
                    CLIME_ALPHAS, {'degree': d}
                )
                if m:
                    r = evaluate_model(m, dataset)
                    r.update({'method': 'Poly-CLIME-Tuned', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}, 'best_alpha': best_alpha, 'best_val_r2': val_r2})
                    results.append(r)
                    print(f"α={best_alpha}, R²={r['test_r2']:.3f}, TPR={r['tpr']:.2f}, FDR={r['fdr']:.2f}, n_sel={r['n_selected']}")
                else:
                    print("All alphas failed")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Poly-OMP (TUNED)
            print("    Poly-OMP (tuned)...", end=' ', flush=True)
            try:
                m, best_k, val_r2 = tune_and_fit(
                    PolyOMP, X_train, y_train, X_val, y_val,
                    OMP_KS, {'degree': d}
                )
                if m:
                    r = evaluate_model(m, dataset)
                    r.update({'method': 'Poly-OMP-Tuned', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}, 'best_k': best_k, 'best_val_r2': val_r2})
                    results.append(r)
                    print(f"k={best_k}, R²={r['test_r2']:.3f}, TPR={r['tpr']:.2f}, FDR={r['fdr']:.2f}, n_sel={r['n_selected']}")
                else:
                    print("All k values failed")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # Poly-STLSQ (TUNED)
            print("    Poly-STLSQ (tuned)...", end=' ', flush=True)
            try:
                m, best_thresh, val_r2 = tune_and_fit(
                    ImprovedSparsePolySTLSQ, X_train, y_train, X_val, y_val,
                    STLSQ_THRESHOLDS, {'degree': d}
                )
                if m:
                    r = evaluate_model(m, dataset)
                    r.update({'method': 'Poly-STLSQ-Tuned', 'config': {**cfg, 'trial': trial, 'noise': NOISE_STD}, 'best_threshold': best_thresh, 'best_val_r2': val_r2})
                    results.append(r)
                    print(f"thresh={best_thresh}, R²={r['test_r2']:.3f}, TPR={r['tpr']:.2f}, FDR={r['fdr']:.2f}, n_sel={r['n_selected']}")
                else:
                    print("All thresholds failed")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"Done! {len(results)} results in {elapsed/60:.1f} minutes")
    print(f"Saved to: {RESULTS_FILE}")
    print("="*80)


if __name__ == '__main__':
    main()

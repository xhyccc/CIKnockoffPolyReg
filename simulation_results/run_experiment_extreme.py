#!/usr/bin/env python3
"""Experiment V3: EXTREME CHALLENGES

Ultimate robustness test combining:
- Block-diagonal correlation (intra-block ρ=0.95)
- Heavy-tailed features (Student-t with df=3)
- Heteroscedastic noise (variance depends on features)
- ULTRA-HIGH noise: σ ∈ {10.0, 15.0}

Parameters match previous experiments for fair comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import json
import time
import psutil
from datetime import datetime

# Import extreme data generator
from simulations.data_generator_v3 import generate_extreme_simulation

from ic_knockoff_poly_reg import ICKnockoffPolyReg
from ic_knockoff_poly_reg.evaluation import normalize_polynomial_term
from baselines.poly_knockoff import PolyKnockoff
from baselines.poly_clime import PolyCLIME
from baselines.poly_lasso import PolyLasso
from baselines.poly_omp import PolyOMP
from baselines.improved_stlsq import ImprovedSparsePolySTLSQ

print(f"Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")

BACKEND = "rust"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'large_scale_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(OUTPUT_DIR, f'final_val_extreme_{TIMESTAMP}.json')
N_TRIALS = 3
N_VAL = 100

# EXTREME parameters
NOISE_LEVELS = [10.0, 15.0]  # ULTRA-HIGH noise
N_BLOCKS = 2
INTRA_BLOCK_CORR = 0.95
HETERO_FACTOR = 0.5
HEAVY_TAIL_DF = 3


def compute_metrics(y_true, y_pred):
    """Compute R2 and RMSE."""
    ss = np.sum((y_true - y_pred) ** 2)
    st = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss/st if st > 0 else 0
    rmse = np.sqrt(ss / len(y_true))
    return float(r2), float(rmse)


def val_score(ModelClass, X_train, y_train, X_val, y_val, kwargs):
    """Fit on train, score on validation."""
    try:
        m = ModelClass(**kwargs)
        m.fit(X_train, y_train)
        p = m.predict(X_val)
        r2, rmse = compute_metrics(y_val, p)
        return m, r2, rmse
    except Exception as e:
        print(f"    val_score error: {e}")
        return None, -float('inf'), float('inf')


def run(model, dataset, X_train=None, y_train=None, X_val=None, y_val=None):
    """Run and return metrics."""
    r = {'success': False}
    try:
        t0 = time.time()
        Xt = X_train if X_train is not None else dataset.X
        yt = y_train if y_train is not None else dataset.y
        model.fit(Xt, yt)
        r['time'] = time.time() - t0
        return _evaluate_model(model, dataset, r, Xt, yt, X_val, y_val)
    except Exception as e:
        r['error'] = str(e)
        import traceback
        traceback.print_exc()
    return r


def run_with_model(model, dataset, X_train=None, y_train=None, X_val=None, y_val=None):
    """Evaluate already-fitted model."""
    r = {'success': False}
    try:
        return _evaluate_model(model, dataset, r, X_train, y_train, X_val, y_val)
    except Exception as e:
        r['error'] = str(e)
        import traceback
        traceback.print_exc()
    return r


def _to_json_serializable(obj):
    """Convert to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj


def _evaluate_model(model, dataset, r, X_train=None, y_train=None, X_val=None, y_val=None):
    """Evaluate on train, val, and test sets."""
    
    # Test set predictions
    y_pred_test = model.predict(dataset.X_test)
    yt = dataset.y_test_noisy if hasattr(dataset, 'y_test_noisy') and dataset.y_test_noisy is not None else dataset.y_test
    
    # Test metrics
    test_r2, test_rmse = compute_metrics(yt, y_pred_test)
    r['test_r2'] = test_r2
    r['test_rmse'] = test_rmse
    
    # Train metrics
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        train_r2, train_rmse = compute_metrics(y_train, y_pred_train)
        r['train_r2'] = train_r2
        r['train_rmse'] = train_rmse
    else:
        r['train_r2'] = None
        r['train_rmse'] = None
    
    # Validation metrics
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        val_r2, val_rmse = compute_metrics(y_val, y_pred_val)
        r['val_r2'] = val_r2
        r['val_rmse'] = val_rmse
    else:
        r['val_r2'] = None
        r['val_rmse'] = None
    
    # Terms selection
    sel = []
    if hasattr(model, 'result_') and model.result_:
        sel = model.result_.selected_terms
    elif hasattr(model, '_sel_terms'):
        raw_terms = model._sel_terms
        sel = []
        for t in raw_terms:
            if len(t) >= 2:
                base_idx = t[0]
                exp = t[1]
                interaction = t[2] if len(t) > 2 else []
                sel.append([base_idx, exp, interaction] if interaction else [base_idx, exp])
            else:
                sel.append(t)
    elif hasattr(model, 'selected_indices'):
        indices = model.selected_indices
        sel = []
        for idx in indices:
            base_idx = model._base_feature_indices[idx]
            exp = model._power_exponents[idx]
            if hasattr(model, '_interaction_indices') and idx < len(model._interaction_indices):
                interaction = model._interaction_indices[idx]
                if interaction is not None and len(interaction) > 0:
                    sel.append([base_idx, exp, interaction])
                else:
                    sel.append([base_idx, exp])
            else:
                sel.append([base_idx, exp])
    
    # Compute FDR/TPR
    ss_set = set(normalize_polynomial_term(t) for t in sel)
    ts_set = set(normalize_polynomial_term(t) for t in dataset.true_poly_terms)
    tp = len(ss_set & ts_set)
    fp = len(ss_set - ts_set)
    
    r['n_selected'] = len(sel)
    r['fdr'] = fp / len(ss_set) if ss_set else 0.0
    r['tpr'] = tp / len(ts_set) if ts_set else 0.0
    r['selected_terms'] = _to_json_serializable(sel)
    r['true_terms'] = _to_json_serializable(dataset.true_poly_terms)
    r['success'] = True
    
    return r


def main():
    print("="*80)
    print("EXPERIMENT V3 - EXTREME CHALLENGES")
    print("="*80)
    print("Challenges:")
    print("  - Block-diagonal correlation (intra-block ρ=0.95)")
    print("  - Heavy-tailed features (Student-t, df=3)")
    print("  - Heteroscedastic noise (variance depends on x)")
    print("  - ULTRA-HIGH noise: σ ∈ {10.0, 15.0}")
    print("="*80)
    
    # Build configs (same as before but with extreme noise)
    configs = []
    for n in [50, 75, 100]:
        for p in [2, 4, 6, 8, 10]:  # Even numbers for block structure
            for degree in [2, 3]:
                max_k = 2 * degree * p
                for k in [2, 3, 5]:
                    if k > max_k:
                        continue
                    for noise in NOISE_LEVELS:
                        for trial in range(N_TRIALS):
                            configs.append({
                                'n': n, 'p': p, 'k': k,
                                'degree': degree, 'noise': noise,
                                'trial': trial,
                                'n_blocks': N_BLOCKS,
                                'intra_block_corr': INTRA_BLOCK_CORR,
                            })
    
    total = len(configs) * 12
    print(f"Total configurations: {len(configs)}")
    print(f"Total experiments: {total}")
    print(f"Expected time: ~{total * 5 / 3600:.1f} hours")
    print("="*80)
    
    results = []
    completed = 0
    start = time.time()
    
    for ci in configs:
        n, p, k, d, noise, trial = ci['n'], ci['p'], ci['k'], ci['degree'], ci['noise'], ci['trial']
        rs = trial
        
        print(f"\n{'='*80}")
        print(f"Config: n={n}, p={p}, k={k}, degree={d}, noise={noise}, trial={trial}")
        print(f"{'='*80}")
        
        # Generate EXTREME data
        try:
            dataset = generate_extreme_simulation(
                n_labeled=n, p=p, k=k,
                degree=d,
                noise_std=noise,
                n_blocks=N_BLOCKS,
                intra_block_corr=INTRA_BLOCK_CORR,
                hetero_factor=HETERO_FACTOR,
                heavy_tail_df=HEAVY_TAIL_DF,
                n_unlabeled=0,  # Will generate separate val set
                n_test=200,
                random_state=rs + n * 1000 + p * 100 + k * 10 + d * 10000 + int(noise * 100)
            )
        except Exception as e:
            print(f"  ERROR generating data: {e}")
            continue
        
        X_train, y_train = dataset.X, dataset.y
        
        # Generate validation set separately
        try:
            val_dataset = generate_extreme_simulation(
                n_labeled=N_VAL, p=p, k=k,
                degree=d,
                noise_std=noise,
                n_blocks=N_BLOCKS,
                intra_block_corr=INTRA_BLOCK_CORR,
                hetero_factor=HETERO_FACTOR,
                heavy_tail_df=HEAVY_TAIL_DF,
                n_unlabeled=0, n_test=0,
                random_state=rs + 50000
            )
            X_val, y_val = val_dataset.X, val_dataset.y
        except Exception as e:
            print(f"  ERROR generating validation: {e}")
            continue
        
        # IC-Knock-Poly
        print(f"  IC-Knock-Poly: fitting Q=0.05, 0.10, 0.15...")
        ic_models = {}
        for q in [0.05, 0.10, 0.15]:
            print(f"    Fitting Q={q}...")
            try:
                m = ICKnockoffPolyReg(degree=d, Q=q, random_state=rs, backend=BACKEND)
                r = run(m, dataset, X_train, y_train, X_val, y_val)
                ic_models[q] = m
                print(f"      Test R²={r.get('test_r2', 0):.3f}")
            except Exception as e:
                print(f"      ERROR: {e}")
        
        # IC-Knock-Poly-Val
        if ic_models:
            print(f"  IC-Knock-Poly-Val: selecting best Q...")
            best_m, best_q, best_val_r2 = None, None, -float('inf')
            for q, m in ic_models.items():
                try:
                    y_pred_val = m.predict(X_val)
                    val_r2, val_rmse = compute_metrics(y_val, y_pred_val)
                    print(f"    Q={q}: val_r2={val_r2:.3f}")
                    if val_r2 > best_val_r2:
                        best_val_r2, best_q, best_m = val_r2, q, m
                except Exception as e:
                    print(f"    ERROR val Q={q}: {e}")
            
            if best_m:
                r = run_with_model(best_m, dataset, X_train, y_train, X_val, y_val)
                r.update({'method': 'IC-Knock-Poly-Val', 'config': ci, 'best_Q': best_q})
                results.append(r)
                print(f"    Selected Q={best_q}, Test R²={r.get('test_r2', 0):.3f}")
        
        # Poly-Lasso
        print(f"  Poly-Lasso-Val: selecting best alpha...")
        best_m, best_a, best_val_r2 = None, None, -float('inf')
        for a in [0.001, 0.01, 0.1]:
            m, val_r2, val_rmse = val_score(PolyLasso, X_train, y_train, X_val, y_val,
                         {'degree': d, 'alpha': a})
            if val_r2 > best_val_r2:
                best_val_r2, best_a, best_m = val_r2, a, m
        if best_m:
            r = run_with_model(best_m, dataset, X_train, y_train, X_val, y_val)
            r.update({'method': 'Poly-Lasso-Val', 'config': ci, 'best_alpha': best_a})
            results.append(r)
            print(f"    Best alpha={best_a}, Test R²={r.get('test_r2', 0):.3f}")
        
        # Poly-OMP
        print(f"  Poly-OMP-Val: selecting best k...")
        best_m, best_k_param, best_val_r2 = None, None, -float('inf')
        for nk in [k-1, k, k+1, k+2]:
            if nk < 1:
                continue
            m, val_r2, val_rmse = val_score(PolyOMP, X_train, y_train, X_val, y_val,
                         {'degree': d, 'n_nonzero_coefs': nk})
            if val_r2 > best_val_r2:
                best_val_r2, best_k_param, best_m = val_r2, nk, m
        if best_m:
            r = run_with_model(best_m, dataset, X_train, y_train, X_val, y_val)
            r.update({'method': 'Poly-OMP-Val', 'config': ci, 'best_k': best_k_param})
            results.append(r)
            print(f"    Best k={best_k_param}, Test R²={r.get('test_r2', 0):.3f}")
        
        # Poly-CLIME
        print(f"  Poly-CLIME-Val: selecting best alpha...")
        best_m, best_a, best_val_r2 = None, None, -float('inf')
        for a in [0.005, 0.05, 0.5]:
            m, val_r2, val_rmse = val_score(PolyCLIME, X_train, y_train, X_val, y_val,
                         {'degree': d, 'alpha': a, 'Q': 0.10})
            if val_r2 > best_val_r2:
                best_val_r2, best_a, best_m = val_r2, a, m
        if best_m:
            r = run_with_model(best_m, dataset, X_train, y_train, X_val, y_val)
            r.update({'method': 'Poly-CLIME-Val', 'config': ci, 'best_alpha': best_a})
            results.append(r)
            print(f"    Best alpha={best_a}, Test R²={r.get('test_r2', 0):.3f}")
        
        # Improved Poly-STLSQ
        print(f"  Poly-STLSQ-Val (Improved with Ridge)...")
        best_m, best_t, best_val_r2 = None, None, -float('inf')
        for t in [0.05, 0.1, 0.2]:
            m, val_r2, val_rmse = val_score(ImprovedSparsePolySTLSQ, X_train, y_train, X_val, y_val,
                         {'degree': d, 'threshold': t, 'threshold_mode': 'relative', 
                          'alpha': 0.01, 'max_iter': 10})
            print(f"    Relative threshold={t}: val_r2={val_r2:.3f}")
            if val_r2 > best_val_r2:
                best_val_r2, best_t, best_m = val_r2, t, m
        if best_m:
            r = run_with_model(best_m, dataset, X_train, y_train, X_val, y_val)
            r.update({'method': 'Poly-STLSQ-Val', 'config': ci, 'best_threshold': best_t})
            results.append(r)
            print(f"    Best threshold={best_t}, Test R²={r.get('test_r2', 0):.3f}")
        
        # Poly-Knockoff
        print(f"  Poly-Knockoff: fitting Q=0.05, 0.10, 0.15...")
        pk_models = {}
        for q in [0.05, 0.10, 0.15]:
            print(f"    Fitting Q={q}...")
            try:
                m = PolyKnockoff(degree=d, Q=q, random_state=rs)
                r = run(m, dataset, X_train, y_train, X_val, y_val)
                results.append(r)
                pk_models[q] = m
                print(f"      Test R²={r.get('test_r2', 0):.3f}")
            except Exception as e:
                print(f"      ERROR: {e}")
        
        # Poly-Knockoff-Val
        if pk_models:
            print(f"  Poly-Knockoff-Val: selecting best Q...")
            best_m, best_q, best_val_r2 = None, None, -float('inf')
            for q, m in pk_models.items():
                try:
                    y_pred_val = m.predict(X_val)
                    val_r2, val_rmse = compute_metrics(y_val, y_pred_val)
                    print(f"    Q={q}: val_r2={val_r2:.3f}")
                    if val_r2 > best_val_r2:
                        best_val_r2, best_q, best_m = val_r2, q, m
                except Exception as e:
                    print(f"    ERROR val Q={q}: {e}")
            
            if best_m:
                r = run_with_model(best_m, dataset, X_train, y_train, X_val, y_val)
                r.update({'method': 'Poly-Knockoff-Val', 'config': ci, 'best_Q': best_q})
                results.append(r)
                print(f"    Selected Q={best_q}, Test R²={r.get('test_r2', 0):.3f}")
        
        completed += 1
        if completed % 5 == 0:
            with open(RESULTS_FILE, 'w') as f:
                json.dump({'results': results, 'completed': completed, 'total': total}, f)
            el = time.time() - start
            rem = (el / completed) * (total - completed)
            print(f"\n>>> Progress: {completed}/{total} ({100*completed/total:.1f}%) - {rem/60:.0f}min left\n")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'results': results, 'completed': completed, 'total': total}, f)
    
    print(f"\nDONE! {len(results)} experiments -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()

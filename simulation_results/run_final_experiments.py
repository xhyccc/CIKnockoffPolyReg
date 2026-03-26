#!/usr/bin/env python3
"""Final Experiment: Validation Set (100 samples) instead of CV

Training: n samples
Validation: 100 samples (for hyperparameter selection)
Test: 200 samples

Methods:
- IC-Knock-Poly: Q=0.05, 0.10, 0.15 (fixed) + CV from 3 Q values
- Poly-Lasso: select from 3 alphas
- Poly-OMP: select from 3 k values  
- Poly-CLIME: select from 3 alphas
- Poly-Knockoff: select from 3 Q values
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
from simulations.data_generator import generate_simulation
from ic_knockoff_poly_reg import ICKnockoffPolyReg
from baselines.poly_knockoff import PolyKnockoff
from baselines.poly_clime import PolyCLIME
from baselines.poly_lasso import PolyLasso
from baselines.poly_omp import PolyOMP

print(f"Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")

from ic_knockoff_poly_reg.rust_gmm import _get_rust_lib
print(f"Rust: {_get_rust_lib()._name}")

BACKEND = "rust"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'large_scale_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(OUTPUT_DIR, f'final_val_{TIMESTAMP}.json')
N_TRIALS = 3
N_VAL = 100


def val_score(ModelClass, X_train, y_train, X_val, y_val, kwargs):
    """Fit on train, score on validation. Returns (model, r2_score)."""
    try:
        m = ModelClass(**kwargs)
        m.fit(X_train, y_train)
        p = m.predict(X_val)
        ss = np.sum((y_val - p) ** 2)
        st = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - ss/st if st > 0 else 0
        return m, r2
    except:
        return None, -float('inf')


def run(model, dataset, X_train=None, y_train=None):
    """Run and return metrics (fit then evaluate)."""
    r = {'success': False}
    try:
        t0 = time.time()
        # Use provided train data or full dataset
        Xt = X_train if X_train is not None else dataset.X
        yt = y_train if y_train is not None else dataset.y
        model.fit(Xt, yt)
        r['time'] = time.time() - t0
        return _evaluate_model(model, dataset, r)
    except Exception as e:
        r['error'] = str(e)
    return r


def run_with_model(model, dataset):
    """Evaluate already-fitted model on test set."""
    r = {'success': False}
    try:
        return _evaluate_model(model, dataset, r)
    except Exception as e:
        r['error'] = str(e)
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


def _evaluate_model(model, dataset, r):
    """Common evaluation logic."""
    # Predictions on test set
    ytp = model.predict(dataset.X_test)
    yt = dataset.y_test_noisy if hasattr(dataset, 'y_test_noisy') else dataset.y_test
    
    # R²
    ss = np.sum((yt - ytp) ** 2)
    st = np.sum((yt - np.mean(yt)) ** 2)
    r['test_r2'] = float(1 - ss/st if st > 0 else 0)
    
    # RMSE
    r['test_rmse'] = float(np.sqrt(ss / len(yt)))
    
    # Terms - CORRECTED to handle different model formats
    sel = []
    if hasattr(model, 'result_') and model.result_:
        sel = model.result_.selected_terms
    elif hasattr(model, '_sel_terms'):
        # PolyKnockoff: _sel_terms is [[base_idx, exp], ...] without interaction
        # Need to expand to match ground truth format [base, exp, interaction]
        raw_terms = model._sel_terms
        sel = []
        for t in raw_terms:
            if len(t) >= 2:
                base_idx = t[0]
                exp = t[1]
                # Check if there's interaction info in the original term
                interaction = t[2] if len(t) > 2 else []
                sel.append([base_idx, exp, interaction] if interaction else [base_idx, exp])
            else:
                sel.append(t)
    elif hasattr(model, 'selected_indices'):
        # PolyLasso/PolyOMP: selected_indices are column indices
        # Need to map to base_feature_indices, power_exponents, interaction_indices
        indices = model.selected_indices
        sel = []
        for idx in indices:
            base_idx = model._base_feature_indices[idx]
            exp = model._power_exponents[idx]
            # Check if model has interaction_indices attribute
            if hasattr(model, '_interaction_indices') and idx < len(model._interaction_indices):
                interaction = model._interaction_indices[idx]
                if interaction is not None and len(interaction) > 0:
                    sel.append([base_idx, exp, interaction])
                else:
                    sel.append([base_idx, exp])
            else:
                sel.append([base_idx, exp])
    
    def norm(t):
        if not isinstance(t, (list, tuple)): return t
        b, e = t[0], t[1]
        i = tuple(t[2]) if len(t) > 2 and isinstance(t[2], (list, tuple)) else ()
        return (b, e, i)
    
    ss_set = set(norm(t) for t in sel)
    ts_set = set(norm(t) for t in dataset.true_poly_terms)
    tp = len(ss_set & ts_set)
    fp = len(ss_set - ts_set)
    
    r['n_selected'] = len(sel)
    r['fdr'] = fp / len(ss_set) if ss_set else 0.0
    r['tpr'] = tp / len(ts_set) if ts_set else 0.0
    # Convert to JSON serializable format
    r['selected_terms'] = _to_json_serializable(sel)
    r['true_terms'] = _to_json_serializable(dataset.true_poly_terms)
    r['success'] = True
    return r


def main():
    print("="*80)
    print("FINAL EXPERIMENT - Validation Set (100 samples)")
    print("="*80)
    print("Training: n | Validation: 100 | Test: 200")
    print("="*80)
    
    configs = []
    for n in [50, 80, 100]:
        for p in [5, 8, 10]:
            for k in [3, 5]:
                for noise in [0.1, 0.5]:
                    for d in [2, 3]:
                        configs.append((n, p, k, d, noise))
    
    total = len(configs) * N_TRIALS
    print(f"\nConfigs: {len(configs)} x {N_TRIALS} trials = {total}")
    print(f"Methods per config:")
    print(f"  - IC-Knock-Poly: 3 fixed Q + 1 Val-CV = 4")
    print(f"  - Poly-Knockoff: 3 fixed Q + 1 Val-CV = 4")
    print(f"  - Poly-Lasso: 1 (val-select) = 1")
    print(f"  - Poly-OMP: 1 (val-select) = 1")
    print(f"  - Poly-CLIME: 1 (val-select) = 1")
    print(f"  Total: 11 methods per config\n")
    
    results = []
    completed = 0
    start = time.time()
    
    for n, p, k, d, noise in configs:
        for trial in range(N_TRIALS):
            rs = abs(hash(f"{n}_{p}_{k}_{d}_{noise}_{trial}")) % (2**31)
            
            # Generate: n_train + 100 val + 200 test
            dataset = generate_simulation(
                n_labeled=n + N_VAL,  # train + val
                p=p, k=k, degree=d, 
                noise_std=noise, 
                n_test=200, 
                random_state=rs
            )
            
            # Split train/val
            X_train = dataset.X[:n]
            y_train = dataset.y[:n]
            X_val = dataset.X[n:n+N_VAL]
            y_val = dataset.y[n:n+N_VAL]
            
            ci = {'n': n, 'p': p, 'k': k, 'degree': d, 'noise': noise, 'trial': trial}
            
            print(f"\nConfig: n={n}, p={p}, k={k}, d={d}, noise={noise}, trial={trial}")
            
            # IC-Knock-Poly: 3 fixed Q (on full train+val for best performance)
            for q in [0.05, 0.10, 0.15]:
                print(f"  IC-Knock-Poly-Q{q}...")
                m = ICKnockoffPolyReg(degree=d, Q=q, max_iter=10, backend=BACKEND, random_state=rs)
                r = run(m, dataset, X_train, y_train)  # Only use train
                r.update({'method': f'IC-Knock-Poly-Q{q}', 'config': ci})
                results.append(r)
            
            # IC-Knock-Poly-Val: select Q from 3 values using validation set
            print(f"  IC-Knock-Poly-Val...")
            best_m, best_q, best_s = None, None, -float('inf')
            for q in [0.05, 0.10, 0.15]:
                m, s = val_score(ICKnockoffPolyReg, X_train, y_train, X_val, y_val, 
                             {'degree': d, 'Q': q, 'max_iter': 10, 'backend': BACKEND, 'random_state': rs})
                if s > best_s:
                    best_s, best_q, best_m = s, q, m
            r = run_with_model(best_m, dataset)
            r.update({'method': 'IC-Knock-Poly-Val', 'config': ci, 'best_Q': best_q})
            results.append(r)
            
            # Poly-Lasso: select alpha from 3 values
            print(f"  Poly-Lasso-Val...")
            best_m, best_a, best_s = None, None, -float('inf')
            for a in [0.001, 0.01, 0.1]:
                m, s = val_score(PolyLasso, X_train, y_train, X_val, y_val,
                             {'degree': d, 'alpha': a, 'random_state': rs})
                if s > best_s:
                    best_s, best_a, best_m = s, a, m
            r = run_with_model(best_m, dataset)
            r.update({'method': 'Poly-Lasso-Val', 'config': ci, 'best_alpha': best_a})
            results.append(r)
            
            # Poly-OMP: select k from 3 values
            print(f"  Poly-OMP-Val...")
            best_m, best_k, best_s = None, None, -float('inf')
            for nk in [3, 7, 15]:
                m, s = val_score(PolyOMP, X_train, y_train, X_val, y_val,
                             {'degree': d, 'n_nonzero_coefs': nk})
                if s > best_s:
                    best_s, best_k, best_m = s, nk, m
            r = run_with_model(best_m, dataset)
            r.update({'method': 'Poly-OMP-Val', 'config': ci, 'best_k': best_k})
            results.append(r)
            
            # Poly-CLIME: select alpha from 3 values
            print(f"  Poly-CLIME-Val...")
            best_m, best_a, best_s = None, None, -float('inf')
            for a in [0.005, 0.05, 0.5]:
                m, s = val_score(PolyCLIME, X_train, y_train, X_val, y_val,
                             {'degree': d, 'alpha': a, 'Q': 0.10, 'random_state': rs})
                if s > best_s:
                    best_s, best_a, best_m = s, a, m
            r = run_with_model(best_m, dataset)
            r.update({'method': 'Poly-CLIME-Val', 'config': ci, 'best_alpha': best_a})
            results.append(r)
            
            # Poly-Knockoff: 3 fixed Q (to show FDR control)
            for q in [0.05, 0.10, 0.15]:
                print(f"  Poly-Knockoff-Q{q}...")
                m = PolyKnockoff(degree=d, Q=q, random_state=rs)
                r = run(m, dataset, X_train, y_train)
                r.update({'method': f'Poly-Knockoff-Q{q}', 'config': ci})
                results.append(r)
            
            # Poly-Knockoff-Val: select Q from 3 values using validation
            print(f"  Poly-Knockoff-Val...")
            best_m, best_q, best_s = None, None, -float('inf')
            for q in [0.05, 0.10, 0.15]:
                m, s = val_score(PolyKnockoff, X_train, y_train, X_val, y_val,
                             {'degree': d, 'Q': q, 'random_state': rs})
                if s > best_s:
                    best_s, best_q, best_m = s, q, m
            r = run_with_model(best_m, dataset)
            r.update({'method': 'Poly-Knockoff-Val', 'config': ci, 'best_Q': best_q})
            results.append(r)
            
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

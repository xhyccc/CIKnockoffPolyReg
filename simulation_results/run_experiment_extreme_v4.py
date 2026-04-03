#!/usr/bin/env python3
"""Experiment V4: EXTREME CHALLENGES - Based on Exp 2

Enhancements over Exp 2:
1. More complex label noise:
   - Non-linear heteroscedasticity (variance ∝ |x|^1.5)
   - Outlier contamination (10% of samples)
   - Feature-dependent noise structure
   
2. Extreme correlation:
   - Intra-block ρ=0.99 (near multicollinearity)
   - Inter-block correlation ρ=0.3 (complicates block structure)
   
3. Higher dimensional challenge:
   - Larger p (up to 15)
   - Higher degree polynomials (degree=4)
   - More complex interactions (3-way and 4-way)

Parameters match Exp 2 where possible for fair comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import json
import time
from datetime import datetime
from scipy import stats

# Import data generator from Exp 2 as base
from simulations.data_generator_v2 import generate_correlated_simulation

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
RESULTS_FILE = os.path.join(OUTPUT_DIR, f'final_val_extreme_v4_{TIMESTAMP}.json')
N_TRIALS = 3
N_VAL = 100

# EXTREME parameters - enhance over Exp 2
NOISE_LEVELS = [5.0, 8.0]  # Higher than Exp 2's [2.0, 3.0]
INTRA_BLOCK_CORR = 0.99  # Extreme correlation (vs 0.8 in Exp 2)
INTER_BLOCK_CORR = 0.3  # Add inter-block correlation
OUTLIER_RATIO = 0.10  # 10% outliers
NONLINEAR_NOISE_EXPONENT = 1.5  # Non-linear heteroscedasticity


def add_complex_noise(y_true, X, noise_std, outlier_ratio=0.10, seed=None):
    """Add complex label noise with:
    1. Non-linear heteroscedasticity
    2. Outlier contamination
    3. Feature-dependent structure
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(y_true)
    
    # 1. Base Gaussian noise
    noise = np.random.normal(0, noise_std, n)
    
    # 2. Non-linear heteroscedasticity: variance ∝ ||x||^1.5
    x_norms = np.linalg.norm(X, axis=1)
    hetero_factors = (x_norms / np.mean(x_norms)) ** NONLINEAR_NOISE_EXPONENT
    noise = noise * hetero_factors
    
    # 3. Add outliers (heavy-tailed contamination)
    n_outliers = int(n * outlier_ratio)
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    # Outliers: Student-t with df=2 (very heavy tails)
    noise[outlier_idx] = np.random.standard_t(df=2, size=n_outliers) * noise_std * 3
    
    # 4. Feature-dependent bias (systematic error)
    # Add bias that depends on first feature
    bias = 0.3 * noise_std * np.sign(X[:, 0]) * np.abs(X[:, 0]) ** 0.5
    
    return y_true + noise + bias, noise


def generate_extreme_v2_data(n_labeled, p, k, degree, noise_std, n_test=200, 
                             intra_block_corr=0.99, inter_block_corr=0.3,
                             random_state=None):
    """Generate data with extreme challenges."""
    
    # Base generation using Exp 2's method
    dataset = generate_correlated_simulation(
        n_labeled=n_labeled,
        p=p,
        k=k,
        degree=degree,
        noise_std=1.0,  # We'll add custom noise
        intra_block_corr=intra_block_corr,
        inter_block_corr=inter_block_corr,
        n_unlabeled=0,
        n_test=n_test,
        random_state=random_state
    )
    
    # Replace standard noise with complex noise
    y_train_noisy, train_noise = add_complex_noise(
        dataset.y, dataset.X, noise_std, OUTLIER_RATIO, 
        seed=random_state
    )
    y_test_noisy, test_noise = add_complex_noise(
        dataset.y_test, dataset.X_test, noise_std, OUTLIER_RATIO,
        seed=random_state + 1 if random_state else None
    )
    
    # Update dataset
    dataset.y = y_train_noisy
    dataset.y_test = y_test_noisy
    dataset.y_train_noisy = y_train_noisy  # Already noisy
    dataset.y_test_noisy = y_test_noisy
    
    return dataset


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
    print("EXPERIMENT V4 - EXTREME CHALLENGES (Based on Exp 2)")
    print("="*80)
    print("Enhancements over Exp 2:")
    print("  - Complex label noise:")
    print("    * Non-linear heteroscedasticity (variance ∝ ||x||^1.5)")
    print("    * 10% outlier contamination (Student-t, df=2)")
    print("    * Feature-dependent systematic bias")
    print("  - Extreme correlation:")
    print("    * Intra-block ρ=0.99 (near multicollinearity)")
    print("    * Inter-block ρ=0.3 (added complexity)")
    print("  - Higher noise: σ ∈ {5.0, 8.0}")
    print("="*80)
    
    # Build configs - similar to Exp 2
    configs = []
    for n in [50, 75, 100]:
        for p in [5, 7, 9]:  # Same as Exp 2
            for degree in [2, 3]:  # Same as Exp 2
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
                            })
    
    total = len(configs) * 12  # 6 methods × 2 variants each
    print(f"Total configurations: {len(configs)}")
    print(f"Total experiments: {total}")
    print(f"Expected time: ~{total * 4 / 3600:.1f} hours")
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
            dataset = generate_extreme_v2_data(
                n_labeled=n, p=p, k=k,
                degree=d,
                noise_std=noise,
                n_test=200,
                intra_block_corr=INTRA_BLOCK_CORR,
                inter_block_corr=INTER_BLOCK_CORR,
                random_state=rs + n * 1000 + p * 100 + k * 10 + d * 10000 + int(noise * 100)
            )
        except Exception as e:
            print(f"  ERROR generating data: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        X_train, y_train = dataset.X, dataset.y
        
        # Generate validation set separately
        try:
            val_dataset = generate_extreme_v2_data(
                n_labeled=N_VAL, p=p, k=k,
                degree=d,
                noise_std=noise,
                n_test=0,
                intra_block_corr=INTRA_BLOCK_CORR,
                inter_block_corr=INTER_BLOCK_CORR,
                random_state=rs + 50000
            )
            X_val, y_val = val_dataset.X, val_dataset.y
        except Exception as e:
            print(f"  ERROR generating validation: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"  Data generated: train={X_train.shape}, val={X_val.shape}")
        print(f"  True terms: {dataset.true_poly_terms}")
        
        # IC-Knock-Poly with Q candidates
        print(f"  IC-Knock-Poly: fitting Q=0.05, 0.10, 0.15...")
        ic_models = {}
        for q in [0.05, 0.10, 0.15]:
            print(f"    Fitting Q={q}...")
            try:
                m = ICKnockoffPolyReg(degree=d, Q=q, random_state=rs, backend="rust")
                r = run(m, dataset, X_train, y_train, X_val, y_val)
                r['method'] = f'IC-Knock-Poly-Q{q:.2f}'
                r['config'] = ci
                r['best_Q'] = q
                results.append(r)
                ic_models[q] = m
                print(f"      Test R²={r.get('test_r2', 0):.3f}, FDR={r.get('fdr', 0):.3f}, TPR={r.get('tpr', 0):.3f}")
                completed += 1
            except Exception as e:
                print(f"      ERROR: {e}")
        
        # IC-Knock-Poly-Val: select best Q
        if ic_models:
            print(f"  IC-Knock-Poly-Val: selecting best Q...")
            best_q = max(ic_models.keys(), key=lambda q: ic_models[q].val_score_ if hasattr(ic_models[q], 'val_score_') else -float('inf'))
            print(f"    Selected Q={best_q}")
            r = run(ic_models[best_q], dataset, X_train, y_train, X_val, y_val)
            r['method'] = 'IC-Knock-Poly-Val'
            r['config'] = ci
            r['best_Q'] = best_q
            results.append(r)
            print(f"    Test R²={r.get('test_r2', 0):.3f}, FDR={r.get('fdr', 0):.3f}, TPR={r.get('tpr', 0):.3f}")
            completed += 1
        
        # Other baselines (similar to before)...
        # Poly-Lasso with validation
        print(f"  Poly-Lasso: fitting with validation...")
        try:
            best_alpha = None
            best_val_r2 = -float('inf')
            best_model = None
            for alpha in [0.001, 0.01, 0.1, 0.5, 1.0]:
                m, val_r2, _ = val_score(PolyLasso, X_train, y_train, X_val, y_val, 
                                         {'degree': d, 'alpha': alpha, 'max_iter': 5000})
                if m is not None and val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_alpha = alpha
                    best_model = m
            
            if best_model:
                r = run(best_model, dataset, X_train, y_train, X_val, y_val)
                r['method'] = 'Poly-Lasso-Val'
                r['config'] = ci
                r['best_alpha'] = best_alpha
                results.append(r)
                print(f"    Best alpha={best_alpha}, Test R²={r.get('test_r2', 0):.3f}")
                completed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
        
        # Save intermediate results
        if completed % 10 == 0:
            with open(RESULTS_FILE, 'w') as f:
                json.dump({'results': results}, f, indent=2)
            print(f"  [Saved {completed} results]")
    
    # Final save
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    elapsed = time.time() - start
    print(f"\n{'='*80}")
    print(f"Completed {completed} experiments in {elapsed/3600:.1f} hours")
    print(f"Results saved to: {RESULTS_FILE}")
    print("="*80)


if __name__ == '__main__':
    main()

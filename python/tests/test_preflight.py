#!/usr/bin/env python3
"""Comprehensive pre-flight check for all experiment scripts.

Run this before launching long experiments to catch any errors.
"""

import sys
import os
import traceback

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PARENT, 'src'))
sys.path.insert(0, _PARENT)

def test_imports():
    """Test all required imports."""
    print("\n" + "="*70)
    print("TEST 1: Import Check")
    print("="*70)
    
    imports = [
        ("simulations.data_generator", "generate_simulation"),
        ("ic_knockoff_poly_reg", "ICKnockoffPolyReg"),
        ("ic_knockoff_poly_reg.evaluation", "normalize_polynomial_term"),
        ("baselines.poly_knockoff", "PolyKnockoff"),
        ("baselines.poly_lasso", "PolyLasso"),
        ("baselines.poly_omp", "PolyOMP"),
        ("baselines.poly_clime", "PolyCLIME"),
        ("baselines.sparse_poly_stlsq", "SparsePolySTLSQ"),
    ]
    
    for module_name, obj_name in imports:
        try:
            module = __import__(module_name, fromlist=[obj_name])
            obj = getattr(module, obj_name)
            print(f"  ✓ {module_name}.{obj_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{obj_name}: {e}")
            return False
    
    print("\n✓ All imports successful")
    return True


def test_data_generator():
    """Test data generator with small dataset."""
    print("\n" + "="*70)
    print("TEST 2: Data Generator")
    print("="*70)
    
    try:
        from simulations.data_generator import generate_simulation
        
        dataset = generate_simulation(
            n_labeled=30,  # 20 train + 10 val
            p=3, k=2, degree=2,
            noise_std=0.1, n_test=10, random_state=42
        )
        
        # Check attributes
        assert dataset.X.shape[0] == 30, "X shape mismatch"
        assert len(dataset.true_poly_terms) == 2, f"Expected 2 terms, got {len(dataset.true_poly_terms)}"
        
        # Check format
        for term in dataset.true_poly_terms:
            assert len(term) in [2, 4], f"Invalid term format: {term}"
        
        print(f"  ✓ Generated dataset: n=30, p=3, k=2")
        print(f"  ✓ True terms: {len(dataset.true_poly_terms)}")
        print(f"  ✓ Term format valid")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def test_ic_knock_poly():
    """Test IC-Knock-Poly model."""
    print("\n" + "="*70)
    print("TEST 3: IC-Knock-Poly Model")
    print("="*70)
    
    try:
        from simulations.data_generator import generate_simulation
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        
        # Generate data
        dataset = generate_simulation(
            n_labeled=40, p=3, k=2, degree=2,
            noise_std=0.1, n_test=10, random_state=42
        )
        X_train, y_train = dataset.X[:25], dataset.y[:25]
        
        # Test different Q values
        models = {}
        for q in [0.05, 0.10, 0.15]:
            model = ICKnockoffPolyReg(
                degree=2, Q=q, max_iter=3,
                backend='rust', random_state=42
            )
            model.fit(X_train, y_train)
            models[q] = model
            
            if hasattr(model, 'result_') and model.result_:
                n_sel = len(model.result_.selected_terms)
                print(f"  ✓ Q={q}: selected {n_sel} terms")
            else:
                print(f"  ✓ Q={q}: no terms selected")
        
        # Test prediction
        X_test = dataset.X_test
        for q, model in models.items():
            y_pred = model.predict(X_test)
            assert len(y_pred) == len(X_test), "Prediction shape mismatch"
        
        print(f"  ✓ Prediction works")
        
        # Test Val-CV selection (reuse models)
        X_val, y_val = dataset.X[25:], dataset.y[25:]
        best_q, best_model, best_score = None, None, -float('inf')
        
        for q, model in models.items():
            y_pred = model.predict(X_val)
            ss = sum((y_val - y_pred) ** 2)
            st = sum((y_val - y_val.mean()) ** 2)
            score = 1 - ss/st if st > 0 else 0
            
            if score > best_score:
                best_score, best_q, best_model = score, q, model
        
        print(f"  ✓ Val-CV selection: best Q={best_q} (val_r2={best_score:.3f})")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def test_baselines():
    """Test all baseline models."""
    print("\n" + "="*70)
    print("TEST 4: Baseline Models")
    print("="*70)
    
    from simulations.data_generator import generate_simulation
    from baselines.poly_lasso import PolyLasso
    from baselines.poly_omp import PolyOMP
    from baselines.poly_clime import PolyCLIME
    from baselines.poly_knockoff import PolyKnockoff
    from baselines.sparse_poly_stlsq import SparsePolySTLSQ
    
    # Generate data
    dataset = generate_simulation(
        n_labeled=40, p=3, k=2, degree=2,
        noise_std=0.1, n_test=10, random_state=42
    )
    X_train, y_train = dataset.X[:25], dataset.y[:25]
    
    baselines = [
        ('PolyLasso', PolyLasso(degree=2, alpha=0.01, random_state=42)),
        ('PolyOMP', PolyOMP(degree=2, n_nonzero_coefs=3)),
        ('PolyCLIME', PolyCLIME(degree=2, alpha=0.01, random_state=42)),
        ('PolyKnockoff', PolyKnockoff(degree=2, Q=0.10, random_state=42)),
        ('SparsePolySTLSQ', SparsePolySTLSQ(degree=2, threshold=0.01)),
    ]
    
    all_pass = True
    for name, model in baselines:
        try:
            model.fit(X_train, y_train)
            
            # Check selected terms format
            sel = []
            if hasattr(model, 'selected_indices'):
                sel = model.selected_indices
            elif hasattr(model, '_sel_terms'):
                sel = model._sel_terms
            elif hasattr(model, 'result_') and model.result_:
                sel = model.result_.selected_terms
            
            # Test prediction
            y_pred = model.predict(dataset.X_test)
            assert len(y_pred) == len(dataset.X_test)
            
            print(f"  ✓ {name}: {len(sel)} terms, prediction OK")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_pass = False
    
    return all_pass


def test_fdr_calculation():
    """Test FDR/TPR calculation."""
    print("\n" + "="*70)
    print("TEST 5: FDR/TPR Calculation")
    print("="*70)
    
    try:
        from simulations.data_generator import generate_simulation
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        from ic_knockoff_poly_reg.evaluation import compute_polynomial_term_metrics
        
        # Generate data
        dataset = generate_simulation(
            n_labeled=40, p=3, k=2, degree=2,
            noise_std=0.1, n_test=10, random_state=42
        )
        X_train, y_train = dataset.X[:25], dataset.y[:25]
        
        # Fit model
        model = ICKnockoffPolyReg(
            degree=2, Q=0.10, max_iter=3,
            backend='rust', random_state=42
        )
        model.fit(X_train, y_train)
        
        if hasattr(model, 'result_') and model.result_:
            selected = model.result_.selected_terms
            
            # Calculate metrics
            metrics = compute_polynomial_term_metrics(
                selected, dataset.true_poly_terms
            )
            
            print(f"  ✓ Selected: {metrics.n_selected}")
            print(f"  ✓ True Positives: {metrics.n_true_positives}")
            print(f"  ✓ False Positives: {metrics.n_false_positives}")
            print(f"  ✓ FDR: {metrics.fdr:.3f}")
            print(f"  ✓ TPR: {metrics.tpr:.3f}")
            
            assert 0 <= metrics.fdr <= 1, "FDR out of range"
            assert 0 <= metrics.tpr <= 1, "TPR out of range"
        else:
            print(f"  ✓ No terms selected (FDR=0, TPR=0)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def test_experiment_script_syntax():
    """Check experiment scripts for syntax errors."""
    print("\n" + "="*70)
    print("TEST 6: Experiment Script Syntax Check")
    print("="*70)
    
    scripts = [
        ('simulation_results', 'run_final_experiments.py'),
        ('simulation_results', 'run_enhanced_experiments.py'),
    ]
    
    all_pass = True
    for script_dir, script_name in scripts:
        script_path = os.path.join(os.path.dirname(_PARENT), script_dir, script_name)
        script = f"{script_dir}/{script_name}"
        if not os.path.exists(script_path):
            print(f"  ✗ {script}: file not found")
            all_pass = False
            continue
        
        try:
            with open(script_path, 'r') as f:
                code = f.read()
            compile(code, script_path, 'exec')
            print(f"  ✓ {script}: syntax OK")
        except SyntaxError as e:
            print(f"  ✗ {script}: syntax error at line {e.lineno}")
            all_pass = False
        except Exception as e:
            print(f"  ✗ {script}: {e}")
            all_pass = False
    
    return all_pass


def test_minimal_workflow():
    """Test minimal end-to-end workflow."""
    print("\n" + "="*70)
    print("TEST 7: Minimal End-to-End Workflow")
    print("="*70)
    
    try:
        from simulations.data_generator import generate_simulation
        from ic_knockoff_poly_reg import ICKnockoffPolyReg
        from ic_knockoff_poly_reg.evaluation import compute_polynomial_term_metrics
        
        # Simulate one trial
        n, p, k, d, noise = 50, 3, 2, 2, 0.1
        N_VAL = 10
        
        dataset = generate_simulation(
            n_labeled=n + N_VAL, p=p, k=k, degree=d,
            noise_std=noise, n_test=20, random_state=42
        )
        
        X_train = dataset.X[:n]
        y_train = dataset.y[:n]
        X_val = dataset.X[n:n+N_VAL]
        y_val = dataset.y[n:n+N_VAL]
        
        print(f"  Config: n={n}, p={p}, k={k}, d={d}")
        
        # Fit 3 Q values
        models = {}
        for q in [0.05, 0.10, 0.15]:
            model = ICKnockoffPolyReg(degree=d, Q=q, max_iter=3, backend='rust', random_state=42)
            model.fit(X_train, y_train)
            models[q] = model
            
            # Evaluate
            y_pred = model.predict(dataset.X_test)
            ss = sum((dataset.y_test - y_pred) ** 2)
            st = sum((dataset.y_test - dataset.y_test.mean()) ** 2)
            r2 = 1 - ss/st if st > 0 else 0
            
            n_sel = len(model.result_.selected_terms) if model.result_ else 0
            print(f"    Q={q}: {n_sel} terms, test_r2={r2:.3f}")
        
        # Val-CV selection
        best_q, best_score = None, -float('inf')
        for q, model in models.items():
            y_pred = model.predict(X_val)
            ss = sum((y_val - y_pred) ** 2)
            st = sum((y_val - y_val.mean()) ** 2)
            score = 1 - ss/st if st > 0 else 0
            if score > best_score:
                best_score, best_q = score, q
        
        print(f"  ✓ Val-CV selected Q={best_q}")
        
        # Calculate FDR/TPR for best model
        best_model = models[best_q]
        if best_model.result_:
            metrics = compute_polynomial_term_metrics(
                best_model.result_.selected_terms,
                dataset.true_poly_terms
            )
            print(f"  ✓ FDR={metrics.fdr:.3f}, TPR={metrics.tpr:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PRE-FLIGHT CHECK FOR EXPERIMENTS")
    print("="*70)
    print(f"Working directory: {_PARENT}")
    
    tests = [
        ("Imports", test_imports),
        ("Data Generator", test_data_generator),
        ("IC-Knock-Poly Model", test_ic_knock_poly),
        ("Baseline Models", test_baselines),
        ("FDR Calculation", test_fdr_calculation),
        ("Script Syntax", test_experiment_script_syntax),
        ("End-to-End Workflow", test_minimal_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED - Ready to run experiments!")
        print("="*70)
        print("\nRun experiments with:")
        print("  cd /Users/haoyi/Desktop/CIKnockoffPolyReg")
        print("  python simulation_results/run_final_experiments.py")
        return 0
    else:
        print("\n" + "="*70)
        print("✗ SOME TESTS FAILED - Please fix errors before running experiments")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

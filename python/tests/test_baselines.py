"""Unit tests for baseline methods and experiment scripts.

Ensures all baselines produce consistent output formats.
"""

import sys
import os
import unittest
import numpy as np

# Add paths
_HERE = os.path.dirname(__file__)
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PARENT, 'src'))
sys.path.insert(0, _PARENT)

from simulations.data_generator import generate_simulation
from ic_knockoff_poly_reg.evaluation import normalize_polynomial_term, compute_polynomial_term_metrics
from ic_knockoff_poly_reg import ICKnockoffPolyReg


class TestMainPipeline(unittest.TestCase):
    """Test main pipeline components - format consistency."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dataset = generate_simulation(
            n_labeled=60, p=4, k=2, degree=2,
            noise_std=0.1, n_test=20, random_state=42
        )
        self.X_train = self.dataset.X[:40]
        self.y_train = self.dataset.y[:40]
        
    def test_data_generator_format(self):
        """Data generator should produce 2 or 4 element terms."""
        for term in self.dataset.true_poly_terms:
            self.assertIn(len(term), [2, 4])
            normalized = normalize_polynomial_term(term)
            self.assertIsInstance(normalized, tuple)
            
    def test_ic_knock_poly_format(self):
        """IC-Knock-Poly should output normalizable terms."""
        model = ICKnockoffPolyReg(
            degree=2, Q=0.10, max_iter=3,
            backend='rust', random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        if hasattr(model, 'result_') and model.result_:
            selected = model.result_.selected_terms
            for term in selected:
                normalized = normalize_polynomial_term(term)
                self.assertIsInstance(normalized, tuple)
                
    def test_fdr_calculation_with_real_data(self):
        """FDR calculation should work end-to-end."""
        model = ICKnockoffPolyReg(
            degree=2, Q=0.10, max_iter=3,
            backend='rust', random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        if hasattr(model, 'result_') and model.result_:
            selected = model.result_.selected_terms
            metrics = compute_polynomial_term_metrics(
                selected, self.dataset.true_poly_terms
            )
            
            self.assertGreaterEqual(metrics.fdr, 0.0)
            self.assertLessEqual(metrics.fdr, 1.0)
            self.assertGreaterEqual(metrics.tpr, 0.0)
            self.assertLessEqual(metrics.tpr, 1.0)


class TestExperimentWorkflow(unittest.TestCase):
    """Test experiment script workflows."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dataset = generate_simulation(
            n_labeled=100, p=4, k=2, degree=2,
            noise_std=0.1, n_test=50, random_state=42
        )
        
    def test_train_val_test_split(self):
        """Train/val/test split should work."""
        n, n_val = 50, 30
        X_train = self.dataset.X[:n]
        y_train = self.dataset.y[:n]
        X_val = self.dataset.X[n:n+n_val]
        y_val = self.dataset.y[n:n+n_val]
        X_test = self.dataset.X[n+n_val:]
        y_test = self.dataset.y[n+n_val:]
        
        self.assertEqual(len(X_train), n)
        self.assertEqual(len(X_val), n_val)
        self.assertEqual(len(y_train), n)
        self.assertEqual(len(y_val), n_val)
        
    def test_model_prediction(self):
        """Model should be able to predict on test set."""
        X_train = self.dataset.X[:50]
        y_train = self.dataset.y[:50]
        X_test = self.dataset.X_test
        
        model = ICKnockoffPolyReg(
            degree=2, Q=0.10, max_iter=3,
            backend='rust', random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        self.assertEqual(len(y_pred), len(X_test))
        
    def test_config_generation(self):
        """Config generation for experiments."""
        configs = []
        for n in [50, 80]:
            for p in [4, 5]:
                for k in [2, 3]:
                    for d in [2]:
                        if k <= 2 * d * p:
                            configs.append({'n': n, 'p': p, 'k': k, 'd': d})
                            
        self.assertGreater(len(configs), 0)
        for cfg in configs:
            self.assertIn('n', cfg)
            self.assertIn('p', cfg)
            self.assertIn('k', cfg)


class TestFormatNormalization(unittest.TestCase):
    """Test term format normalization."""
    
    def test_monomial_2_elements(self):
        """[base, exp] format."""
        term = [0, 2]
        norm = normalize_polynomial_term(term)
        self.assertEqual(norm, (0, 2))
        
    def test_interaction_4_elements(self):
        """[base, exp, indices, exponents] format."""
        term = [-2, 2, [0, 1], [1, -1]]
        norm = normalize_polynomial_term(term)
        self.assertEqual(norm, (-2, 2, ((0, 1), (1, -1))))
        
    def test_interaction_sorted(self):
        """Interaction features should be sorted."""
        term = [-2, 2, [1, 0], [-1, 1]]  # Unsorted
        norm = normalize_polynomial_term(term)
        # Should be sorted by feature index
        self.assertEqual(norm, (-2, 2, ((0, 1), (1, -1))))
        
    def test_legacy_3_elements(self):
        """Backward compatibility: 3-element format."""
        term = [-2, 2, [0, 1]]
        norm = normalize_polynomial_term(term)
        self.assertEqual(norm, (-2, 2, (0, 1)))
        
    def test_term_list_normalization(self):
        """normalize_term_list should work."""
        from ic_knockoff_poly_reg.evaluation import normalize_term_list
        terms = [[0, 2], [-2, 2, [0, 1], [1, -1]]]
        norms = normalize_term_list(terms)
        self.assertEqual(len(norms), 2)
        self.assertIsInstance(norms[0], tuple)
        self.assertIsInstance(norms[1], tuple)


class TestFDRCalculation(unittest.TestCase):
    """Test FDR/TPR calculation."""
    
    def test_perfect_match(self):
        """Perfect selection: FDR=0, TPR=1."""
        true = [[0, 2], [-2, 2, [0, 1], [1, 2]]]
        selected = [[0, 2], [-2, 2, [0, 1], [1, 2]]]
        
        metrics = compute_polynomial_term_metrics(selected, true)
        self.assertEqual(metrics.fdr, 0.0)
        self.assertEqual(metrics.tpr, 1.0)
        
    def test_all_false_positives(self):
        """All false positives: FDR=1, TPR=0."""
        true = [[0, 2]]
        selected = [[1, 2]]
        
        metrics = compute_polynomial_term_metrics(selected, true)
        self.assertEqual(metrics.fdr, 1.0)
        self.assertEqual(metrics.tpr, 0.0)
        
    def test_mixed(self):
        """Mixed: 2 TP, 1 FP."""
        true = [[0, 2], [1, -1], [-2, 2, [0, 1], [1, 2]]]
        selected = [[0, 2], [1, -1], [2, 1]]
        
        metrics = compute_polynomial_term_metrics(selected, true)
        self.assertAlmostEqual(metrics.fdr, 1/3, places=5)
        self.assertAlmostEqual(metrics.tpr, 2/3, places=5)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMainPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestFDRCalculation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

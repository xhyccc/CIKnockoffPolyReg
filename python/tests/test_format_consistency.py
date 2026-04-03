"""Comprehensive unit tests for IC-Knock-Poly format consistency and FDR calculation.

Tests every component to ensure data generator, models, and evaluation use
consistent term formats.
"""

import sys
import os
import unittest
import numpy as np

# Add paths - test is in python/tests/, so parent is python/
_HERE = os.path.dirname(__file__)
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PARENT, 'src'))
sys.path.insert(0, _PARENT)

from simulations.data_generator import generate_simulation, SimulatedDataset
from ic_knockoff_poly_reg.polynomial import PolynomialDictionary, ExpandedFeatures
from ic_knockoff_poly_reg.evaluation import (
    normalize_polynomial_term,
    normalize_term_list,
    compute_polynomial_term_metrics,
    DiscoveryMetrics,
)
from ic_knockoff_poly_reg import ICKnockoffPolyReg


class TestPolynomialFormat(unittest.TestCase):
    """Test polynomial term format consistency."""
    
    def test_monomial_format_2_elements(self):
        """Monomials should have 2 elements: [base_idx, exp]."""
        term = [0, 2]
        normalized = normalize_polynomial_term(term)
        self.assertEqual(normalized, (0, 2))
        
    def test_monomial_format_negative_exp(self):
        """Monomials can have negative exponents."""
        term = [1, -2]
        normalized = normalize_polynomial_term(term)
        self.assertEqual(normalized, (1, -2))
        
    def test_interaction_format_4_elements(self):
        """Interactions should have 4 elements: [base, exp, indices, exponents]."""
        term = [-2, 2, [0, 1], [1, -1]]
        normalized = normalize_polynomial_term(term)
        self.assertEqual(normalized, (-2, 2, ((0, 1), (1, -1))))
        
    def test_interaction_format_single_feature(self):
        """Interactions with single feature."""
        term = [-2, 1, [0], [2]]
        normalized = normalize_polynomial_term(term)
        self.assertEqual(normalized, (-2, 1, ((0, 2),)))
        
    def test_legacy_format_3_elements(self):
        """Backward compatibility: 3-element format."""
        term = [-2, 2, [0, 1]]
        normalized = normalize_polynomial_term(term)
        self.assertEqual(normalized, (-2, 2, (0, 1)))
        
    def test_normalize_term_list(self):
        """normalize_term_list should work on lists."""
        terms = [[0, 2], [1, -1], [-2, 2, [0, 1], [1, 2]]]
        normalized = normalize_term_list(terms)
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[0], (0, 2))
        self.assertEqual(normalized[1], (1, -1))
        self.assertEqual(normalized[2], (-2, 2, ((0, 1), (1, 2))))
        
    def test_hashability(self):
        """Normalized terms must be hashable (for set operations)."""
        terms = [
            [0, 2],
            [-2, 2, [0, 1], [1, -1]],
            [-2, 3, [1, 2, 3], [1, 1, -1]],
        ]
        for term in terms:
            normalized = normalize_polynomial_term(term)
            # Should not raise TypeError
            try:
                {normalized}
            except TypeError:
                self.fail(f"Term {term} -> {normalized} is not hashable")


class TestPolynomialDictionary(unittest.TestCase):
    """Test PolynomialDictionary expansion."""
    
    def test_expanded_features_structure(self):
        """ExpandedFeatures should have all required fields."""
        poly = PolynomialDictionary(degree=2, include_bias=True)
        X = np.random.randn(10, 3)
        result = poly.expand(X)
        
        self.assertIsInstance(result.matrix, np.ndarray)
        self.assertIsInstance(result.feature_names, list)
        self.assertIsInstance(result.base_feature_indices, list)
        self.assertIsInstance(result.power_exponents, list)
        self.assertIsInstance(result.interaction_indices, list)
        self.assertIsInstance(result.interaction_exponents, list)
        
    def test_monomial_exponents_length(self):
        """Monomials should have None in interaction_exponents."""
        poly = PolynomialDictionary(degree=2, include_interactions=False)
        X = np.random.randn(10, 2)
        result = poly.expand(X)
        
        # All should be monomials
        for exp_list in result.interaction_exponents:
            self.assertIsNone(exp_list)
            
    def test_interaction_exponents_present(self):
        """Interactions should have exponents in interaction_exponents."""
        poly = PolynomialDictionary(degree=2, include_interactions=True)
        X = np.random.randn(10, 3)
        result = poly.expand(X)
        
        # Find interaction terms (base_feature_indices == -2)
        found_interaction = False
        for i, base_idx in enumerate(result.base_feature_indices):
            if base_idx == -2:
                found_interaction = True
                self.assertIsNotNone(result.interaction_indices[i])
                self.assertIsNotNone(result.interaction_exponents[i])
                self.assertEqual(
                    len(result.interaction_indices[i]),
                    len(result.interaction_exponents[i])
                )
                
        self.assertTrue(found_interaction, "No interaction terms found")
        
    def test_expanded_features_consistency(self):
        """All lists in ExpandedFeatures should have same length."""
        poly = PolynomialDictionary(degree=3, include_interactions=True)
        X = np.random.randn(10, 4)
        result = poly.expand(X)
        
        n_cols = result.matrix.shape[1]
        self.assertEqual(len(result.feature_names), n_cols)
        self.assertEqual(len(result.base_feature_indices), n_cols)
        self.assertEqual(len(result.power_exponents), n_cols)
        self.assertEqual(len(result.interaction_indices), n_cols)
        self.assertEqual(len(result.interaction_exponents), n_cols)


class TestDataGenerator(unittest.TestCase):
    """Test data generator output format."""
    
    def test_true_poly_terms_format(self):
        """Generated true_poly_terms should have correct format."""
        dataset = generate_simulation(
            n_labeled=50, p=4, k=3, degree=2,
            noise_std=0.1, n_test=20, random_state=42
        )
        
        self.assertIsInstance(dataset.true_poly_terms, list)
        self.assertEqual(len(dataset.true_poly_terms), 3)
        
        for term in dataset.true_poly_terms:
            self.assertIsInstance(term, list)
            self.assertIn(len(term), [2, 4], 
                         f"Term {term} has invalid length {len(term)}")
            
            if len(term) == 4:
                # Interaction: [base, exp, indices, exponents]
                self.assertIsInstance(term[2], list)  # indices
                self.assertIsInstance(term[3], list)  # exponents
                self.assertEqual(len(term[2]), len(term[3]))
                
    def test_data_generator_reproducibility(self):
        """Same random_state should produce same data."""
        ds1 = generate_simulation(
            n_labeled=50, p=3, k=2, degree=2,
            noise_std=0.1, n_test=10, random_state=42
        )
        ds2 = generate_simulation(
            n_labeled=50, p=3, k=2, degree=2,
            noise_std=0.1, n_test=10, random_state=42
        )
        
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.y, ds2.y)
        self.assertEqual(ds1.true_poly_terms, ds2.true_poly_terms)
        
    def test_k_constraint(self):
        """k should not exceed total possible terms."""
        # For p=3, d=2: 2*d*p = 12 monomials + interactions
        # Should work for k <= 12 (monomials only if no interactions)
        with self.assertRaises(ValueError):
            generate_simulation(
                n_labeled=50, p=3, k=50, degree=2,  # k too large
                noise_std=0.1, n_test=10, random_state=42
            )
            
    def test_dataset_attributes(self):
        """SimulatedDataset should have all required attributes."""
        dataset = generate_simulation(
            n_labeled=50, p=4, k=3, degree=2,
            noise_std=0.1, n_test=20, random_state=42
        )
        
        self.assertIsNotNone(dataset.X)
        self.assertIsNotNone(dataset.y)
        self.assertIsNotNone(dataset.X_test)
        self.assertIsNotNone(dataset.y_test)
        self.assertIsInstance(dataset.true_base_indices, set)
        self.assertIsInstance(dataset.true_poly_terms, list)
        self.assertIsInstance(dataset.true_coef, np.ndarray)


class TestFDRCalculation(unittest.TestCase):
    """Test FDR/TPR calculation accuracy."""
    
    def test_perfect_selection(self):
        """When selected == true, FDR=0, TPR=1."""
        true_terms = [[0, 2], [1, -1], [-2, 2, [0, 1], [1, 2]]]
        selected_terms = [[0, 2], [1, -1], [-2, 2, [0, 1], [1, 2]]]
        
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        
        self.assertEqual(metrics.fdr, 0.0)
        self.assertEqual(metrics.tpr, 1.0)
        self.assertEqual(metrics.n_selected, 3)
        self.assertEqual(metrics.n_true_positives, 3)
        self.assertEqual(metrics.n_false_positives, 0)
        
    def test_all_false_positives(self):
        """When selected != true, FDR=1, TPR=0."""
        true_terms = [[0, 2]]
        selected_terms = [[1, 2]]
        
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        
        self.assertEqual(metrics.fdr, 1.0)
        self.assertEqual(metrics.tpr, 0.0)
        self.assertEqual(metrics.n_true_positives, 0)
        self.assertEqual(metrics.n_false_positives, 1)
        
    def test_mixed_selection(self):
        """Mixed true positives and false positives."""
        true_terms = [[0, 2], [1, -1], [-2, 2, [0, 1], [1, 2]]]
        selected_terms = [[0, 2], [2, 1]]  # 1 TP, 1 FP
        
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        
        self.assertAlmostEqual(metrics.fdr, 0.5, places=5)
        self.assertAlmostEqual(metrics.tpr, 1/3, places=5)
        self.assertEqual(metrics.n_true_positives, 1)
        self.assertEqual(metrics.n_false_positives, 1)
        self.assertEqual(metrics.n_false_negatives, 2)
        
    def test_empty_selection(self):
        """Empty selection: FDR=0, TPR=0."""
        true_terms = [[0, 2], [1, -1]]
        selected_terms = []
        
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        
        self.assertEqual(metrics.fdr, 0.0)
        self.assertEqual(metrics.tpr, 0.0)
        self.assertEqual(metrics.n_selected, 0)
        
    def test_no_ground_truth(self):
        """When true_terms is empty: division by zero protection."""
        true_terms = []
        selected_terms = [[0, 2]]
        
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        
        self.assertEqual(metrics.fdr, 1.0)  # 1 FP / 1 selected
        self.assertEqual(metrics.tpr, 0.0)  # No true terms to find
        
    def test_interaction_term_matching(self):
        """Interaction terms should match on all fields."""
        true_terms = [[-2, 2, [0, 1], [1, -1]]]
        
        # Same interaction
        selected_terms = [[-2, 2, [0, 1], [1, -1]]]
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        self.assertEqual(metrics.tpr, 1.0)
        
        # Different indices
        selected_terms = [[-2, 2, [0, 2], [1, -1]]]
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        self.assertEqual(metrics.tpr, 0.0)
        
        # Different exponents
        selected_terms = [[-2, 2, [0, 1], [1, 1]]]
        metrics = compute_polynomial_term_metrics(selected_terms, true_terms)
        self.assertEqual(metrics.tpr, 0.0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests with full pipeline."""
    
    def test_full_pipeline_format_consistency(self):
        """Data generator and model should use same format."""
        # Generate data
        dataset = generate_simulation(
            n_labeled=100, p=4, k=2, degree=2,
            noise_std=0.1, n_test=30, random_state=42
        )
        
        # Train model
        X_train = dataset.X[:60]
        y_train = dataset.y[:60]
        
        model = ICKnockoffPolyReg(
            degree=2, Q=0.10, max_iter=3,
            backend='rust', random_state=42
        )
        model.fit(X_train, y_train)
        
        # Get selected terms
        if hasattr(model, 'result_') and model.result_:
            selected = model.result_.selected_terms
            
            # Verify format
            for term in selected:
                self.assertIn(len(term), [2, 4])
                # Should be normalizable
                try:
                    normalize_polynomial_term(term)
                except Exception as e:
                    self.fail(f"Cannot normalize selected term {term}: {e}")
                    
    def test_fdr_calculation_with_real_data(self):
        """FDR calculation should work with real generated data."""
        dataset = generate_simulation(
            n_labeled=80, p=3, k=2, degree=2,
            noise_std=0.1, n_test=20, random_state=42
        )
        
        X_train = dataset.X[:50]
        y_train = dataset.y[:50]
        
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
            
            # Verify metrics are valid
            self.assertGreaterEqual(metrics.fdr, 0.0)
            self.assertLessEqual(metrics.fdr, 1.0)
            self.assertGreaterEqual(metrics.tpr, 0.0)
            self.assertLessEqual(metrics.tpr, 1.0)
            
            # Verify counts
            self.assertEqual(
                metrics.n_true_positives + metrics.n_false_positives,
                metrics.n_selected
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_invalid_term_format(self):
        """Invalid term format should raise error."""
        with self.assertRaises(ValueError):
            normalize_polynomial_term("invalid")
            
    def test_empty_list(self):
        """Empty list should raise error."""
        with self.assertRaises(IndexError):
            normalize_polynomial_term([])
            
    def test_single_element(self):
        """Single element should fail."""
        with self.assertRaises(IndexError):
            normalize_polynomial_term([0])
            
    def test_five_element_format(self):
        """5-element format should use fallback."""
        term = [0, 2, 1, 1, 1]
        normalized = normalize_polynomial_term(term)
        self.assertEqual(normalized, (0, 2))  # Fallback to first two
        
    def test_none_in_interaction(self):
        """None values in interaction fields."""
        term = [-2, 2, None, None]
        normalized = normalize_polynomial_term(term)
        self.assertEqual(normalized, (-2, 2, None))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomialFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomialDictionary))
    suite.addTests(loader.loadTestsFromTestCase(TestDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestFDRCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

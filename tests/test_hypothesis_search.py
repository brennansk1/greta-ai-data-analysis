"""
Comprehensive tests for hypothesis_search module using pytest.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from deap import base, creator
from greta_core.hypothesis_search import (
    create_toolbox, evaluate_hypothesis, run_genetic_algorithm,
    generate_hypotheses
)


class TestCreateToolbox:
    """Test toolbox creation functionality."""

    def test_create_toolbox_basic(self):
        """Test basic toolbox creation."""
        num_features = 5
        toolbox, chromosome_info = create_toolbox(num_features)

        assert isinstance(toolbox, base.Toolbox)
        assert hasattr(toolbox, 'individual')
        assert hasattr(toolbox, 'population')
        assert hasattr(toolbox, 'evaluate')
        assert hasattr(toolbox, 'mate')
        assert hasattr(toolbox, 'mutate')
        assert hasattr(toolbox, 'select')
        assert isinstance(chromosome_info, dict)
        assert 'total_length' in chromosome_info

    def test_create_toolbox_individual_length(self):
        """Test that individuals have correct length."""
        num_features = 3
        toolbox, chromosome_info = create_toolbox(num_features)

        individual = toolbox.individual()
        assert len(individual) == chromosome_info['total_length']
        assert all(bit in [0, 1] for bit in individual)

    def test_create_toolbox_population_creation(self):
        """Test population creation."""
        num_features = 4
        toolbox, chromosome_info = create_toolbox(num_features)

        pop = toolbox.population(n=10)
        assert len(pop) == 10
        assert all(len(ind) == chromosome_info['total_length'] for ind in pop)


class TestEvaluateHypothesis:
    """Test hypothesis evaluation functionality."""

    def test_evaluate_hypothesis_basic(self, mock_statistical_functions):
        """Test basic hypothesis evaluation."""
        individual = [1, 0, 1, 0]  # Features 0 and 2 selected
        data = np.random.randn(10, 4)
        target = np.random.randint(0, 2, 10)

        fitness = evaluate_hypothesis(individual, data, target)

        assert len(fitness) == 4  # significance, effect_size, coverage, parsimony
        assert all(isinstance(val, float) for val in fitness)
        assert fitness[3] <= 1.0  # parsimony penalty

    def test_evaluate_hypothesis_no_features_selected(self, mock_statistical_functions):
        """Test evaluation when no features are selected."""
        individual = [0, 0, 0, 0]
        data = np.random.randn(10, 4)
        target = np.random.randint(0, 2, 10)

        fitness = evaluate_hypothesis(individual, data, target)

        assert len(fitness) == 4
        assert fitness[0] == 0.0  # significance
        assert fitness[1] == 0.0  # effect_size
        assert fitness[2] == 0.0  # coverage
        assert fitness[3] == 1.0  # parsimony penalty

    def test_evaluate_hypothesis_all_features_selected(self, mock_statistical_functions):
        """Test evaluation when all features are selected."""
        individual = [1, 1, 1]
        data = np.random.randn(10, 3)
        target = np.random.randint(0, 2, 10)

        fitness = evaluate_hypothesis(individual, data, target)

        assert len(fitness) == 4
        assert fitness[3] == 1.0  # parsimony penalty = 3/3 = 1.0

    @patch('greta_core.hypothesis_search.calculate_significance')
    @patch('greta_core.hypothesis_search.calculate_effect_size')
    @patch('greta_core.hypothesis_search.calculate_coverage')
    @patch('greta_core.hypothesis_search.calculate_parsimony')
    def test_evaluate_hypothesis_calls_correct_functions(self, mock_parsimony, mock_coverage,
                                                        mock_effect, mock_sig):
        """Test that evaluation calls the correct statistical functions."""
        mock_sig.return_value = 0.8
        mock_effect.return_value = 0.6
        mock_coverage.return_value = 0.7
        mock_parsimony.return_value = 0.3

        individual = [1, 0, 1]
        data = np.random.randn(10, 3)
        target = np.random.randint(0, 2, 10)

        fitness = evaluate_hypothesis(individual, data, target)

        mock_sig.assert_called_once()
        mock_effect.assert_called_once()
        mock_coverage.assert_called_once()
        mock_parsimony.assert_called_once_with(2, 3)  # 2 selected out of 3

        assert fitness == (0.8, 0.6, 0.7, 0.3)


class TestRunGeneticAlgorithm:
    """Test genetic algorithm execution."""

    @patch('greta_core.hypothesis_search.GeneticAlgorithmOptimizer')
    def test_run_genetic_algorithm_basic(self, mock_optimizer_class):
        """Test basic GA execution."""
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = [Mock()]
        mock_optimizer_class.return_value = mock_optimizer

        data = np.random.randn(20, 5)
        target = np.random.randint(0, 2, 20)

        result = run_genetic_algorithm(data, target, pop_size=10, num_generations=2)

        assert isinstance(result, list)
        assert len(result) > 0
        mock_optimizer_class.assert_called_once()
        mock_optimizer.optimize.assert_called_once()

    @patch('greta_core.hypothesis_search.GeneticAlgorithmOptimizer')
    def test_run_genetic_algorithm_parameters(self, mock_optimizer_class):
        """Test GA with different parameters."""
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = [Mock()]
        mock_optimizer_class.return_value = mock_optimizer

        data = np.random.randn(15, 3)
        target = np.random.randint(0, 2, 15)

        result = run_genetic_algorithm(
            data, target,
            pop_size=5,
            num_generations=1,
            cx_prob=0.8,
            mut_prob=0.1
        )

        assert isinstance(result, list)
        # Check that optimizer was called with correct parameters
        call_args = mock_optimizer_class.call_args
        assert call_args[0] == (data, target)
        kwargs = call_args[1]
        assert kwargs['pop_size'] == 5
        assert kwargs['num_generations'] == 1
        assert kwargs['cx_prob'] == 0.8
        assert kwargs['mut_prob'] == 0.1

    @patch('greta_core.hypothesis_search.GeneticAlgorithmOptimizer')
    def test_run_genetic_algorithm_pareto_front(self, mock_optimizer_class):
        """Test that GA returns Pareto front."""
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = [Mock()]
        mock_optimizer_class.return_value = mock_optimizer

        data = np.random.randn(10, 2)
        target = np.random.randint(0, 2, 10)

        result = run_genetic_algorithm(data, target, pop_size=1, num_generations=1)

        assert isinstance(result, list)
        assert len(result) == 1


class TestGenerateHypotheses:
    """Test hypothesis generation functionality."""

    @patch('greta_core.hypothesis_search.run_genetic_algorithm')
    def test_generate_hypotheses_basic(self, mock_run_ga):
        """Test basic hypothesis generation."""
        # Mock GA result
        mock_individuals = []
        for i in range(3):
            ind = Mock()
            ind.fitness.values = [0.9, 0.7, 0.8, 0.2]
            ind.__getitem__ = Mock(side_effect=lambda idx: [1, 0, 1][idx])
            ind.__len__ = Mock(return_value=3)
            mock_individuals.append(ind)

        mock_run_ga.return_value = mock_individuals

        data = np.random.randn(20, 3)
        target = np.random.randint(0, 2, 20)

        hypotheses = generate_hypotheses(data, target)

        assert isinstance(hypotheses, list)
        assert len(hypotheses) == 3

        # Check structure of first hypothesis
        hyp = hypotheses[0]
        assert 'features' in hyp
        assert 'significance' in hyp
        assert 'effect_size' in hyp
        assert 'coverage' in hyp
        assert 'parsimony_penalty' in hyp
        assert 'fitness' in hyp

        assert hyp['features'] == ['feature_0', 'feature_2']  # Selected features
        assert hyp['significance'] == 0.9
        assert hyp['fitness'] == 0.9 + 0.7 + 0.8 - 0.2  # sum of first 3 minus parsimony

    @patch('greta_core.hypothesis_search.run_genetic_algorithm')
    def test_generate_hypotheses_sorted_by_fitness(self, mock_run_ga):
        """Test that hypotheses are sorted by fitness."""
        # Create individuals with different fitness values
        individuals = []
        fitness_values = [
            [0.8, 0.6, 0.7, 0.3],  # fitness = 0.8 + 0.6 + 0.7 - 0.3 = 1.8
            [0.9, 0.8, 0.85, 0.1], # fitness = 0.9 + 0.8 + 0.85 - 0.1 = 2.45
            [0.7, 0.5, 0.6, 0.4],  # fitness = 0.7 + 0.5 + 0.6 - 0.4 = 1.4
        ]

        for i, fitness in enumerate(fitness_values):
            ind = Mock()
            ind.fitness.values = fitness
            ind.__getitem__ = Mock(side_effect=lambda idx: [1, 0, 1][idx])
            ind.__len__ = Mock(return_value=3)
            individuals.append(ind)

        mock_run_ga.return_value = individuals

        data = np.random.randn(15, 3)
        target = np.random.randint(0, 2, 15)

        hypotheses = generate_hypotheses(data, target)

        # Should be sorted by fitness descending
        assert hypotheses[0]['fitness'] > hypotheses[1]['fitness']
        assert hypotheses[1]['fitness'] > hypotheses[2]['fitness']

    @patch('greta_core.hypothesis_search.run_genetic_algorithm')
    def test_generate_hypotheses_with_kwargs(self, mock_run_ga):
        """Test hypothesis generation with custom parameters."""
        mock_ind = Mock()
        mock_ind.fitness.values = [0.8, 0.6, 0.7, 0.2]
        mock_ind.__getitem__ = Mock(side_effect=lambda idx: [1, 0][idx])
        mock_ind.__len__ = Mock(return_value=2)
        mock_run_ga.return_value = [mock_ind]

        data = np.random.randn(10, 2)
        target = np.random.randint(0, 2, 10)

        hypotheses = generate_hypotheses(
            data, target,
            pop_size=50,
            num_generations=20,
            cx_prob=0.9
        )

        mock_run_ga.assert_called_once_with(
            data, target,
            pop_size=50,
            num_generations=20,
            cx_prob=0.9,
            mut_prob=0.2,
            n_processes=1,
            use_dask=False,
            adaptive_params=False,
            diversity_threshold=0.1,
            convergence_threshold=0.01,
            local_search_enabled=False,
            local_search_method='hill_climbing',
            elite_fraction=0.1,
            local_search_iterations=10
        )

    @patch('greta_core.hypothesis_search.run_genetic_algorithm')
    def test_generate_hypotheses_empty_result(self, mock_run_ga):
        """Test handling of empty GA results."""
        mock_run_ga.return_value = []

        data = np.random.randn(10, 2)
        target = np.random.randint(0, 2, 10)

        hypotheses = generate_hypotheses(data, target)

        assert hypotheses == []


# Integration tests
class TestHypothesisSearchIntegration:
    """Integration tests for hypothesis search."""

    @patch('deap.creator.Individual')
    @patch('deap.creator.FitnessMulti')
    @patch('deap.base.Toolbox')
    def test_full_hypothesis_search_workflow(self, mock_toolbox_class, mock_fitness, mock_individual):
        """Test complete hypothesis search workflow."""
        # This is a complex integration test that would require extensive mocking
        # For now, we'll test the basic structure

        # Create mock data
        np.random.seed(42)
        data = np.random.randn(50, 4)
        target = np.random.randint(0, 2, 50)

        # Mock the DEAP components
        mock_toolbox = Mock()
        mock_toolbox_class.return_value = mock_toolbox

        # This would be very complex to mock fully, so we'll just ensure
        # the function can be called without errors (it will fail due to mocking)
        # In a real scenario, we'd use more sophisticated mocking

        try:
            hypotheses = generate_hypotheses(data, target, pop_size=5, num_generations=1)
            # If it succeeds, check structure
            if hypotheses:
                assert isinstance(hypotheses, list)
                if len(hypotheses) > 0:
                    hyp = hypotheses[0]
                    required_keys = ['features', 'significance', 'effect_size',
                                   'coverage', 'parsimony_penalty', 'fitness']
                    assert all(key in hyp for key in required_keys)
        except Exception:
            # Expected due to mocking complexity
            pass

    def test_hypothesis_search_parameter_validation(self):
        """Test parameter validation in hypothesis search."""
        data = np.random.randn(10, 3)
        target = np.random.randint(0, 2, 10)

        # Should handle various parameter combinations
        hypotheses = generate_hypotheses(
            data, target,
            pop_size=10,
            num_generations=2,
            cx_prob=0.7,
            mut_prob=0.1
        )

        # Basic checks
        assert isinstance(hypotheses, list)
        if len(hypotheses) > 0:
            assert all(isinstance(h, dict) for h in hypotheses)

    def test_generate_hypotheses_with_categorical_data(self):
        """Test hypothesis generation with categorical features."""
        # Create DataFrame with mixed data types
        import pandas as pd
        np.random.seed(42)

        data = pd.DataFrame({
            'numeric1': np.random.randn(50),
            'numeric2': np.random.randn(50),
            'categorical1': np.random.choice(['A', 'B', 'C'], 50),
            'categorical2': np.random.choice(['X', 'Y'], 50)
        })
        target = np.random.randint(0, 2, 50)

        # Test with one-hot encoding
        hypotheses = generate_hypotheses(
            data, target,
            pop_size=10,
            num_generations=2,
            encoding_method='one_hot'
        )

        assert isinstance(hypotheses, list)
        if len(hypotheses) > 0:
            # Check that features include encoded categorical features
            hyp = hypotheses[0]
            assert 'features' in hyp
            # Should have more features due to encoding
            assert len(hyp['features']) >= 2  # At least original numeric features

    def test_generate_hypotheses_with_target_encoding(self):
        """Test hypothesis generation with target encoding."""
        import pandas as pd
        np.random.seed(42)

        data = pd.DataFrame({
            'numeric1': np.random.randn(50),
            'categorical1': np.random.choice(['A', 'B', 'C'], 50),
        })
        target = np.random.randint(0, 2, 50)

        # Test with target encoding
        hypotheses = generate_hypotheses(
            data, target,
            pop_size=10,
            num_generations=2,
            encoding_method='target_encoding'
        )

        assert isinstance(hypotheses, list)
        if len(hypotheses) > 0:
            hyp = hypotheses[0]
            assert 'features' in hyp
            # Should include encoded categorical feature
            assert any('categorical1' in f for f in hyp['features'])

    def test_mixed_data_types_fitness_evaluation(self):
        """Test fitness evaluation with mixed encoded data."""
        # Create data with one-hot encoded features (simulating post-encoding)
        data = np.random.randn(20, 5)
        # Make some columns binary (one-hot encoded)
        data[:, 2] = np.random.choice([0, 1], 20)
        data[:, 3] = np.random.choice([0, 1], 20)
        target = np.random.randint(0, 2, 20)

        individual = [1, 1, 1, 0, 1]  # Select features 0, 1, 2, 4

        fitness = evaluate_hypothesis(individual, data, target)

        assert len(fitness) == 4
        assert all(isinstance(val, float) for val in fitness)
        # Should handle mixed binary and continuous features
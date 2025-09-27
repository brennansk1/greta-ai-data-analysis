"""
Comprehensive tests for statistical_analysis module using pytest.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from greta_core.significance_tests import (
    perform_t_test, perform_anova, calculate_significance,
    calculate_effect_size, calculate_coverage, calculate_parsimony,
    perform_statistical_test, get_target_type
)


class TestPerformTTest:
    """Test t-test functionality."""

    def test_perform_t_test_basic(self):
        """Test basic t-test between two groups."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([2, 3, 4, 5, 6])

        t_stat, p_value = perform_t_test(group1, group2)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_perform_t_test_different_means(self):
        """Test t-test with clearly different groups."""
        group1 = np.array([1, 2, 3])
        group2 = np.array([10, 11, 12])

        t_stat, p_value = perform_t_test(group1, group2)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert p_value < 0.05  # Should be significant

    def test_perform_t_test_identical_groups(self):
        """Test t-test with identical groups."""
        group1 = np.array([1, 2, 3])
        group2 = np.array([1, 2, 3])

        t_stat, p_value = perform_t_test(group1, group2)

        assert t_stat == 0.0
        assert p_value == 1.0

    def test_perform_t_test_single_value_groups(self):
        """Test t-test with single values."""
        group1 = np.array([5])
        group2 = np.array([10])

        t_stat, p_value = perform_t_test(group1, group2)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)

    @pytest.mark.parametrize("group1,group2", [
        ([], [1, 2, 3]),
        ([1, 2, 3], []),
        ([], [])
    ])
    def test_perform_t_test_edge_cases(self, group1, group2):
        """Test t-test with edge cases."""
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            perform_t_test(np.array(group1), np.array(group2))


class TestPerformAnova:
    """Test ANOVA functionality."""

    def test_perform_anova_basic(self):
        """Test basic ANOVA with three groups."""
        group1 = np.array([1, 2, 3])
        group2 = np.array([4, 5, 6])
        group3 = np.array([7, 8, 9])

        f_stat, p_value = perform_anova(group1, group2, group3)

        assert isinstance(f_stat, float)
        assert isinstance(p_value, float)
        assert f_stat > 0
        assert 0 <= p_value <= 1

    def test_perform_anova_two_groups(self):
        """Test ANOVA with only two groups."""
        group1 = np.array([1, 2, 3])
        group2 = np.array([4, 5, 6])

        f_stat, p_value = perform_anova(group1, group2)

        assert isinstance(f_stat, float)
        assert isinstance(p_value, float)

    def test_perform_anova_identical_groups(self):
        """Test ANOVA with identical groups."""
        group1 = np.array([5, 5, 5])
        group2 = np.array([5, 5, 5])
        group3 = np.array([5, 5, 5])

        f_stat, p_value = perform_anova(group1, group2, group3)

        assert f_stat == 0.0
        assert p_value == 1.0

    def test_perform_anova_single_group(self):
        """Test ANOVA with single group."""
class TestGetTargetType:
    """Test get_target_type functionality."""

    def test_get_target_type_categorical_binary(self):
        """Test detection of binary categorical target."""
        y = np.array([0, 1, 0, 1])
        assert get_target_type(y) == 'categorical'

    def test_get_target_type_categorical_multiclass(self):
        """Test detection of multi-class categorical target."""
        y = np.array([0, 1, 2, 1, 0])
        assert get_target_type(y) == 'categorical'

    def test_get_target_type_continuous(self):
        """Test detection of continuous target."""
        y = np.random.randn(50)
        assert get_target_type(y) == 'continuous'

    def test_get_target_type_threshold(self):
        """Test threshold for categorical vs continuous."""
        y = np.arange(4)  # 4 unique, should be continuous
        assert get_target_type(y) == 'continuous'

        y = np.arange(3)  # 3 unique, should be categorical
        assert get_target_type(y) == 'categorical'

    def test_get_target_type_custom_threshold(self):
        """Test custom threshold."""
        y = np.arange(5)
        assert get_target_type(y, threshold=3) == 'continuous'
        assert get_target_type(y, threshold=6) == 'categorical'


class TestCalculateSignificance:
    """Test significance calculation functionality."""

    def test_calculate_significance_binary_target(self):
        """Test significance with binary target."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        sig = calculate_significance(X, y)

        assert isinstance(sig, float)
        assert 0 <= sig <= 1

    def test_calculate_significance_single_feature(self):
        """Test significance with single feature."""
    def test_calculate_significance_categorical_multiclass_single_feature(self):
        """Test significance with multi-class categorical target, single feature."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 1, 2, 1, 0])

        sig = calculate_significance(X, y)

        assert isinstance(sig, float)
        assert 0 <= sig <= 1

    def test_calculate_significance_categorical_multiclass_multiple_features(self):
        """Test significance with multi-class categorical target, multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 2, 1, 0])

        sig = calculate_significance(X, y)

        assert isinstance(sig, float)
        assert 0 <= sig <= 1

    def test_calculate_significance_continuous_many_unique(self):
        """Test significance with continuous target with many unique values."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        sig = calculate_significance(X, y)

        assert isinstance(sig, float)
        assert 0 <= sig <= 1


        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])

        sig = calculate_significance(X, y)

        assert isinstance(sig, float)
        assert 0 <= sig <= 1

    def test_calculate_effect_size_categorical_binary_single_feature(self):
        """Test effect size with binary categorical target, single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])

        effect = calculate_effect_size(X, y)

        assert isinstance(effect, float)
        assert effect >= 0

    def test_calculate_effect_size_categorical_multiclass_single_feature(self):
        """Test effect size with multi-class categorical target, single feature."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 1, 2, 1, 0])

        effect = calculate_effect_size(X, y)

        assert isinstance(effect, float)
        assert effect >= 0

    def test_calculate_effect_size_categorical_multiple_features(self):
        """Test effect size with categorical target, multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        effect = calculate_effect_size(X, y)

        assert isinstance(effect, float)
        assert 0 <= effect <= 1  # accuracy


    def test_calculate_significance_multiclass_target(self):
        """Test significance with multi-class target."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 2, 1, 0])

        sig = calculate_significance(X, y)

        assert isinstance(sig, float)
        assert 0 <= sig <= 1
    def test_calculate_coverage_categorical_binary(self):
        """Test coverage with binary categorical target."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        coverage = calculate_coverage(X, y)

        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1

    def test_calculate_coverage_categorical_multiclass(self):
        """Test coverage with multi-class categorical target."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 2, 1])

        coverage = calculate_coverage(X, y)

        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1



    def test_calculate_significance_perfect_correlation(self):
        """Test significance with perfect correlation."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 2, 3, 4])  # Perfect positive correlation

        sig = calculate_significance(X, y)

        assert sig > 0.9  # Should be highly significant

    def test_calculate_significance_no_correlation(self):
        """Test significance with no correlation."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)  # Random, no correlation

        sig = calculate_significance(X, y)

        assert isinstance(sig, float)
        # May not be exactly 0 due to random chance


class TestCalculateEffectSize:
    """Test effect size calculation functionality."""

    def test_calculate_effect_size_single_feature(self):
        """Test effect size with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 2, 3, 4])

        effect = calculate_effect_size(X, y)

        assert isinstance(effect, float)
        assert effect > 0  # Positive correlation

    def test_calculate_effect_size_multiple_features(self):
        """Test effect size with multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])

        effect = calculate_effect_size(X, y)

        assert isinstance(effect, float)
        assert effect >= 0

    def test_calculate_effect_size_no_relationship(self):
        """Test effect size with no relationship."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        effect = calculate_effect_size(X, y)

        assert isinstance(effect, float)
    def test_perform_statistical_test_auto_categorical_binary(self):
        """Test auto test selection with binary categorical target."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])

        result = perform_statistical_test(X, y, test_type='auto')

        assert result['test'] == 't_test'

    def test_perform_statistical_test_auto_categorical_multiclass(self):
        """Test auto test selection with multi-class categorical target."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 1, 2, 1, 0])

        result = perform_statistical_test(X, y, test_type='auto')

        assert result['test'] == 'anova'

    def test_perform_statistical_test_auto_continuous(self):
        """Test auto test selection with continuous target."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = np.random.randn(20)

        result = perform_statistical_test(X, y, test_type='auto')

        assert result['test'] == 'regression'

    def test_perform_statistical_test_logistic_binary(self):
        """Test logistic regression for binary categorical with multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        result = perform_statistical_test(X, y, test_type='t_test')

        assert result['test'] == 'logistic'
        assert 'accuracy' in result

    def test_perform_statistical_test_logistic_multiclass(self):
        """Test logistic regression for multi-class categorical with multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 2, 1])

        result = perform_statistical_test(X, y, test_type='anova')

        assert result['test'] == 'logistic'
        assert 'accuracy' in result


class TestCalculateCoverage:
    """Test coverage (R-squared) calculation functionality."""

    def test_calculate_coverage_perfect_fit(self):
        """Test coverage with perfect linear relationship."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])  # Perfect linear relationship

        coverage = calculate_coverage(X, y)

        assert isinstance(coverage, float)
        assert coverage > 0.99  # Should be very close to 1

    def test_calculate_coverage_no_relationship(self):
        """Test coverage with no relationship."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        coverage = calculate_coverage(X, y)

        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1

    def test_calculate_coverage_multiple_features(self):
        """Test coverage with multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])

        coverage = calculate_coverage(X, y)

        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1


class TestCalculateParsimony:
    """Test parsimony calculation functionality."""

    def test_calculate_parsimony_basic(self):
        """Test basic parsimony calculation."""
        parsimony = calculate_parsimony(3, 10)  # 3 selected out of 10

        assert isinstance(parsimony, float)
        assert parsimony == 0.3

    def test_calculate_parsimony_all_features(self):
        """Test parsimony with all features selected."""
        parsimony = calculate_parsimony(5, 5)

        assert parsimony == 1.0

    def test_calculate_parsimony_no_features(self):
        """Test parsimony with no features selected."""
        parsimony = calculate_parsimony(0, 5)

        assert parsimony == 0.0

    def test_calculate_parsimony_zero_total(self):
        """Test parsimony with zero total features."""
        parsimony = calculate_parsimony(0, 0)

        assert parsimony == 0.0


class TestPerformStatisticalTest:
    """Test statistical test performance functionality."""

    def test_perform_statistical_test_t_test(self):
        """Test statistical test with t-test."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])

        result = perform_statistical_test(X, y, test_type='t_test')

        assert 'test' in result
        assert result['test'] == 't_test'
        assert 't_stat' in result
        assert 'p_value' in result

    def test_perform_statistical_test_anova(self):
        """Test statistical test with ANOVA."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 1, 2, 1, 0])

        result = perform_statistical_test(X, y, test_type='anova')

        assert 'test' in result
        assert result['test'] == 'anova'
        assert 'f_stat' in result
        assert 'p_value' in result

    def test_perform_statistical_test_auto_binary(self):
        """Test auto test selection with binary target."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])

        result = perform_statistical_test(X, y, test_type='auto')

        assert result['test'] == 't_test'

    def test_perform_statistical_test_auto_multiclass(self):
        """Test auto test selection with multi-class target."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 1, 2, 1, 0])

        result = perform_statistical_test(X, y, test_type='auto')

        assert result['test'] == 'anova'

    def test_perform_statistical_test_regression_fallback(self):
        """Test regression fallback for complex cases."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        result = perform_statistical_test(X, y, test_type='t_test')

        assert result['test'] == 'logistic'
        assert 'accuracy' in result


# Integration tests
class TestStatisticalAnalysisIntegration:
    """Integration tests for statistical analysis."""

    def test_full_statistical_pipeline(self):
        """Test complete statistical analysis pipeline."""
        # Create test data
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] + np.random.randn(50) * 0.1  # Some correlation with first feature

        # Test all functions
        sig = calculate_significance(X, y)
        effect = calculate_effect_size(X, y)
        coverage = calculate_coverage(X, y)
        parsimony = calculate_parsimony(2, 3)

        assert all(isinstance(val, float) for val in [sig, effect, coverage, parsimony])
        assert all(0 <= val <= 1 for val in [sig, coverage, parsimony])
        assert effect >= 0

    def test_statistical_analysis_edge_cases(self):
        """Test statistical analysis with edge cases."""
        # Single sample
        X = np.array([[1]])
        y = np.array([1])

        sig = calculate_significance(X, y)
        assert isinstance(sig, float)

        # All same values
        X = np.array([[1], [1], [1]])
        y = np.array([1, 1, 1])

        sig = calculate_significance(X, y)
        assert isinstance(sig, float)
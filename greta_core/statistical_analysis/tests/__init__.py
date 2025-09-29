"""
Statistical Tests Module

Contains functions for performing various statistical tests and hypothesis evaluation.
"""

from .parametric_tests import perform_t_test, perform_anova
from .nonparametric_tests import perform_mann_whitney, perform_kruskal_wallis, perform_permutation_test
from .hypothesis_evaluation import (
    get_target_type, detect_feature_types_from_matrix, calculate_significance,
    calculate_effect_size, calculate_coverage, calculate_parsimony, perform_statistical_test
)

__all__ = [
    'perform_t_test', 'perform_anova', 'perform_mann_whitney', 'perform_kruskal_wallis',
    'perform_permutation_test', 'get_target_type', 'detect_feature_types_from_matrix',
    'calculate_significance', 'calculate_effect_size', 'calculate_coverage',
    'calculate_parsimony', 'perform_statistical_test'
]
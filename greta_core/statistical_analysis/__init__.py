"""
Statistical Analysis Module

Contains functions for performing various statistical analyses including regression, causal inference, and time series analysis.
"""

from .causal_analysis import *
from .regression_analysis import *
from .time_series_analysis import *
from .tests import *

__all__ = [
    # From causal_analysis
    'define_causal_model', 'identify_confounders', 'estimate_causal_effect',
    'perform_causal_analysis', 'validate_causal_assumptions',
    # From regression_analysis
    'perform_multiple_linear_regression', 'perform_random_forest_regression',
    'perform_regression', 'perform_xgboost_regression',
    # From time_series_analysis
    'detect_trend', 'detect_seasonality', 'basic_forecast',
    # From tests
    'perform_t_test', 'perform_anova', 'perform_mann_whitney', 'perform_kruskal_wallis',
    'perform_permutation_test', 'get_target_type', 'detect_feature_types_from_matrix',
    'calculate_significance', 'calculate_effect_size', 'calculate_coverage',
    'calculate_parsimony', 'perform_statistical_test'
]
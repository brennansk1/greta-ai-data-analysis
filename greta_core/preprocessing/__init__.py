"""
Preprocessing Module

Contains functions for data preprocessing including profiling, normalization, missing value handling, outlier detection, and feature engineering.
"""

from .profiling_stats import profile_data, identify_identifier_columns, detect_feature_types
from .data_normalization import normalize_data_types
from .missing_value_handling import handle_missing_values
from .outlier_detection import detect_outliers, remove_outliers
from .feature_engineering import (
    basic_feature_engineering, apply_categorical_encoding, prepare_features_for_modeling,
    encode_one_hot, encode_label, encode_ordinal, encode_frequency, encode_target_mean,
    generate_polynomial_features, generate_trigonometric_features, generate_logarithmic_features,
    generate_binning_features, generate_interaction_features, generate_statistical_features
)
from .feature_selection import (
    select_features, select_by_correlation, select_by_mutual_info, select_by_univariate_test,
    select_by_rfe, select_by_importance
)
from .automated_feature_generation import generate_features_automated, AutomatedFeatureGenerator
from .importance_analysis import (
    analyze_feature_importance, compute_shap_importance, compute_permutation_importance,
    compute_model_coefficients, get_top_important_features
)

__all__ = [
    # Original functions
    'profile_data', 'identify_identifier_columns', 'detect_feature_types',
    'normalize_data_types', 'handle_missing_values', 'detect_outliers',
    'remove_outliers', 'basic_feature_engineering', 'apply_categorical_encoding',
    'prepare_features_for_modeling',
    # New encoding functions
    'encode_one_hot', 'encode_label', 'encode_ordinal', 'encode_frequency', 'encode_target_mean',
    # New feature generation functions
    'generate_polynomial_features', 'generate_trigonometric_features', 'generate_logarithmic_features',
    'generate_binning_features', 'generate_interaction_features', 'generate_statistical_features',
    # Feature selection
    'select_features', 'select_by_correlation', 'select_by_mutual_info', 'select_by_univariate_test',
    'select_by_rfe', 'select_by_importance',
    # Automated feature generation
    'generate_features_automated', 'AutomatedFeatureGenerator',
    # Importance analysis
    'analyze_feature_importance', 'compute_shap_importance', 'compute_permutation_importance',
    'compute_model_coefficients', 'get_top_important_features'
]
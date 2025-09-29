"""
Regression Narratives Module

Contains functions for generating narratives from regression analysis results.
"""

from typing import Dict, List, Any
import numpy as np
from ..statistical_analysis import perform_multiple_linear_regression


def generate_regression_narrative(hypothesis: Dict[str, Any], feature_names: List[str], data: np.ndarray, target: np.ndarray, confidence: str, effect: str, coverage_desc: str) -> str:
    """
    Generate narrative for regression analysis.

    Args:
        hypothesis: Hypothesis details.
        feature_names: Feature names.
        data: Feature matrix.
        target: Target variable.
        confidence: Confidence level string.
        effect: Effect strength string.
        coverage_desc: Coverage description.

    Returns:
        Regression narrative.
    """
    # hypothesis['features'] now contains feature names directly
    features = hypothesis['features']

    if data is None or target is None:
        # Fallback to basic narrative if data not available
        if len(features) == 1:
            narrative = f"The analysis suggests that {features[0]} has a {effect} relationship with the target variable. "
        else:
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            narrative = f"The combination of {feature_list} shows a {effect} relationship with the target variable. "
        narrative += f"This finding has {confidence} statistical confidence and {coverage_desc} in the data."
        return narrative

    # For regression analysis, we need to map feature names back to indices
    feature_indices = [feature_names.index(feat) for feat in features]
    selected_data = data[:, feature_indices]

    # Perform regression to get detailed results
    reg_results = perform_multiple_linear_regression(selected_data, target)
    coefficients = reg_results['coefficients']
    p_values = reg_results['p_values']
    adj_r_squared = reg_results['adj_r_squared']

    # Identify significant predictors
    significant_features = [features[i] for i, p in enumerate(p_values) if p < 0.05]

    if len(features) == 1:
        coeff_desc = f"with a coefficient of {coefficients[0]:.3f}"
        narrative = f"Using advanced feature engineering and selection methods (including automated generation, encoding techniques, and statistical selection), multiple linear regression shows that {features[0]} has a {effect} linear relationship with the target variable {coeff_desc}. "
    else:
        feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
        narrative = f"Using advanced feature engineering and selection methods (including automated generation, encoding techniques, and statistical selection), multiple linear regression analysis of {feature_list} reveals a {effect} relationship with the target variable. "

        if significant_features:
            sig_list = ", ".join(significant_features[:-1]) + f" and {significant_features[-1]}" if len(significant_features) > 1 else significant_features[0]
            narrative += f"The most significant predictors are {sig_list}. "

    narrative += f"The model has {confidence} statistical confidence, explains {adj_r_squared:.1%} of the variance in the target variable, and {coverage_desc}."

    return narrative
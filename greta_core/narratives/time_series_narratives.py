"""
Time Series Narratives Module

Contains functions for generating narratives from time series analysis results.
"""

from typing import Dict, List, Any
import numpy as np
from ..statistical_analysis import detect_trend, detect_seasonality, perform_multiple_linear_regression


def generate_time_series_narrative(hypothesis: Dict[str, Any], feature_names: List[str], data: np.ndarray, target: np.ndarray, confidence: str, effect: str, coverage_desc: str) -> str:
    """
    Generate narrative for time-series analysis.

    Args:
        hypothesis: Hypothesis details.
        feature_names: Feature names.
        data: Feature matrix.
        target: Target variable.
        confidence: Confidence level string.
        effect: Effect strength string.
        coverage_desc: Coverage description.

    Returns:
        Time-series narrative.
    """
    if target is None:
        # Fallback to basic narrative if target not available
        features = [feature_names[i] for i in hypothesis['features']]
        if len(features) == 1:
            narrative = f"The analysis suggests that {features[0]} has a {effect} relationship with the target variable. "
        else:
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            narrative = f"The combination of {feature_list} shows a {effect} relationship with the target variable. "
        narrative += f"We're {confidence} about this finding and it {coverage_desc} in the data."
        return narrative

    features = [feature_names[i] for i in hypothesis['features']]

    if len(features) == 0:
        # Analyze trend in target
        trend_results = detect_trend(target)
        slope = trend_results['slope']
        trend_strength = trend_results['trend_strength']

        direction = "increasing" if slope > 0 else "decreasing"
        strength_desc = "strong" if trend_strength > 0.5 else "moderate" if trend_strength > 0.3 else "weak"

        narrative = f"Using advanced feature engineering techniques (including automated transformations and encoding), time-series analysis of the target variable shows a {strength_desc} {direction} trend over time. "
        narrative += f"We're {confidence} about this trend and it {coverage_desc}."
    else:
        if data is None:
            # Fallback if data not available
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            verb = "has" if len(features) == 1 else "have"
            narrative = f"Using enhanced feature engineering and selection methods (including automated generation and advanced transformations), analysis of trends over time suggests that {feature_list} {verb} a {effect} relationship with the target variable. "
            narrative += f"We're {confidence} about this finding and it {coverage_desc}."
        else:
            # Regression on time-series predictors
            selected_data = data[:, hypothesis['features']]
            reg_results = perform_multiple_linear_regression(selected_data, target)
            adj_r_squared = reg_results['adj_r_squared']

            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            verb = "has" if len(features) == 1 else "have"
            narrative = f"Using advanced feature engineering and selection methods (including automated generation, encoding techniques, and statistical selection), analysis of trends over time shows that {feature_list} {verb} a {effect} relationship with the target variable. "
            narrative += f"We're {confidence} about this model, which explains {adj_r_squared:.1%} of what's happening in the target variable, and {coverage_desc}."

            # Check for seasonality if possible
            if len(target) > 12:
                seasonal_results = detect_seasonality(target)
                if seasonal_results['seasonal_strength'] > 0.1:
                    narrative += f" Additionally, there appears to be seasonal patterns in the data."

    return narrative
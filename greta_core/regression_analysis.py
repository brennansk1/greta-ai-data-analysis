"""
Regression Analysis Module

Contains functions for performing regression analysis and related statistical modeling.
"""

import numpy as np
import statsmodels.api as sm
from typing import Union, Tuple, List, Dict


def perform_multiple_linear_regression(X: np.ndarray, y: np.ndarray) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
    Perform multiple linear regression analysis.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target variable (n_samples,).

    Returns:
        Dictionary with regression results: coefficients, intercept, r_squared, adj_r_squared, p_values, f_stat, f_p_value.
    """
    # Add constant for intercept
    X_with_const = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()

    # Extract results
    coefficients = model.params[1:]  # Exclude intercept
    intercept = model.params[0]
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    p_values = model.pvalues[1:]  # Exclude intercept p-value
    f_stat = model.fvalue
    f_p_value = model.f_pvalue

    return {
        'coefficients': coefficients,
        'intercept': intercept,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'p_values': p_values,
        'f_stat': f_stat,
        'f_p_value': f_p_value
    }
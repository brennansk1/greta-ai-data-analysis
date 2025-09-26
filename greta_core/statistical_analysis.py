"""
Statistical Analysis Module

Conducts rigorous statistical testing on generated hypotheses. Supports basic tests
such as t-tests and ANOVA for Phase 1, with extensible design for future additions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Union, Tuple, List, Dict


def perform_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Perform independent t-test between two groups.

    Args:
        group1: First group data.
        group2: Second group data.

    Returns:
        Tuple of (t-statistic, p-value).
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value


def perform_anova(*groups: np.ndarray) -> Tuple[float, float]:
    """
    Perform one-way ANOVA on multiple groups.

    Args:
        *groups: Variable number of group arrays.

    Returns:
        Tuple of (F-statistic, p-value).
    """
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value


def calculate_significance(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate statistical significance for a hypothesis.

    For regression: uses F-test p-value.
    For classification: uses t-test or ANOVA if target is categorical.

    Args:
        X: Feature matrix.
        y: Target variable.

    Returns:
        Significance score (1 - p_value, higher is better).
    """
    if len(np.unique(y)) <= 2:  # Binary target
        # Assume features are binary or continuous
        # For simplicity, use correlation or t-test
        if X.shape[1] == 1:
            corr, p_value = stats.pearsonr(X.flatten(), y)
            return 1 - p_value
        else:
            # Multiple features, use linear regression F-test
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            f_stat = ((ss_tot - ss_res) / X.shape[1]) / (ss_res / (len(y) - X.shape[1] - 1))
            p_value = 1 - stats.f.cdf(f_stat, X.shape[1], len(y) - X.shape[1] - 1)
            return 1 - p_value
    else:
        # Multi-class or general, use linear regression F-test
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        f_stat = ((ss_tot - ss_res) / X.shape[1]) / (ss_res / (len(y) - X.shape[1] - 1))
        p_value = 1 - stats.f.cdf(f_stat, X.shape[1], len(y) - X.shape[1] - 1)
        return 1 - p_value


def calculate_effect_size(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate effect size for a hypothesis.

    Args:
        X: Feature matrix.
        y: Target variable.

    Returns:
        Effect size (absolute value).
    """
    if X.shape[1] == 1:
        # Cohen's d for single feature
        corr, _ = stats.pearsonr(X.flatten(), y)
        return abs(corr)
    else:
        # Multiple features, use multiple R
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        return np.sqrt(r2)  # Multiple correlation coefficient


def calculate_coverage(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate coverage (proportion of variance explained).

    Args:
        X: Feature matrix.
        y: Target variable.

    Returns:
        Coverage score (R-squared).
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return r2_score(y, y_pred)


def calculate_parsimony(num_selected: int, total_features: int) -> float:
    """
    Calculate parsimony penalty.

    Args:
        num_selected: Number of selected features.
        total_features: Total number of features.

    Returns:
        Parsimony penalty (higher when more features selected).
    """
    if total_features == 0:
        return 0.0
    return num_selected / total_features


def perform_statistical_test(X: np.ndarray, y: np.ndarray, test_type: str = 'auto') -> Dict[str, float]:
    """
    Perform statistical test on hypothesis.

    Args:
        X: Feature matrix.
        y: Target variable.
        test_type: Type of test ('t_test', 'anova', 'auto').

    Returns:
        Dictionary with test results.
    """
    results = {}

    if test_type == 'auto':
        if len(np.unique(y)) == 2:
            test_type = 't_test'
        else:
            test_type = 'anova'

    if test_type == 't_test' and len(np.unique(y)) == 2:
        # Split data by target
        unique_vals = np.unique(y)
        group1 = X[y == unique_vals[0]]
        group2 = X[y == unique_vals[1]]
        if X.shape[1] == 1:
            t_stat, p_value = perform_t_test(group1.flatten(), group2.flatten())
            results = {'test': 't_test', 't_stat': t_stat, 'p_value': p_value}
        else:
            # Multiple features, use regression
            reg_results = perform_multiple_linear_regression(X, y)
            results = {'test': 'regression', 'r_squared': reg_results['adj_r_squared'], 'p_value': reg_results['f_p_value']}
    elif test_type == 'anova':
        # For ANOVA, need to group by target
        groups = [X[y == val] for val in np.unique(y)]
        if all(g.shape[1] == 1 for g in groups):
            f_stat, p_value = perform_anova(*[g.flatten() for g in groups])
            results = {'test': 'anova', 'f_stat': f_stat, 'p_value': p_value}
        else:
            reg_results = perform_multiple_linear_regression(X, y)
            results = {'test': 'regression', 'r_squared': reg_results['adj_r_squared'], 'p_value': reg_results['f_p_value']}

    return results


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


def detect_trend(time_series: np.ndarray) -> Dict[str, float]:
    """
    Detect trend in time series using linear regression.

    Args:
        time_series: Time series data.

    Returns:
        Dictionary with trend slope, intercept, r_squared, p_value.
    """
    time_index = np.arange(len(time_series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, time_series)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend_strength': abs(r_value)  # Absolute correlation as trend strength
    }


def detect_seasonality(time_series: np.ndarray, period: int = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Detect seasonality in time series using seasonal decomposition.

    Args:
        time_series: Time series data.
        period: Seasonal period (if None, will try to infer).

    Returns:
        Dictionary with seasonal strength, seasonal component.
    """
    if period is None:
        # Simple heuristic: assume monthly if length > 24, weekly if > 7, etc.
        n = len(time_series)
        if n >= 365:
            period = 365  # Daily data
        elif n >= 52:
            period = 52   # Weekly
        elif n >= 12:
            period = 12   # Monthly
        else:
            period = max(2, n // 4)  # Fallback

    try:
        decomposition = seasonal_decompose(time_series, period=period, model='additive')
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        trend = decomposition.trend

        # Calculate seasonal strength as variance explained by seasonal component
        total_var = np.var(time_series)
        seasonal_var = np.var(seasonal)
        seasonal_strength = seasonal_var / total_var if total_var > 0 else 0

        return {
            'seasonal_strength': seasonal_strength,
            'seasonal_component': seasonal,
            'trend_component': trend,
            'residual': residual,
            'period': period
        }
    except:
        # If decomposition fails, return zeros
        return {
            'seasonal_strength': 0.0,
            'seasonal_component': np.zeros_like(time_series),
            'trend_component': np.zeros_like(time_series),
            'residual': np.zeros_like(time_series),
            'period': period
        }


def basic_forecast(time_series: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Perform basic forecasting using exponential smoothing.

    Args:
        time_series: Historical time series data.
        steps: Number of steps to forecast.

    Returns:
        Forecasted values.
    """
    try:
        model = ExponentialSmoothing(time_series, trend='add', seasonal=None)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps)
        return forecast
    except:
        # Fallback to simple moving average
        if len(time_series) < 3:
            return np.full(steps, np.mean(time_series))
        ma = np.convolve(time_series, np.ones(3)/3, mode='valid')
        last_ma = ma[-1] if len(ma) > 0 else np.mean(time_series)
        return np.full(steps, last_ma)
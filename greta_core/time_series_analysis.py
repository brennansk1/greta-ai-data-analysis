"""
Time Series Analysis Module

Contains functions for analyzing time series data, including trend detection,
seasonality analysis, and forecasting.
"""

import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Union, Tuple, List, Dict


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
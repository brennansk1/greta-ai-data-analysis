"""
Regression Analysis Module

Contains functions for performing regression analysis and related statistical modeling.
Supports multiple regression methods including linear regression, Random Forest, and XGBoost,
with cross-validation and model diagnostics.
"""

import numpy as np
import statsmodels.api as sm
from typing import Union, Tuple, List, Dict
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings


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


def perform_random_forest_regression(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, random_state: int = 42, cv_folds: int = 5) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
    Perform Random Forest regression analysis with cross-validation and diagnostics.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target variable (n_samples,).
        n_estimators: Number of trees in the forest.
        random_state: Random state for reproducibility.
        cv_folds: Number of folds for cross-validation.

    Returns:
        Dictionary with regression results: cv_scores, test_mse, test_r2, feature_importances, model.
    """
    # Split data for diagnostics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2').mean()

    # Fit model on training data
    model.fit(X_train, y_train)

    # Diagnostics on test set
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # Feature importances
    feature_importances = model.feature_importances_

    return {
        'cv_mse': cv_mse,
        'cv_r2': cv_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'feature_importances': feature_importances,
        'model': model
    }


def perform_regression(X: np.ndarray, y: np.ndarray, model_type: str = 'linear', **kwargs) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
    Perform regression analysis using the specified model type.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target variable (n_samples,).
        model_type: Type of regression model ('linear', 'rf', 'xgb').
        **kwargs: Additional keyword arguments for the specific model.

    Returns:
        Dictionary with regression results depending on the model type.
    """
    if model_type == 'linear':
        return perform_multiple_linear_regression(X, y)
    elif model_type == 'rf':
        return perform_random_forest_regression(X, y, **kwargs)
    elif model_type == 'xgb':
        return perform_xgboost_regression(X, y, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'linear', 'rf', 'xgb'")


def perform_xgboost_regression(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42, cv_folds: int = 5) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
    Perform XGBoost regression analysis with cross-validation and diagnostics.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target variable (n_samples,).
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage used in update to prevents overfitting.
        random_state: Random state for reproducibility.
        cv_folds: Number of folds for cross-validation.

    Returns:
        Dictionary with regression results: cv_scores, test_mse, test_r2, feature_importances, model.
    """
    # Suppress XGBoost warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

    # Split data for diagnostics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize model
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2').mean()

    # Fit model on training data
    model.fit(X_train, y_train)

    # Diagnostics on test set
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # Feature importances
    feature_importances = model.feature_importances_

    return {
        'cv_mse': cv_mse,
        'cv_r2': cv_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'feature_importances': feature_importances,
        'model': model
    }
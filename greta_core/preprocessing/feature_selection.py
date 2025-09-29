"""
Feature Selection Module

Contains functions for selecting the most relevant features using various methods.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Optional
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2, RFE, RFECV
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import warnings

# Type alias for DataFrame
DataFrame = Union[pd.DataFrame, np.ndarray]


def select_by_correlation(X: DataFrame, y: Union[np.ndarray, pd.Series],
                         threshold: float = 0.8) -> Tuple[DataFrame, List[str], Dict]:
    """
    Select features based on correlation with target and multicollinearity.

    Args:
        X: Feature matrix.
        y: Target variable.
        threshold: Correlation threshold for removing multicollinear features.

    Returns:
        Tuple of (selected features, selected feature names, selection info).
    """
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    else:
        X_df = X.copy()
        feature_names = list(X.columns)

    # Correlation with target
    if isinstance(y, pd.Series):
        y_series = y
    else:
        y_series = pd.Series(y)

    correlations = X_df.corrwith(y_series).abs().sort_values(ascending=False)

    # Remove highly correlated features
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper.columns:
        if column in correlations.index and correlations[column] > 0.1:  # Keep features correlated with target
            correlated_features = upper[column][upper[column] > threshold].index.tolist()
            if correlated_features:
                # Keep the one most correlated with target
                best_feature = max(correlated_features + [column],
                                 key=lambda x: correlations.get(x, 0))
                to_drop.extend([f for f in correlated_features + [column] if f != best_feature])

    to_drop = list(set(to_drop))
    selected_features = [f for f in feature_names if f not in to_drop]

    if isinstance(X, np.ndarray):
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, selected_indices]
    else:
        X_selected = X[selected_features]

    selection_info = {
        'method': 'correlation',
        'threshold': threshold,
        'original_features': len(feature_names),
        'selected_features': len(selected_features),
        'dropped_features': to_drop,
        'target_correlations': correlations.to_dict()
    }

    return X_selected, selected_features, selection_info


def select_by_mutual_info(X: DataFrame, y: Union[np.ndarray, pd.Series],
                         k: Optional[int] = None, percentile: Optional[int] = None) -> Tuple[DataFrame, List[str], Dict]:
    """
    Select features based on mutual information with target.

    Args:
        X: Feature matrix.
        y: Target variable.
        k: Number of top features to select.
        percentile: Percentage of features to select.

    Returns:
        Tuple of (selected features, selected feature names, selection info).
    """
    if isinstance(X, np.ndarray):
        X_array = X
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    else:
        X_array = X.values
        feature_names = list(X.columns)

    # Determine target type
    if len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer):
        # Classification
        score_func = mutual_info_classif
    else:
        # Regression
        score_func = mutual_info_regression

    if k is not None:
        selector = SelectKBest(score_func=score_func, k=k)
    elif percentile is not None:
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
    else:
        selector = SelectKBest(score_func=score_func, k=min(10, X_array.shape[1]))

    X_selected = selector.fit_transform(X_array, y)
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]

    scores = selector.scores_
    feature_scores = dict(zip(feature_names, scores))

    selection_info = {
        'method': 'mutual_info',
        'k': k,
        'percentile': percentile,
        'original_features': len(feature_names),
        'selected_features': len(selected_features),
        'feature_scores': feature_scores,
        'selected_scores': {f: feature_scores[f] for f in selected_features}
    }

    return X_selected, selected_features, selection_info


def select_by_univariate_test(X: DataFrame, y: Union[np.ndarray, pd.Series],
                             k: Optional[int] = None, percentile: Optional[int] = None) -> Tuple[DataFrame, List[str], Dict]:
    """
    Select features based on univariate statistical tests.

    Args:
        X: Feature matrix.
        y: Target variable.
        k: Number of top features to select.
        percentile: Percentage of features to select.

    Returns:
        Tuple of (selected features, selected feature names, selection info).
    """
    if isinstance(X, np.ndarray):
        X_array = X
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    else:
        X_array = X.values
        feature_names = list(X.columns)

    # Determine target type and appropriate test
    unique_vals = len(np.unique(y))
    if unique_vals == 2:
        # Binary classification
        score_func = f_classif
    elif unique_vals <= 10 and np.issubdtype(y.dtype, np.integer):
        # Multiclass classification
        score_func = f_classif
    else:
        # Regression
        score_func = f_regression

    if k is not None:
        selector = SelectKBest(score_func=score_func, k=k)
    elif percentile is not None:
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
    else:
        selector = SelectKBest(score_func=score_func, k=min(10, X_array.shape[1]))

    X_selected = selector.fit_transform(X_array, y)
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]

    scores = selector.scores_
    pvalues = selector.pvalues_
    feature_scores = dict(zip(feature_names, scores))
    feature_pvalues = dict(zip(feature_names, pvalues))

    selection_info = {
        'method': 'univariate_test',
        'k': k,
        'percentile': percentile,
        'original_features': len(feature_names),
        'selected_features': len(selected_features),
        'feature_scores': feature_scores,
        'feature_pvalues': feature_pvalues,
        'selected_scores': {f: feature_scores[f] for f in selected_features},
        'selected_pvalues': {f: feature_pvalues[f] for f in selected_features}
    }

    return X_selected, selected_features, selection_info


def select_by_rfe(X: DataFrame, y: Union[np.ndarray, pd.Series],
                 estimator=None, n_features_to_select: Optional[int] = None,
                 cv: int = 5) -> Tuple[DataFrame, List[str], Dict]:
    """
    Select features using Recursive Feature Elimination (RFE).

    Args:
        X: Feature matrix.
        y: Target variable.
        estimator: Base estimator for RFE.
        n_features_to_select: Number of features to select.
        cv: Cross-validation folds for RFECV.

    Returns:
        Tuple of (selected features, selected feature names, selection info).
    """
    if isinstance(X, np.ndarray):
        X_array = X
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    else:
        X_array = X.values
        feature_names = list(X.columns)

    # Default estimator based on target type
    if estimator is None:
        unique_vals = len(np.unique(y))
        if unique_vals == 2:
            estimator = LogisticRegression(random_state=42, max_iter=1000)
        elif unique_vals <= 10 and np.issubdtype(y.dtype, np.integer):
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)

    if n_features_to_select is None:
        # Use RFECV to automatically determine number of features
        selector = RFECV(estimator, step=1, cv=cv, min_features_to_select=1)
    else:
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)

    X_selected = selector.fit_transform(X_array, y)
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]

    # Get feature rankings
    rankings = selector.ranking_
    feature_rankings = dict(zip(feature_names, rankings))

    selection_info = {
        'method': 'rfe',
        'estimator': str(type(estimator).__name__),
        'n_features_to_select': n_features_to_select,
        'cv': cv,
        'original_features': len(feature_names),
        'selected_features': len(selected_features),
        'feature_rankings': feature_rankings,
        'selected_rankings': {f: feature_rankings[f] for f in selected_features}
    }

    if hasattr(selector, 'cv_results_'):
        selection_info['cv_results'] = selector.cv_results_

    return X_selected, selected_features, selection_info


def select_by_importance(X: DataFrame, y: Union[np.ndarray, pd.Series],
                        n_estimators: int = 100, k: Optional[int] = None) -> Tuple[DataFrame, List[str], Dict]:
    """
    Select features based on tree-based feature importance.

    Args:
        X: Feature matrix.
        y: Target variable.
        n_estimators: Number of trees in the forest.
        k: Number of top features to select.

    Returns:
        Tuple of (selected features, selected feature names, selection info).
    """
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    else:
        X_df = X.copy()
        feature_names = list(X.columns)

    # Choose model based on target type
    unique_vals = len(np.unique(y))
    if unique_vals == 2:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif unique_vals <= 10 and np.issubdtype(y.dtype, np.integer):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    model.fit(X_df, y)
    importances = model.feature_importances_
    feature_importances = dict(zip(feature_names, importances))

    # Sort by importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    if k is None:
        k = min(10, len(feature_names))

    selected_features = [f[0] for f in sorted_features[:k]]

    if isinstance(X, np.ndarray):
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, selected_indices]
    else:
        X_selected = X[selected_features]

    selection_info = {
        'method': 'importance',
        'model': str(type(model).__name__),
        'n_estimators': n_estimators,
        'k': k,
        'original_features': len(feature_names),
        'selected_features': len(selected_features),
        'feature_importances': feature_importances,
        'selected_importances': {f: feature_importances[f] for f in selected_features}
    }

    return X_selected, selected_features, selection_info


def select_features(X: DataFrame, y: Union[np.ndarray, pd.Series],
                   method: str = 'mutual_info', **kwargs) -> Tuple[DataFrame, List[str], Dict]:
    """
    Unified interface for feature selection methods.

    Args:
        X: Feature matrix.
        y: Target variable.
        method: Selection method ('correlation', 'mutual_info', 'univariate', 'rfe', 'importance').
        **kwargs: Method-specific parameters.

    Returns:
        Tuple of (selected features, selected feature names, selection info).
    """
    method_map = {
        'correlation': select_by_correlation,
        'mutual_info': select_by_mutual_info,
        'univariate': select_by_univariate_test,
        'rfe': select_by_rfe,
        'importance': select_by_importance
    }

    if method not in method_map:
        raise ValueError(f"Unknown selection method: {method}. Available: {list(method_map.keys())}")

    selector_func = method_map[method]
    return selector_func(X, y, **kwargs)
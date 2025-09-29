"""
Importance Analysis Integration Module

Provides a unified interface for feature importance analysis using various methods.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Optional
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..statistical_analysis import get_target_type
from .feature_selection import select_by_importance


def compute_shap_importance(model, X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           max_evals: int = 1000) -> Dict[str, Any]:
    """
    Compute SHAP feature importance.

    Args:
        model: Trained model.
        X: Feature matrix.
        y: Target variable.
        max_evals: Maximum evaluations for SHAP.

    Returns:
        Dictionary with SHAP importance results.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available. Install with: pip install shap")

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_array = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_array = X

    target_type = get_target_type(y)

    try:
        if target_type == 'categorical':
            explainer = shap.Explainer(model, X_array)
            shap_values = explainer(X_array, max_evals=max_evals)
        else:
            explainer = shap.Explainer(model, X_array)
            shap_values = explainer(X_array, max_evals=max_evals)

        # For multi-class, shap_values is a list
        if isinstance(shap_values, list):
            # Take mean absolute SHAP values across classes
            shap_importance = np.mean([np.abs(sv.values).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values.values).mean(axis=0)

        feature_importance = dict(zip(feature_names, shap_importance))

        # Sort by importance
        importance_ranking = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        return {
            'method': 'shap',
            'feature_importance': feature_importance,
            'importance_ranking': importance_ranking,
            'shap_values': shap_values,
            'explainer': explainer
        }

    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}")


def compute_permutation_importance(model, X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, pd.Series],
                                  n_repeats: int = 5) -> Dict[str, Any]:
    """
    Compute permutation feature importance.

    Args:
        model: Trained model.
        X: Feature matrix.
        y: Target variable.
        n_repeats: Number of permutation repeats.

    Returns:
        Dictionary with permutation importance results.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_array = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_array = X

    perm_importance = permutation_importance(model, X_array, y, n_repeats=n_repeats, random_state=42)

    feature_importance = dict(zip(feature_names, perm_importance.importances_mean))
    importance_ranking = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    return {
        'method': 'permutation',
        'feature_importance': feature_importance,
        'importance_ranking': importance_ranking,
        'importances_mean': perm_importance.importances_mean,
        'importances_std': perm_importance.importances_std,
        'n_repeats': n_repeats
    }


def compute_model_coefficients(model, feature_names: List[str]) -> Dict[str, Any]:
    """
    Extract feature importance from model coefficients (for linear models).

    Args:
        model: Trained linear model.
        feature_names: List of feature names.

    Returns:
        Dictionary with coefficient-based importance.
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model does not have coefficients (not a linear model)")

    coefficients = model.coef_

    # Handle multi-class case
    if coefficients.ndim > 1:
        # Take mean absolute coefficients across classes
        importance_scores = np.mean(np.abs(coefficients), axis=0)
    else:
        importance_scores = np.abs(coefficients)

    feature_importance = dict(zip(feature_names, importance_scores))
    importance_ranking = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    return {
        'method': 'coefficients',
        'feature_importance': feature_importance,
        'importance_ranking': importance_ranking,
        'coefficients': coefficients
    }


def analyze_feature_importance(X: Union[np.ndarray, pd.DataFrame],
                              y: Union[np.ndarray, pd.Series],
                              methods: List[str] = None,
                              model=None) -> Dict[str, Any]:
    """
    Comprehensive feature importance analysis using multiple methods.

    Args:
        X: Feature matrix.
        y: Target variable.
        methods: List of importance methods to use.
        model: Pre-trained model (optional).

    Returns:
        Dictionary with importance analysis results from all methods.
    """
    if methods is None:
        methods = ['tree_importance', 'permutation']
        if SHAP_AVAILABLE:
            methods.append('shap')

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_array = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_array = X

    results = {}

    # Train a default model if none provided
    if model is None:
        target_type = get_target_type(y)
        if target_type == 'categorical':
            if len(np.unique(y)) == 2:
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_array, y)

    # Tree-based importance (built into select_by_importance)
    if 'tree_importance' in methods:
        try:
            _, _, tree_results = select_by_importance(X, y, k=None)
            results['tree_importance'] = {
                'feature_importance': tree_results['feature_importances'],
                'importance_ranking': sorted(tree_results['feature_importances'].items(),
                                           key=lambda x: x[1], reverse=True)
            }
        except Exception as e:
            results['tree_importance'] = {'error': str(e)}

    # SHAP importance
    if 'shap' in methods and SHAP_AVAILABLE:
        try:
            shap_results = compute_shap_importance(model, X, y)
            results['shap'] = shap_results
        except Exception as e:
            results['shap'] = {'error': str(e)}

    # Permutation importance
    if 'permutation' in methods:
        try:
            perm_results = compute_permutation_importance(model, X, y)
            results['permutation'] = perm_results
        except Exception as e:
            results['permutation'] = {'error': str(e)}

    # Coefficient-based importance (for linear models)
    if 'coefficients' in methods:
        try:
            if hasattr(model, 'coef_'):
                coeff_results = compute_model_coefficients(model, feature_names)
                results['coefficients'] = coeff_results
            else:
                results['coefficients'] = {'error': 'Model does not have coefficients'}
        except Exception as e:
            results['coefficients'] = {'error': str(e)}

    # Aggregate rankings across methods
    method_rankings = {}
    for method, result in results.items():
        if 'importance_ranking' in result:
            method_rankings[method] = [item[0] for item in result['importance_ranking']]

    # Compute consensus ranking
    if method_rankings:
        all_features = set()
        for ranking in method_rankings.values():
            all_features.update(ranking)

        consensus_scores = {}
        for feature in all_features:
            score = 0
            for method, ranking in method_rankings.items():
                if feature in ranking:
                    # Higher score for features ranked higher
                    score += len(ranking) - ranking.index(feature)
            consensus_scores[feature] = score

        consensus_ranking = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)

        results['consensus'] = {
            'method': 'consensus',
            'importance_ranking': consensus_ranking,
            'method_contributions': list(method_rankings.keys())
        }

    return results


def get_top_important_features(importance_results: Dict[str, Any],
                              method: str = 'consensus', top_k: int = 10) -> List[str]:
    """
    Extract top-k important features from importance analysis results.

    Args:
        importance_results: Results from analyze_feature_importance.
        method: Importance method to use for ranking.
        top_k: Number of top features to return.

    Returns:
        List of top important feature names.
    """
    if method not in importance_results:
        raise ValueError(f"Method '{method}' not found in importance results")

    result = importance_results[method]
    if 'importance_ranking' not in result:
        raise ValueError(f"Method '{method}' does not have importance ranking")

    ranking = result['importance_ranking']
    return [feature for feature, _ in ranking[:top_k]]
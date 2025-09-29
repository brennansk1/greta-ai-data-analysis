"""
Feature Importance Utilities Module

Contains utilities for computing feature importance.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import permutation_importance

# Try to import SHAP, fallback to permutation importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from greta_core.statistical_analysis import get_target_type


def _compute_feature_importance(individual: List[int], data: np.ndarray, target: np.ndarray, feature_names: List[str], chromosome_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute feature importance for a hypothesis using SHAP or permutation importance.

    Args:
        individual: Binary list representing feature selection and engineering.
        data: Feature matrix.
        target: Target variable.
        feature_names: List of original feature names.
        chromosome_info: Dictionary with chromosome structure info.

    Returns:
        Dictionary with feature_importance and importance_ranking.
    """
    num_features = data.shape[1]

    # Parse chromosome as in evaluate_hypothesis
    selection_start, selection_end = chromosome_info['selection']
    squared_start, squared_end = chromosome_info['squared']
    # cubed_start, cubed_end = chromosome_info['cubed']  # Removed
    interactions_start, interactions_end = chromosome_info['interactions']
    interaction_pairs = chromosome_info['interaction_pairs']

    # Feature selection
    selection_bits = individual[selection_start:selection_end]
    selected_original = [i for i, bit in enumerate(selection_bits) if bit == 1]

    # Engineered features
    engineered_features = []
    engineered_names = []

    # Squared terms
    squared_bits = individual[squared_start:squared_end]
    for i, bit in enumerate(squared_bits):
        if bit == 1:
            engineered_features.append(data[:, i] ** 2)
            engineered_names.append(f'{feature_names[i]}_squared')

    # Interaction terms (cubed removed)
    interaction_bits = individual[interactions_start:interactions_end]
    for idx, bit in enumerate(interaction_bits):
        if bit == 1:
            i, j = interaction_pairs[idx]
            engineered_features.append(data[:, i] * data[:, j])
            engineered_names.append(f'{feature_names[i]}_{feature_names[j]}_interaction')

    # Constraint: limit total engineered features
    max_engineered = num_features
    if len(engineered_features) > max_engineered:
        engineered_features = engineered_features[:max_engineered]
        engineered_names = engineered_names[:max_engineered]

    # Combine selected original and engineered features
    feature_list = [data[:, selected_original]] if selected_original else []
    feature_list.extend(engineered_features)

    if not feature_list:
        return {'feature_importance': {}, 'importance_ranking': []}

    X_selected = np.column_stack(feature_list)

    # Feature names for X_selected
    selected_names = [feature_names[i] for i in selected_original] + engineered_names

    # Determine target type and choose model
    target_type = get_target_type(target)
    if target_type == 'categorical':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = LinearRegression()

    # Train model
    model.fit(X_selected, target)

    # Compute importance
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Computing feature importance for {len(selected_names)} features")
    logger.info(f"SHAP available: {SHAP_AVAILABLE}")

    if SHAP_AVAILABLE:
        try:
            logger.info("Attempting SHAP computation")
            if target_type == 'categorical':
                explainer = shap.LinearExplainer(model, X_selected)
                shap_values = explainer.shap_values(X_selected)
                # For binary classification, shap_values is list, take first
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
            else:
                explainer = shap.LinearExplainer(model, X_selected)
                shap_values = explainer.shap_values(X_selected)
            importance_scores = np.mean(np.abs(shap_values), axis=0)
            logger.info("SHAP computation successful")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}, falling back to permutation importance")
            # Fallback to permutation
            perm_importance = permutation_importance(model, X_selected, target, n_repeats=5, random_state=42)
            importance_scores = perm_importance.importances_mean
    else:
        logger.info("Using permutation importance (SHAP not available)")
        perm_importance = permutation_importance(model, X_selected, target, n_repeats=5, random_state=42)
        importance_scores = perm_importance.importances_mean

    # Create importance dict
    feature_importance = dict(zip(selected_names, importance_scores))

    # Sort by importance descending
    importance_ranking = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    return {
        'feature_importance': feature_importance,
        'importance_ranking': importance_ranking
    }
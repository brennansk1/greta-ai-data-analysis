"""
Evaluation Utilities Module

Contains functions for evaluating hypotheses.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Union

from greta_core.statistical_analysis import (
    calculate_significance, calculate_effect_size, calculate_coverage, calculate_parsimony
)


def evaluate_hypothesis(individual: Union[List[int], np.ndarray], data: np.ndarray, target: np.ndarray, chromosome_info: Dict[str, Any] = None) -> Tuple[float, float, float, float]:
    """
    Evaluate a hypothesis represented by the individual with feature engineering.

    Args:
        individual: Binary list or array representing feature selection and engineering.
        data: Feature matrix.
        target: Target variable.
        chromosome_info: Dictionary with chromosome structure info.

    Returns:
        Tuple of (significance, effect_size, coverage, parsimony_penalty).
    """
    import logging
    logger = logging.getLogger(__name__)

    # Convert numpy array to list if necessary
    if isinstance(individual, np.ndarray):
        individual = individual.tolist()

    num_features = data.shape[1]
    logger.debug(f"Evaluating individual with {len(individual)} genes on data shape {data.shape}")

    # Backward compatibility: if chromosome_info is None or individual length matches old format, use old evaluation
    if chromosome_info is None or len(individual) == num_features:
        selected_features = [i for i, bit in enumerate(individual) if bit == 1]
        if not selected_features:
            return 0.0, 0.0, 0.0, 1.0
        X_selected = data[:, selected_features]
        significance = calculate_significance(X_selected, target)
        effect_size = calculate_effect_size(X_selected, target)
        coverage = calculate_coverage(X_selected, target)
        parsimony = calculate_parsimony(len(selected_features), num_features)
        return significance, effect_size, coverage, parsimony

    # Parse chromosome with constraints
    selection_start, selection_end = chromosome_info['selection']
    squared_start, squared_end = chromosome_info['squared']
    # cubed_start, cubed_end = chromosome_info['cubed']  # Removed
    interactions_start, interactions_end = chromosome_info['interactions']
    interaction_pairs = chromosome_info['interaction_pairs']
    max_engineered_features = chromosome_info.get('max_engineered_features', num_features)

    # Feature selection (from available features in data)
    selected_features = []
    selected_indices = []

    # Original features that can be selected directly
    selection_bits = individual[selection_start:selection_end]
    for i, bit in enumerate(selection_bits):
        if bit == 1 and i < data.shape[1]:  # Ensure we don't exceed data dimensions
            selected_features.append(data[:, i])
            selected_indices.append(i)

    # Engineered features (only on first max_engineered_features)
    # Squared terms
    squared_bits = individual[squared_start:squared_end]
    for i, bit in enumerate(squared_bits):
        if bit == 1 and i < max_engineered_features and i < data.shape[1]:
            selected_features.append(data[:, i] ** 2)
            selected_indices.append(i)  # Reuse index for engineered version

    # Interaction terms (cubed terms removed)
    interaction_bits = individual[interactions_start:interactions_end]
    for idx, bit in enumerate(interaction_bits):
        if bit == 1 and idx < len(interaction_pairs):
            i, j = interaction_pairs[idx]
            if i < data.shape[1] and j < data.shape[1]:
                selected_features.append(data[:, i] * data[:, j])
                selected_indices.append((i, j))  # Tuple for interaction

    # Limit total features to prevent explosion
    max_total_features = min(len(selected_features), data.shape[1])
    selected_features = selected_features[:max_total_features]

    if not selected_features:
        return 0.0, 0.0, 0.0, 1.0  # No features, poor fitness

    X_selected = np.column_stack(selected_features)

    # Basic analysis
    significance = calculate_significance(X_selected, target)
    effect_size = calculate_effect_size(X_selected, target)
    coverage = calculate_coverage(X_selected, target)

    # Calculate parsimony (based on total features used)
    total_features_used = len(selected_indices)
    parsimony = calculate_parsimony(total_features_used, num_features)

    return significance, effect_size, coverage, parsimony
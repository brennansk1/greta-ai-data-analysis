"""
Generate Hypotheses Module

Contains the main function for generating hypotheses using genetic algorithms.
"""

import random
import multiprocessing
import time
import math
from functools import partial
from collections import Counter
from deap import base, creator, tools
import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import List, Tuple, Dict, Any, Union, Optional
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import permutation_importance

# Try to import SHAP, fallback to permutation importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from greta_core.statistical_analysis import (
    calculate_significance, calculate_effect_size, calculate_coverage, calculate_parsimony,
    perform_multiple_linear_regression, detect_trend, detect_seasonality, perform_causal_analysis
)
from greta_core.statistical_analysis.tests.significance_tests import get_target_type
from greta_core.preprocessing import detect_feature_types, prepare_features_for_modeling

from .optimizers import GeneticAlgorithmOptimizer
from .chromosome_utils import get_chromosome_info
from .evaluation_utils import evaluate_hypothesis
from .data_prep_utils import _compute_mutual_info, _select_top_features, _prepare_data_for_ga
from .importance_utils import _compute_feature_importance

# Type alias for DataFrame
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def generate_hypotheses(data: Union[np.ndarray, DataFrame], target: Union[np.ndarray, DataFrame, str], bootstrap_iterations: int = 1, bootstrap_sample_frac: float = 0.8, **kwargs) -> List[Dict[str, Any]]:
    """
    Generate hypotheses using genetic algorithm with support for mixed data types.

    Args:
        data: Feature matrix (numpy array or DataFrame).
        target: Target variable (numpy array, DataFrame column, or column name).
        bootstrap_iterations: Number of bootstrap iterations for stability.
        bootstrap_sample_frac: Fraction of data to sample for each bootstrap.
        **kwargs: Additional parameters for GA, including use_causal_prioritization, pre_filter_fraction,
                  adaptive_params, diversity_threshold, convergence_threshold, encoding_method, feature_names.

    Returns:
        List of hypothesis dictionaries with features and fitness values.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Starting generate_hypotheses")
    logger.info(f"Data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
    logger.info(f"Bootstrap iterations: {bootstrap_iterations}")

    progress_callback = kwargs.pop('progress_callback', None)
    use_causal_prioritization = kwargs.pop('use_causal_prioritization', False)
    pre_filter_fraction = kwargs.pop('pre_filter_fraction', 1.0)
    encoding_method = kwargs.pop('encoding_method', 'one_hot')
    feature_names_param = kwargs.pop('feature_names', None)

    logger.info(f"use_causal_prioritization: {use_causal_prioritization}")
    logger.info(f"pre_filter_fraction: {pre_filter_fraction}")
    logger.info(f"encoding_method: {encoding_method}")
    logger.info(f"feature_names provided: {feature_names_param is not None}")

    # Bootstrap for stability selection
    feature_counter = Counter()
    if bootstrap_iterations > 1:
        for _ in range(bootstrap_iterations):
            # Sample data for this iteration
            data_sample, target_sample, feature_names_sample = _prepare_data_for_ga(data, target, sample_frac=bootstrap_sample_frac, encoding_method=encoding_method)
            # Pre-filter features if specified
            if pre_filter_fraction < 1.0:
                if use_causal_prioritization and causal_scores:
                    scores = np.array([causal_scores.get(name, 0.0) for name in feature_names_sample])
                else:
                    scores = _compute_mutual_info(data_sample, target_sample)
                top_indices = _select_top_features(scores, pre_filter_fraction)
                data_sample = data_sample[:, top_indices]
                feature_names_sample = [feature_names_sample[i] for i in top_indices]
            # Run GA on sample
            best_individuals_sample = run_genetic_algorithm(data_sample, target_sample, **kwargs)
            # Get chromosome info
            chromosome_info = get_chromosome_info(data_sample.shape[1])
            # Collect selected features from all hypotheses
            for ind in best_individuals_sample:
                # Parse to get selected_features (similar to below)
                selection_start, selection_end = chromosome_info['selection']
                squared_start, squared_end = chromosome_info['squared']
                interactions_start, interactions_end = chromosome_info['interactions']
                interaction_pairs = chromosome_info['interaction_pairs']
                max_engineered_features = chromosome_info.get('max_engineered_features', data_sample.shape[1])
                num_features = data_sample.shape[1]
                if len(ind) == num_features:
                    selected_features = [feature_names_sample[i] for i, bit in enumerate(ind) if bit == 1]
                else:
                    selection_bits = ind[selection_start:selection_end]
                    selected_original = [i for i, bit in enumerate(selection_bits) if bit == 1]
                    engineered_features = []
                    squared_bits = ind[squared_start:squared_end]
                    for i, bit in enumerate(squared_bits):
                        if bit == 1 and i < max_engineered_features:
                            engineered_features.append(f'{feature_names_sample[i]}_squared')
                    interaction_bits = ind[interactions_start:interactions_end]
                    for idx, bit in enumerate(interaction_bits):
                        if bit == 1 and idx < len(interaction_pairs):
                            i, j = interaction_pairs[idx]
                            engineered_features.append(f'{feature_names_sample[i]}_{feature_names_sample[j]}_interaction')
                    selected_original_names = [feature_names_sample[i] for i in selected_original]
                    selected_features = selected_original_names + engineered_features
                feature_counter.update(selected_features)

    # Compute stability scores
    stability_scores = {feature: count / bootstrap_iterations for feature, count in feature_counter.items()} if bootstrap_iterations > 1 else {}

    # Compute causal scores if enabled
    causal_scores = {}
    if use_causal_prioritization and isinstance(data, pd.DataFrame):
        logger.info("Starting causal prioritization analysis")
        target_col = target if isinstance(target, str) else '__target__'
        data_for_causal = data.copy()
        if not isinstance(target, str):
            data_for_causal[target_col] = target

        total_features = len([f for f in data_for_causal.columns if f != target_col])
        logger.info(f"Performing causal analysis on {total_features} features")

        for i, feature in enumerate(data_for_causal.columns):
            if feature != target_col:
                try:
                    logger.debug(f"Causal analysis for feature {i+1}/{total_features}: {feature}")
                    result = perform_causal_analysis(data_for_causal, feature, target_col)
                    causal_scores[feature] = abs(result['estimation']['estimate'])
                except Exception as e:
                    logger.warning(f"Causal analysis failed for {feature}: {e}")
                    causal_scores[feature] = 0.0

        logger.info("Causal prioritization analysis completed")
    else:
        logger.info("Causal prioritization disabled or not applicable")

    # Convert inputs to numpy arrays for final GA
    data_array, target_array, feature_names = _prepare_data_for_ga(data, target, sample_frac=1.0, encoding_method=encoding_method)

    # Override feature names if provided (for CLI integration)
    if feature_names_param is not None:
        logger.info(f"Overriding feature names from {len(feature_names)} to {len(feature_names_param)} names")
        feature_names = feature_names_param

    # Pre-filter features if specified
    if pre_filter_fraction < 1.0:
        if use_causal_prioritization and causal_scores:
            scores = np.array([causal_scores.get(name, 0.0) for name in feature_names])
        else:
            scores = _compute_mutual_info(data_array, target_array)
        top_indices = _select_top_features(scores, pre_filter_fraction)
        data_reduced = data_array[:, top_indices]
        feature_names_reduced = [feature_names[i] for i in top_indices]
    else:
        data_reduced = data_array
        feature_names_reduced = feature_names

    # Set defaults and update with kwargs
    defaults = {
        'pop_size': 100,
        'num_generations': 50,
        'cx_prob': 0.7,
        'mut_prob': 0.2,
        'n_processes': 1,
        'use_dask': False,
        'adaptive_params': False,
        'diversity_threshold': 0.1,
        'convergence_threshold': 0.01,
        'local_search_enabled': False,
        'local_search_method': 'hill_climbing',
        'elite_fraction': 0.1,
        'local_search_iterations': 10
    }
    defaults.update(kwargs)

    kwargs_for_ga = defaults.copy()
    if progress_callback is not None:
        kwargs_for_ga['progress_callback'] = progress_callback
    best_individuals = run_genetic_algorithm(data_reduced, target_array, **kwargs_for_ga)

    # Get chromosome info for parsing
    chromosome_info = get_chromosome_info(data_reduced.shape[1])

    hypotheses = []
    for ind in best_individuals:
        # Parse chromosome
        selection_start, selection_end = chromosome_info['selection']
        squared_start, squared_end = chromosome_info['squared']
        # cubed_start, cubed_end = chromosome_info['cubed']  # Removed
        interactions_start, interactions_end = chromosome_info['interactions']
        interaction_pairs = chromosome_info['interaction_pairs']

        # Constrained parsing to match evaluate_hypothesis
        max_engineered_features = chromosome_info.get('max_engineered_features', data_reduced.shape[1])

        selected_features = []

        # Original features
        selection_bits = ind[selection_start:selection_end]
        for i, bit in enumerate(selection_bits):
            if bit == 1 and i < len(feature_names_reduced):
                selected_features.append(feature_names_reduced[i])

        # Squared terms (only on first max_engineered_features)
        squared_bits = ind[squared_start:squared_end]
        for i, bit in enumerate(squared_bits):
            if bit == 1 and i < max_engineered_features and i < len(feature_names_reduced):
                selected_features.append(f'{feature_names_reduced[i]}_squared')

        # Interaction terms (cubed terms removed)
        interaction_bits = ind[interactions_start:interactions_end]
        for idx, bit in enumerate(interaction_bits):
            if bit == 1 and idx < len(interaction_pairs):
                i, j = interaction_pairs[idx]
                # Ensure indices are within bounds and refer to original features (not engineered ones)
                if i < max_engineered_features and j < max_engineered_features and i < len(feature_names_reduced) and j < len(feature_names_reduced):
                    selected_features.append(f'{feature_names_reduced[i]}_{feature_names_reduced[j]}_interaction')

        fitness_values = ind.fitness.values

        hypothesis = {
            'features': selected_features,
            'significance': fitness_values[0],
            'effect_size': fitness_values[1],
            'coverage': fitness_values[2],
            'parsimony_penalty': fitness_values[3],
            'fitness': sum(fitness_values[:3]) - fitness_values[3]  # Overall fitness
        }

        # Add stability scores
        if stability_scores:
            hypothesis['feature_stability'] = {f: stability_scores.get(f, 0) for f in selected_features}
            hypothesis['stability_scores'] = stability_scores

        # Compute feature importance
        importance_dict = _compute_feature_importance(ind, data_reduced, target_array, feature_names_reduced, chromosome_info)
        hypothesis.update(importance_dict)

        hypotheses.append(hypothesis)

    # Sort by overall fitness
    hypotheses.sort(key=lambda x: x['fitness'], reverse=True)

    return hypotheses


def run_genetic_algorithm(data: np.ndarray, target: np.ndarray, pop_size: int = 100, num_generations: int = 50,
                          cx_prob: float = 0.7, mut_prob: float = 0.2, n_processes: int = 1, use_dask: bool = False,
                          adaptive_params: bool = False, diversity_threshold: float = 0.1, convergence_threshold: float = 0.01,
                          progress_callback=None, local_search_enabled: bool = False, local_search_method: str = 'hill_climbing',
                          elite_fraction: float = 0.1, local_search_iterations: int = 10) -> List[List[int]]:
    """
    Run the genetic algorithm to find optimal hypotheses using the modular optimizer framework.

    Args:
        data: Feature matrix.
        target: Target variable.
        pop_size: Population size.
        num_generations: Number of generations.
        cx_prob: Crossover probability.
        mut_prob: Mutation probability.
        n_processes: Number of processes for parallel evaluation. If 1, uses sequential evaluation.
        use_dask: Whether to use Dask for distributed execution.
        adaptive_params: Whether to adaptively adjust cx_prob and mut_prob based on diversity and convergence.
        diversity_threshold: Threshold for population diversity (fitness variance) below which mutation is increased.
        convergence_threshold: Threshold for fitness improvement rate below which parameters are adjusted.
        progress_callback: Optional callback function for progress updates.

    Returns:
        List of best individuals (hypotheses).
    """
    optimizer = GeneticAlgorithmOptimizer(
        data, target,
        pop_size=pop_size,
        num_generations=num_generations,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        n_processes=n_processes,
        use_dask=use_dask,
        adaptive_params=adaptive_params,
        diversity_threshold=diversity_threshold,
        convergence_threshold=convergence_threshold,
        progress_callback=progress_callback,
        local_search_enabled=local_search_enabled,
        local_search_method=local_search_method,
        elite_fraction=elite_fraction,
        local_search_iterations=local_search_iterations
    )
    return optimizer.optimize()
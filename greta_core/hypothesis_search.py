"""
Hypothesis Search Module

Implements the genetic algorithm for exploring potential relationships and patterns in data.
Generates candidate hypotheses through evolutionary optimization, focusing on combinations
of variables that may explain target outcomes.
"""

import random
import multiprocessing
import time
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
from .statistical_analysis import (
    calculate_significance, calculate_effect_size, calculate_coverage, calculate_parsimony,
    perform_multiple_linear_regression, detect_trend, detect_seasonality
)
from .significance_tests import get_target_type
from .causal_analysis import perform_causal_analysis
from .preprocessing import detect_feature_types, prepare_features_for_modeling

# Type alias for DataFrame
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def get_chromosome_info(num_features: int) -> Dict[str, Any]:
    """
    Get chromosome structure information for feature selection and engineering.
    Only allows engineering on original features to prevent feature explosion.

    Args:
        num_features: Number of original features.

    Returns:
        Dictionary with chromosome structure info.
    """
    # Only original features can be selected
    selection_end = num_features

    # Engineering operations only apply to original features
    # Note: We limit engineering to prevent explosion
    max_engineered_per_type = 0  # Disable interactions to avoid feature name mapping issues

    squared_end = selection_end + max_engineered_per_type
    # Remove cubed terms since preprocessing doesn't create them
    # cubed_end = squared_end + max_engineered_per_type

    # Define interaction pairs (only between original features)
    interaction_pairs = []
    for i in range(max_engineered_per_type):
        for j in range(i + 1, max_engineered_per_type):
            interaction_pairs.append((i, j))

    num_interactions = len(interaction_pairs)
    interactions_end = squared_end + num_interactions

    return {
        'total_length': interactions_end,
        'selection': (0, selection_end),
        'squared': (selection_end, squared_end),
        # 'cubed': (squared_end, cubed_end),  # Removed cubed terms
        'interactions': (squared_end, interactions_end),
        'interaction_pairs': interaction_pairs,
        'max_engineered_features': max_engineered_per_type
    }


def _compute_mutual_info(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute mutual information scores for features.

    Args:
        data: Feature matrix.
        target: Target variable.

    Returns:
        Array of mutual information scores.
    """
    target_type = get_target_type(target)
    if target_type == 'categorical':
        return mutual_info_classif(data, target)
    else:
        return mutual_info_regression(data, target)


def _select_top_features(scores: np.ndarray, fraction: float) -> np.ndarray:
    """
    Select top fraction of features based on scores.

    Args:
        scores: Array of scores.
        fraction: Fraction to select (e.g., 0.5 for top 50%).

    Returns:
        Indices of top features.
    """
    num_top = max(1, int(len(scores) * fraction))
    top_indices = np.argsort(scores)[-num_top:]
    return top_indices


# Create fitness class for multi-objective optimization
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, -1.0))  # Maximize significance, effect size, coverage; minimize parsimony penalty
creator.create("Individual", list, fitness=creator.FitnessMulti)


def create_toolbox(num_features: int) -> Tuple[base.Toolbox, Dict[str, Any]]:
    """
    Create DEAP toolbox for the genetic algorithm with feature engineering.

    Args:
        num_features: Number of features in the dataset.

    Returns:
        Tuple of (configured DEAP toolbox, chromosome info dictionary).
    """
    chromosome_info = get_chromosome_info(num_features)
    total_length = chromosome_info['total_length']

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Structure initializers: total_length for feature selection and engineering
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, total_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", evaluate_hypothesis)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)  # NSGA-II for multi-objective

    return toolbox, chromosome_info


def evaluate_hypothesis(individual: List[int], data: np.ndarray, target: np.ndarray, chromosome_info: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float, float]:
    """
    Evaluate a hypothesis represented by the individual with feature engineering.

    Args:
        individual: Binary list representing feature selection and engineering.
        data: Feature matrix.
        target: Target variable.
        chromosome_info: Dictionary with chromosome structure info.

    Returns:
        Tuple of (significance, effect_size, coverage, parsimony_penalty).
    """
    import logging
    logger = logging.getLogger(__name__)

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
    # cubed_start, cubed_end = chromosome_info['cubed']  # Removed cubed
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


def run_genetic_algorithm(data: np.ndarray, target: np.ndarray, pop_size: int = 100, num_generations: int = 50,
                         cx_prob: float = 0.7, mut_prob: float = 0.2, n_processes: int = 1, use_dask: bool = False,
                         adaptive_params: bool = False, diversity_threshold: float = 0.1, convergence_threshold: float = 0.01,
                         progress_callback=None) -> List[List[int]]:
    """
    Run the genetic algorithm to find optimal hypotheses.

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
    num_features = data.shape[1]

    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    elif n_processes < 1:
        n_processes = 1

    toolbox, chromosome_info = create_toolbox(num_features)
    eval_partial = partial(evaluate_hypothesis, data=data, target=target, chromosome_info=chromosome_info)
    toolbox.register("evaluate", eval_partial)

    pool = None
    client = None
    if use_dask:
        try:
            from dask.distributed import Client
            client = Client(processes=False, threads_per_worker=1, n_workers=n_processes)
            def dask_map(func, iterable):
                futures = client.map(func, iterable)
                return [f.result() for f in futures]
            if toolbox.map == map:
                toolbox.register("map", dask_map)
        except ImportError:
            use_dask = False

    if not use_dask:
        if n_processes > 1:
            pool = multiprocessing.Pool(processes=n_processes)
            if toolbox.map == map:
                toolbox.register("map", pool.map)
        elif toolbox.map == map:
            toolbox.register("map", map)

    # Create initial population
    pop = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Initialize adaptive parameters tracking
    if adaptive_params:
        fitness_sums = [sum(ind.fitness.values) for ind in pop]
        prev_best_fitness_sum = max(fitness_sums)

    # Run evolution
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Starting GA evolution for {num_generations} generations with population size {pop_size}")

    for gen in range(num_generations):
        logger.info(f"Generation {gen + 1}/{num_generations} starting")

        # Select offspring
        offspring = toolbox.select(pop, len(pop))
        logger.debug(f"Selected {len(offspring)} offspring")

        # Clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        crossover_count = 0
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                crossover_count += 1
                try:
                    del child1.fitness.values
                except AttributeError:
                    pass
                try:
                    del child2.fitness.values
                except AttributeError:
                    pass
        logger.debug(f"Applied crossover to {crossover_count} pairs")

        # Apply mutation
        mutation_count = 0
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                mutation_count += 1
                try:
                    del mutant.fitness.values
                except AttributeError:
                    pass
        logger.debug(f"Applied mutation to {mutation_count} individuals")

        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        logger.info(f"Evaluating {len(invalid_ind)} invalid individuals")

        if invalid_ind:
            start_eval = time.time()
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            eval_time = time.time() - start_eval
            logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Log best fitness
        best_fitness = max(sum(ind.fitness.values) for ind in pop)
        logger.info(f"Generation {gen + 1} completed. Best fitness: {best_fitness:.4f}")

        # Adaptive parameter adjustment
        if adaptive_params:
            fitness_sums = [sum(ind.fitness.values) for ind in pop]
            diversity = np.var(fitness_sums)
            current_best = max(fitness_sums)
            improvement = current_best - prev_best_fitness_sum
            # Adjust parameters based on diversity and convergence
            if diversity < diversity_threshold:
                mut_prob = min(mut_prob + 0.05, 0.5)
            if improvement < convergence_threshold:
                cx_prob = min(cx_prob + 0.05, 0.9)
                mut_prob = min(mut_prob + 0.02, 0.5)
            prev_best_fitness_sum = current_best

        # Update progress
        if progress_callback:
            progress_callback()

    # Return Pareto front (best solutions)
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    # Cleanup
    if pool:
        pool.close()
        pool.join()
    if client:
        client.close()

    return pareto_front


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
        mi_scores = _compute_mutual_info(data_array, target_array)
        if use_causal_prioritization and causal_scores:
            causal_array = np.array([causal_scores.get(name, 0.0) for name in feature_names])
            # Weight MI with causal scores (add 1 to avoid zero multiplication)
            weighted_scores = mi_scores * (1 + causal_array)
        else:
            weighted_scores = mi_scores
        top_indices = _select_top_features(weighted_scores, pre_filter_fraction)
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
        'convergence_threshold': 0.01
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

        # Interaction terms (only between first max_engineered_features, cubed removed)
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


def _prepare_data_for_ga(data: Union[np.ndarray, DataFrame], target: Union[np.ndarray, DataFrame, str], sample_frac: float = 1.0,
                        encoding_method: str = 'one_hot') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data and target for genetic algorithm with automatic categorical encoding.

    Args:
        data: Feature matrix.
        target: Target variable.
        sample_frac: Fraction of data to sample.
        encoding_method: Method for encoding categorical variables ('one_hot' or 'target_encoding').

    Returns:
        Tuple of (data_array, target_array, feature_names) as numpy arrays and list.
    """
    # Handle data
    if isinstance(data, np.ndarray):
        data_array = data
        feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        if sample_frac < 1.0:
            n_samples = int(len(data_array) * sample_frac)
            indices = np.random.choice(len(data_array), size=n_samples, replace=False)
            data_array = data_array[indices]
            if isinstance(target, np.ndarray):
                target_array = target[indices]
            else:
                target_array = target
    elif hasattr(data, 'compute'):  # Dask DataFrame
        # For large datasets, consider sampling
        if sample_frac < 1.0:
            data_sample = data.sample(frac=sample_frac)
            target_sample = target.sample(frac=sample_frac) if hasattr(target, 'sample') else target
        else:
            sample_size = min(10000, len(data))  # Sample up to 10k rows for GA
            data_sample = data.sample(frac=sample_size / len(data))
            if hasattr(target, 'sample'):
                target_sample = target.sample(frac=sample_size / len(data))
            else:
                target_sample = target

        # Apply preprocessing and encoding
        data_sample_computed = data_sample.compute()
        target_sample_computed = target_sample.compute() if hasattr(target_sample, 'compute') else target_sample

        # Prepare target for encoding
        if isinstance(target, str):
            target_for_encoding = data_sample_computed[target]
        else:
            target_for_encoding = target_sample_computed

        # Apply feature preparation
        data_prepared, _ = prepare_features_for_modeling(data_sample_computed, target_for_encoding, encoding_method)

        data_array = data_prepared.values
        feature_names = list(data_prepared.columns)
        target_array = target_for_encoding.values if hasattr(target_for_encoding, 'values') else target_for_encoding

    else:  # Pandas DataFrame
        if sample_frac < 1.0:
            data_sample = data.sample(frac=sample_frac)
            if isinstance(target, str):
                target_sample = data_sample[target]
            elif hasattr(target, 'sample'):
                target_sample = target.sample(frac=sample_frac)
            else:
                target_sample = target
        else:
            data_sample = data
            if isinstance(target, str):
                target_sample = data[target]
            else:
                target_sample = target

        # Apply preprocessing and encoding
        if isinstance(target, str):
            target_for_encoding = data_sample[target]
        else:
            target_for_encoding = target_sample

        # Apply feature preparation
        data_prepared, _ = prepare_features_for_modeling(data_sample, target_for_encoding, encoding_method)

        data_array = data_prepared.values
        feature_names = list(data_prepared.columns)
        target_array = target_for_encoding.values if hasattr(target_for_encoding, 'values') else target_for_encoding

    # Handle target for numpy case (already handled above for DataFrames)
    if isinstance(data, np.ndarray):
        if isinstance(target, np.ndarray):
            target_array = target
            if sample_frac < 1.0:
                target_array = target_array[indices]
        elif isinstance(target, str):
            raise ValueError("Cannot use string target with numpy array data")
        else:
            target_array = target

    return data_array, target_array, feature_names


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
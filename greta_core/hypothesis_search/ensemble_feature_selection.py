"""
Ensemble Feature Selection Module

Contains functions for ensemble feature selection using multiple optimizers.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Union

from greta_core.preprocessing import prepare_features_for_modeling

from .optimizers import GeneticAlgorithmOptimizer, BayesianOptimization, PSOOptimizer
from .chromosome_utils import get_chromosome_info
from .evaluation_utils import evaluate_hypothesis


def ensemble_feature_selection(data: Union[np.ndarray, dict], target: Union[np.ndarray, dict, str], **kwargs) -> Dict[str, Any]:
    """
    Perform ensemble feature selection by running multiple optimizers (GA, BO, PSO) and combining their results.

    This function leverages different optimization strategies to provide more robust feature selection.
    Results are combined using voting (majority selection) and weighted averaging based on fitness scores.

    Args:
        data: Feature matrix (numpy array or DataFrame).
        target: Target variable (numpy array, DataFrame column, or column name).
        **kwargs: Additional parameters for optimizers, including:
            - GA params: pop_size, num_generations, cx_prob, mut_prob, n_processes, use_dask, etc.
            - BO params: n_calls, n_initial_points, random_state
            - PSO params: n_particles, iters, c1, c2, w, k, p, random_state
            - encoding_method: Method for encoding categorical variables ('one_hot' or 'target_encoding').

    Returns:
        Dictionary containing the ensemble hypothesis with combined features and fitness values.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Starting ensemble feature selection")

    # Prepare data similar to generate_hypotheses
    encoding_method = kwargs.get('encoding_method', 'one_hot')
    data_array, target_array, feature_names = _prepare_data_for_ga(data, target, sample_frac=1.0, encoding_method=encoding_method)

    chromosome_info = get_chromosome_info(data_array.shape[1])

    # Extract kwargs for each optimizer
    ga_kwargs = {k: v for k, v in kwargs.items() if k in [
        'pop_size', 'num_generations', 'cx_prob', 'mut_prob', 'n_processes', 'use_dask',
        'adaptive_params', 'diversity_threshold', 'convergence_threshold', 'progress_callback',
        'local_search_enabled', 'local_search_method', 'elite_fraction', 'local_search_iterations'
    ]}
    bo_kwargs = {k: v for k, v in kwargs.items() if k in ['n_calls', 'n_initial_points', 'random_state']}
    pso_kwargs = {k: v for k, v in kwargs.items() if k in ['n_particles', 'iters', 'c1', 'c2', 'w', 'k', 'p', 'random_state']}

    # Instantiate optimizers
    ga = GeneticAlgorithmOptimizer(data_array, target_array, **ga_kwargs)
    bo = BayesianOptimization(data_array, target_array, **bo_kwargs)
    pso = PSOOptimizer(data_array, target_array, **pso_kwargs)

    # Run optimizations
    logger.info("Running Genetic Algorithm")
    ga_results = ga.optimize()
    logger.info("Running Bayesian Optimization")
    bo_results = bo.optimize()
    logger.info("Running Particle Swarm Optimization")
    pso_results = pso.optimize()

    # Get best individual from each (first in list)
    best_ga = ga_results[0] if ga_results else []
    best_bo = bo_results[0] if bo_results else []
    best_pso = pso_results[0] if pso_results else []

    # Evaluate fitness for each best solution
    fitness_ga = evaluate_hypothesis(best_ga, data_array, target_array, chromosome_info) if best_ga else (0.0, 0.0, 0.0, 1.0)
    fitness_bo = evaluate_hypothesis(best_bo, data_array, target_array, chromosome_info) if best_bo else (0.0, 0.0, 0.0, 1.0)
    fitness_pso = evaluate_hypothesis(best_pso, data_array, target_array, chromosome_info) if best_pso else (0.0, 0.0, 0.0, 1.0)

    # Parse selected features from each
    def parse_selected_features(ind: List[int]) -> List[str]:
        if not ind:
            return []
        selection_start, selection_end = chromosome_info['selection']
        squared_start, squared_end = chromosome_info['squared']
        interactions_start, interactions_end = chromosome_info['interactions']
        interaction_pairs = chromosome_info['interaction_pairs']
        max_engineered_features = chromosome_info.get('max_engineered_features', data_array.shape[1])

        selected_features = []

        # Original features
        selection_bits = ind[selection_start:selection_end]
        for i, bit in enumerate(selection_bits):
            if bit == 1 and i < len(feature_names):
                selected_features.append(feature_names[i])

        # Squared terms
        squared_bits = ind[squared_start:squared_end]
        for i, bit in enumerate(squared_bits):
            if bit == 1 and i < max_engineered_features and i < len(feature_names):
                selected_features.append(f'{feature_names[i]}_squared')

        # Interaction terms
        interaction_bits = ind[interactions_start:interactions_end]
        for idx, bit in enumerate(interaction_bits):
            if bit == 1 and idx < len(interaction_pairs):
                i, j = interaction_pairs[idx]
                if i < max_engineered_features and j < max_engineered_features and i < len(feature_names) and j < len(feature_names):
                    selected_features.append(f'{feature_names[i]}_{feature_names[j]}_interaction')

        return selected_features

    features_ga = parse_selected_features(best_ga)
    features_bo = parse_selected_features(best_bo)
    features_pso = parse_selected_features(best_pso)

    # Combine results using voting and weighted averaging
    all_selected = features_ga + features_bo + features_pso
    feature_counts = Counter(all_selected)

    # Voting: select features that appear in at least 2 optimizers (majority)
    voted_features = [feat for feat, count in feature_counts.items() if count >= 2]

    # If no majority, fall back to weighted selection
    if not voted_features:
        # Compute weighted scores based on overall fitness
        overall_fitness_ga = sum(fitness_ga[:3]) - fitness_ga[3]
        overall_fitness_bo = sum(fitness_bo[:3]) - fitness_bo[3]
        overall_fitness_pso = sum(fitness_pso[:3]) - fitness_pso[3]

        feature_scores = {}
        all_features = set(all_selected)
        for feat in all_features:
            score = 0
            if feat in features_ga:
                score += overall_fitness_ga
            if feat in features_bo:
                score += overall_fitness_bo
            if feat in features_pso:
                score += overall_fitness_pso
            feature_scores[feat] = score

        # Select top 50% by weighted score
        num_to_select = max(1, len(feature_scores) // 2)
        voted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)[:num_to_select]

    # Compute combined fitness as average
    combined_fitness = [
        (fitness_ga[0] + fitness_bo[0] + fitness_pso[0]) / 3,
        (fitness_ga[1] + fitness_bo[1] + fitness_pso[1]) / 3,
        (fitness_ga[2] + fitness_bo[2] + fitness_pso[2]) / 3,
        (fitness_ga[3] + fitness_bo[3] + fitness_pso[3]) / 3
    ]

    # Create ensemble hypothesis
    hypothesis = {
        'features': voted_features,
        'significance': combined_fitness[0],
        'effect_size': combined_fitness[1],
        'coverage': combined_fitness[2],
        'parsimony_penalty': combined_fitness[3],
        'fitness': sum(combined_fitness[:3]) - combined_fitness[3],
        'optimizer_results': {
            'ga': {'features': features_ga, 'fitness': fitness_ga},
            'bo': {'features': features_bo, 'fitness': fitness_bo},
            'pso': {'features': features_pso, 'fitness': fitness_pso}
        },
        'ensemble_method': 'voting_weighted'
    }

    logger.info(f"Ensemble feature selection completed. Selected {len(voted_features)} features")
    return hypothesis
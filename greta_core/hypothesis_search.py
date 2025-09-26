"""
Hypothesis Search Module

Implements the genetic algorithm for exploring potential relationships and patterns in data.
Generates candidate hypotheses through evolutionary optimization, focusing on combinations
of variables that may explain target outcomes.
"""

import random
from deap import base, creator, tools
import numpy as np
from typing import List, Tuple, Dict, Any
from .statistical_analysis import (
    calculate_significance, calculate_effect_size, calculate_coverage, calculate_parsimony,
    perform_multiple_linear_regression, detect_trend, detect_seasonality
)


# Create fitness class for multi-objective optimization
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, -1.0))  # Maximize significance, effect size, coverage; minimize parsimony penalty
creator.create("Individual", list, fitness=creator.FitnessMulti)


def create_toolbox(num_features: int) -> base.Toolbox:
    """
    Create DEAP toolbox for the genetic algorithm.

    Args:
        num_features: Number of features in the dataset.

    Returns:
        Configured DEAP toolbox.
    """
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Structure initializers: 2 bits for analysis type + num_features for feature selection
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 2 + num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", evaluate_hypothesis)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)  # NSGA-II for multi-objective

    return toolbox


def evaluate_hypothesis(individual: List[int], data: np.ndarray, target: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Evaluate a hypothesis represented by the individual.

    Args:
        individual: Binary list representing analysis type and feature selection.
        data: Feature matrix.
        target: Target variable.

    Returns:
        Tuple of (significance, effect_size, coverage, parsimony_penalty).
    """
    # Decode analysis type from first 2 bits
    analysis_type_bits = individual[:2]
    analysis_type = analysis_type_bits[0] * 2 + analysis_type_bits[1]  # 0: basic, 1: regression, 2: time-series

    # Feature selection bits
    feature_bits = individual[2:]
    selected_features = [i for i, bit in enumerate(feature_bits) if bit == 1]

    if not selected_features:
        return 0.0, 0.0, 0.0, 1.0  # No features selected, poor fitness

    X_selected = data[:, selected_features]

    if analysis_type == 0:  # Basic analysis (existing)
        significance = calculate_significance(X_selected, target)
        effect_size = calculate_effect_size(X_selected, target)
        coverage = calculate_coverage(X_selected, target)
    elif analysis_type == 1:  # Multiple linear regression
        reg_results = perform_multiple_linear_regression(X_selected, target)
        significance = 1 - reg_results['f_p_value']  # Higher is better
        effect_size = np.sqrt(max(0, reg_results['adj_r_squared']))  # Effect size as sqrt(adj R^2), clamped to 0
        coverage = reg_results['adj_r_squared']
    elif analysis_type == 2:  # Time-series analysis
        # For time-series, assume target is time-series, and selected features are predictors
        # Use regression metrics, but could enhance with trend/seasonality if target is univariate
        if X_selected.shape[1] == 0:
            # If no features, analyze trend in target
            trend_results = detect_trend(target)
            significance = 1 - trend_results['p_value']
            effect_size = trend_results['trend_strength']
            coverage = trend_results['r_squared']
        else:
            # Regression on time-series predictors
            reg_results = perform_multiple_linear_regression(X_selected, target)
            significance = 1 - reg_results['f_p_value']
            effect_size = np.sqrt(reg_results['adj_r_squared'])
            coverage = reg_results['adj_r_squared']
    else:
        # Fallback to basic
        significance = calculate_significance(X_selected, target)
        effect_size = calculate_effect_size(X_selected, target)
        coverage = calculate_coverage(X_selected, target)

    parsimony = calculate_parsimony(len(selected_features), data.shape[1])

    return significance, effect_size, coverage, parsimony


def run_genetic_algorithm(data: np.ndarray, target: np.ndarray, pop_size: int = 100, num_generations: int = 50,
                         cx_prob: float = 0.7, mut_prob: float = 0.2, progress_callback=None) -> List[List[int]]:
    """
    Run the genetic algorithm to find optimal hypotheses.

    Args:
        data: Feature matrix.
        target: Target variable.
        pop_size: Population size.
        num_generations: Number of generations.
        cx_prob: Crossover probability.
        mut_prob: Mutation probability.

    Returns:
        List of best individuals (hypotheses).
    """
    num_features = data.shape[1]
    toolbox = create_toolbox(num_features)

    # Create initial population
    pop = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, data, target), pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Run evolution
    for gen in range(num_generations):
        # Select offspring
        offspring = toolbox.select(pop, len(pop))

        # Clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(lambda ind: toolbox.evaluate(ind, data, target), invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Update progress
        if progress_callback:
            progress_callback()

    # Return Pareto front (best solutions)
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    return pareto_front


def generate_hypotheses(data: np.ndarray, target: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    """
    Generate hypotheses using genetic algorithm.

    Args:
        data: Feature matrix.
        target: Target variable.
        **kwargs: Additional parameters for GA.

    Returns:
        List of hypothesis dictionaries with features and fitness values.
    """
    progress_callback = kwargs.pop('progress_callback', None)
    best_individuals = run_genetic_algorithm(data, target, progress_callback=progress_callback, **kwargs)

    hypotheses = []
    for ind in best_individuals:
        # Decode analysis type
        analysis_type_bits = ind[:2]
        analysis_type = analysis_type_bits[0] * 2 + analysis_type_bits[1]
        analysis_type_name = {0: 'basic', 1: 'regression', 2: 'time_series'}.get(analysis_type, 'basic')

        # Feature selection
        feature_bits = ind[2:]
        selected_features = [i for i, bit in enumerate(feature_bits) if bit == 1]
        fitness_values = ind.fitness.values

        hypothesis = {
            'features': selected_features,
            'analysis_type': analysis_type_name,
            'significance': fitness_values[0],
            'effect_size': fitness_values[1],
            'coverage': fitness_values[2],
            'parsimony_penalty': fitness_values[3],
            'fitness': sum(fitness_values[:3]) - fitness_values[3]  # Overall fitness
        }
        hypotheses.append(hypothesis)

    # Sort by overall fitness
    hypotheses.sort(key=lambda x: x['fitness'], reverse=True)

    return hypotheses
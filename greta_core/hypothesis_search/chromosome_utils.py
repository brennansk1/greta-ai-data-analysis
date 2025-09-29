"""
Chromosome Utilities Module

Contains utilities for managing chromosome structure and DEAP toolbox setup.
"""

from deap import base, creator, tools


def get_chromosome_info(num_features: int) -> dict[str, any]:
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


# Create fitness class for multi-objective optimization
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, -1.0))  # Maximize significance, effect size, coverage; minimize parsimony penalty
creator.create("Individual", list, fitness=creator.FitnessMulti)


def create_toolbox(num_features: int) -> tuple[base.Toolbox, dict[str, any]]:
    """
    Create DEAP toolbox for the genetic algorithm with feature engineering.

    Args:
        num_features: Number of features in the dataset.

    Returns:
        Tuple of (configured DEAP toolbox, chromosome info dictionary).
    """
    import random
    from .evaluation_utils import evaluate_hypothesis

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
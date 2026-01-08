"""
Optimizers Module

Contains the abstract Optimizer class and concrete implementations:
- GeneticAlgorithmOptimizer
- BayesianOptimization
- PSOOptimizer
"""

import random
import multiprocessing
import time
import math
from functools import partial
from abc import ABC, abstractmethod
from deap import base, creator, tools
import numpy as np
from typing import List, Tuple, Dict, Any, Union

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

from .chromosome_utils import get_chromosome_info, create_toolbox
from .evaluation_utils import evaluate_hypothesis
from ..scalability_errors import SparkUnavailableError, DistributedComputationError


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms used in hypothesis search.

    This framework allows for modular implementation of different optimizers such as
    Genetic Algorithms, Bayesian Optimization, Particle Swarm Optimization, etc.

    To add a new optimizer, create a subclass that implements the abstract methods:
    - initialize(): Set up any required state (e.g., population, bounds)
    - evaluate(candidate): Compute fitness/objective for a candidate solution
    - optimize(): Run the optimization loop and return best solutions

    Example for Bayesian Optimization:
        class BayesianOptimizer(Optimizer):
            def initialize(self):
                # Set up bounds, acquisition function, etc.
                pass
            def evaluate(self, candidate):
                # candidate is a parameter vector, return objective value
                return objective_function(candidate)
            def optimize(self):
                # Use a BO library to optimize
                return best_candidates

    Example for Particle Swarm Optimization:
        class PSOOptimizer(Optimizer):
            def initialize(self):
                # Initialize swarm, velocities, etc.
                pass
            def evaluate(self, candidate):
                # candidate is particle position
                return fitness(candidate)
            def optimize(self):
                # Run PSO algorithm
                return best_particles
    """

    def __init__(self, data: np.ndarray, target: np.ndarray, **kwargs):
        """
        Initialize the optimizer with data and target.

        Args:
            data: Feature matrix.
            target: Target variable.
            **kwargs: Additional parameters specific to the optimizer.
        """
        self.data = data
        self.target = target
        self.kwargs = kwargs
        self.num_features = data.shape[1]
        self.chromosome_info = get_chromosome_info(self.num_features)

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the optimizer's internal state (e.g., population, parameters).
        """
        pass

    @abstractmethod
    def evaluate(self, candidate: Any) -> Tuple[float, ...]:
        """
        Evaluate a candidate solution.

        Args:
            candidate: The candidate solution to evaluate.

        Returns:
            Tuple of fitness values.
        """
        pass

    @abstractmethod
    def optimize(self) -> List[Any]:
        """
        Run the optimization algorithm.

        Returns:
            List of best candidate solutions found.
        """
        pass


class GeneticAlgorithmOptimizer(Optimizer):
    """
    Genetic Algorithm optimizer for hypothesis search.

    Uses DEAP library to perform evolutionary optimization on feature selection
    and engineering hypotheses.
    """

    def __init__(self, data: np.ndarray, target: np.ndarray, **kwargs):
        super().__init__(data, target, **kwargs)
        self.pop_size = kwargs.get('pop_size', 100)
        self.num_generations = kwargs.get('num_generations', 50)
        self.cx_prob = kwargs.get('cx_prob', 0.7)
        self.mut_prob = kwargs.get('mut_prob', 0.2)
        self.n_processes = kwargs.get('n_processes', 1)
        self.use_dask = kwargs.get('use_dask', False)
        self.use_spark = kwargs.get('use_spark', False)
        self.distributed_backend = kwargs.get('distributed_backend', 'auto')  # "multiprocessing" | "dask" | "spark" | "auto"
        self.adaptive_params = kwargs.get('adaptive_params', False)
        self.diversity_threshold = kwargs.get('diversity_threshold', 0.1)
        self.convergence_threshold = kwargs.get('convergence_threshold', 0.01)
        self.progress_callback = kwargs.get('progress_callback', None)
        self.local_search_enabled = kwargs.get('local_search_enabled', False)
        self.local_search_method = kwargs.get('local_search_method', 'hill_climbing')
        self.elite_fraction = kwargs.get('elite_fraction', 0.1)
        self.local_search_iterations = kwargs.get('local_search_iterations', 10)

        self.toolbox = None
        self.pop = None
        self.pool = None
        self.client = None
        self.spark_session = None

    def initialize(self) -> None:
        """Initialize the GA toolbox and population."""
        self.toolbox, _ = create_toolbox(self.num_features)
        eval_partial = partial(evaluate_hypothesis, data=self.data, target=self.target, chromosome_info=self.chromosome_info)
        self.toolbox.register("evaluate", eval_partial)

        # Setup multiprocessing or dask
        if self.use_dask:
            try:
                from dask.distributed import Client
                self.client = Client(processes=False, threads_per_worker=1, n_workers=self.n_processes)
                def dask_map(func, iterable):
                    futures = self.client.map(func, iterable)
                    return [f.result() for f in futures]
                if self.toolbox.map == map:
                    self.toolbox.register("map", dask_map)
            except ImportError:
                self.use_dask = False

        if not self.use_dask:
            if self.n_processes > 1:
                self.pool = multiprocessing.Pool(processes=self.n_processes)
                if self.toolbox.map == map:
                    self.toolbox.register("map", self.pool.map)
            elif self.toolbox.map == map:
                self.toolbox.register("map", map)

        # Create initial population
        self.pop = self.toolbox.population(n=self.pop_size)

        # Evaluate initial population
        fitnesses = self.toolbox.map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        # Determine evaluation backend
        self.evaluation_backend = "multiprocessing"
        if self.distributed_backend == "auto":
            if self.use_spark:
                self.evaluation_backend = "spark"
            elif self.use_dask:
                self.evaluation_backend = "dask"
        else:
            self.evaluation_backend = self.distributed_backend

        # Setup Spark if needed
        if self.evaluation_backend == "spark":
            try:
                from pyspark.sql import SparkSession
                self.spark_session = SparkSession.builder.getOrCreate()
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("PySpark not available, falling back to dask or multiprocessing")
                self.evaluation_backend = "dask" if self.use_dask else "multiprocessing"

    def evaluate(self, candidate: List[int]) -> Tuple[float, float, float, float]:
        """Evaluate a candidate hypothesis."""
        return evaluate_hypothesis(candidate, self.data, self.target, self.chromosome_info)

    def evaluate_with_spark(self, population: List[List[int]]) -> List[Tuple[float, ...]]:
        """Evaluate population using Spark distributed computing."""
        if not self.spark_session:
            try:
                from pyspark.sql import SparkSession
                self.spark_session = SparkSession.builder.getOrCreate()
            except ImportError:
                raise SparkUnavailableError("PySpark not available for Spark evaluation.")

        # Convert population to RDD and evaluate
        rdd = self.spark_session.sparkContext.parallelize(population)
        def eval_individual(ind):
            return self.evaluate(ind)
        results = rdd.map(eval_individual).collect()
        return results

    def evaluate_with_dask_distributed(self, population: List[List[int]]) -> List[Tuple[float, ...]]:
        """Enhanced Dask distributed evaluation with cluster support."""
        if not self.client:
            try:
                from dask.distributed import Client
                self.client = Client()  # Assume cluster is running
            except ImportError:
                raise DistributedComputationError("Dask distributed not available.")

        futures = self.client.map(self.evaluate, population)
        return [f.result() for f in futures]

    def hill_climbing(self, individual: List[int]) -> Tuple[List[int], float]:
        """Apply hill climbing local search to an individual."""
        current_fitness = sum(individual.fitness.values)
        best_individual = individual[:]
        best_fitness = current_fitness

        for _ in range(self.local_search_iterations):
            # Flip a random bit
            idx = random.randint(0, len(individual) - 1)
            new_individual = individual[:]
            new_individual[idx] = 1 - new_individual[idx]

            # Evaluate
            fitness = self.evaluate(new_individual)
            new_fitness = sum(fitness)

            if new_fitness > best_fitness:
                best_individual = new_individual
                best_fitness = new_fitness

        return best_individual, best_fitness

    def simulated_annealing(self, individual: List[int]) -> Tuple[List[int], float]:
        """Apply simulated annealing local search to an individual."""
        current = individual[:]
        current_fitness = sum(individual.fitness.values)
        best = current[:]
        best_fitness = current_fitness

        temperature = 1.0  # initial temperature
        cooling_rate = 0.95

        for _ in range(self.local_search_iterations):
            # Flip a random bit
            idx = random.randint(0, len(current) - 1)
            neighbor = current[:]
            neighbor[idx] = 1 - neighbor[idx]

            neighbor_fitness = sum(self.evaluate(neighbor))

            # Accept if better or with probability
            if neighbor_fitness > current_fitness or random.random() < math.exp((neighbor_fitness - current_fitness) / temperature):
                current = neighbor
                current_fitness = neighbor_fitness

            if current_fitness > best_fitness:
                best = current[:]
                best_fitness = current_fitness

            temperature *= cooling_rate

        return best, best_fitness

    def optimize(self) -> List[List[int]]:
        """Run the genetic algorithm optimization."""
        if self.pop is None:
            self.initialize()

        # Initialize adaptive parameters tracking
        if self.adaptive_params:
            fitness_sums = [sum(ind.fitness.values) for ind in self.pop]
            prev_best_fitness_sum = max(fitness_sums)

        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Starting GA evolution for {self.num_generations} generations with population size {self.pop_size}")

        # Early stopping parameters
        patience = 5  # Number of generations without improvement before stopping
        best_fitness_overall = float('-inf')
        generations_without_improvement = 0

        for gen in range(self.num_generations):
            logger.info(f"Generation {gen + 1}/{self.num_generations} starting")

            # Select offspring
            offspring = self.toolbox.select(self.pop, len(self.pop))
            logger.debug(f"Selected {len(offspring)} offspring")

            # Clone selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            crossover_count = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
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
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
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
                if self.evaluation_backend == "spark":
                    population_list = [list(ind) for ind in invalid_ind]
                    fitnesses = self.evaluate_with_spark(population_list)
                elif self.evaluation_backend == "dask":
                    population_list = [list(ind) for ind in invalid_ind]
                    fitnesses = self.evaluate_with_dask_distributed(population_list)
                else:
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                eval_time = time.time() - start_eval
                logger.info(f"Evaluation completed in {eval_time:.2f} seconds using {self.evaluation_backend}")

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            # Replace population
            self.pop[:] = offspring

            # Log best fitness
            best_fitness = max(sum(ind.fitness.values) for ind in self.pop)
            logger.info(f"Generation {gen + 1} completed. Best fitness: {best_fitness:.4f}")

            # Early stopping check
            improvement_threshold = 1e-6  # Minimum improvement to consider as better
            if best_fitness > best_fitness_overall + improvement_threshold:
                best_fitness_overall = best_fitness
                generations_without_improvement = 0
                logger.info(f"New best fitness found: {best_fitness:.4f}")
            else:
                generations_without_improvement += 1
                logger.info(f"No improvement for {generations_without_improvement} generations")

            if generations_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {gen + 1} generations due to no improvement for {patience} generations")
                break

            # Adaptive parameter adjustment
            if self.adaptive_params:
                fitness_sums = [sum(ind.fitness.values) for ind in self.pop]
                diversity = np.var(fitness_sums)
                current_best = max(fitness_sums)
                improvement = current_best - prev_best_fitness_sum
                # Adjust parameters based on diversity and convergence
                if diversity < self.diversity_threshold:
                    self.mut_prob = min(self.mut_prob + 0.05, 0.5)
                if improvement < self.convergence_threshold:
                    self.cx_prob = min(self.cx_prob + 0.05, 0.9)
                    self.mut_prob = min(self.mut_prob + 0.02, 0.5)
                prev_best_fitness_sum = current_best

            # Update progress
            if self.progress_callback:
                self.progress_callback()

        # Apply local search to elites
        if self.local_search_enabled:
            # Sort population by fitness
            sorted_pop = sorted(self.pop, key=lambda ind: sum(ind.fitness.values), reverse=True)
            num_elites = max(1, int(len(sorted_pop) * self.elite_fraction))
            elites = sorted_pop[:num_elites]

            for elite in elites:
                if self.local_search_method == 'hill_climbing':
                    improved, improved_fitness = self.hill_climbing(elite)
                elif self.local_search_method == 'simulated_annealing':
                    improved, improved_fitness = self.simulated_annealing(elite)
                else:
                    continue

                # If improved, update the elite in population
                if improved_fitness > sum(elite.fitness.values):
                    elite[:] = improved
                    elite.fitness.values = self.evaluate(improved)

        # Return Pareto front (best solutions)
        pareto_front = tools.sortNondominated(self.pop, len(self.pop), first_front_only=True)[0]

        # Cleanup
        if self.pool:
            self.pool.close()
            self.pool.join()
        if self.client:
            self.client.close()
        if self.spark_session:
            self.spark_session.stop()

        return pareto_front


class BayesianOptimization(Optimizer):
    """
    Bayesian Optimization optimizer for hypothesis search.

    Uses scikit-optimize for Gaussian process-based optimization on binary feature selection
    and engineering hypotheses.
    """

    def __init__(self, data: np.ndarray, target: np.ndarray, **kwargs):
        super().__init__(data, target, **kwargs)
        self.n_calls = kwargs.get('n_calls', 100)
        self.n_initial_points = kwargs.get('n_initial_points', 10)
        self.random_state = kwargs.get('random_state', 42)

    def initialize(self) -> None:
        """Initialize the BO space and objective."""
        try:
            from skopt import gp_minimize
            from skopt.space import Integer
            self.gp_minimize = gp_minimize
            self.space = [Integer(0, 1) for _ in range(self.chromosome_info['total_length'])]
        except ImportError:
            raise ImportError("scikit-optimize is required for BayesianOptimization. Install with: pip install scikit-optimize")

    def evaluate(self, candidate: List[int]) -> Tuple[float, float, float, float]:
        """Evaluate a candidate hypothesis."""
        return evaluate_hypothesis(candidate, self.data, self.target, self.chromosome_info)

    def optimize(self) -> List[List[int]]:
        """Run the Bayesian optimization."""
        if not hasattr(self, 'space'):
            self.initialize()

        def objective(x):
            fitness = self.evaluate(x)
            # gp_minimize minimizes, so return negative of the fitness (maximize significance + effect_size + coverage - parsimony)
            return -(fitness[0] + fitness[1] + fitness[2] - fitness[3])

        res = self.gp_minimize(objective, self.space, n_calls=self.n_calls, n_initial_points=self.n_initial_points, random_state=self.random_state)
        return [res.x]


class PSOOptimizer(Optimizer):
    """
    Particle Swarm Optimization optimizer for hypothesis search.

    Uses pyswarms for swarm-based optimization on binary feature selection
    and engineering hypotheses.
    """

    def __init__(self, data: np.ndarray, target: np.ndarray, **kwargs):
        super().__init__(data, target, **kwargs)
        self.n_particles = kwargs.get('n_particles', 30)
        self.iters = kwargs.get('iters', 100)
        self.c1 = kwargs.get('c1', 2.0)  # cognitive parameter
        self.c2 = kwargs.get('c2', 2.0)  # social parameter
        self.w = kwargs.get('w', 0.9)    # inertia weight
        self.k = kwargs.get('k', 3)      # number of neighbors for ring topology
        self.p = kwargs.get('p', 1)      # power to raise distance to
        self.random_state = kwargs.get('random_state', 42)

    def initialize(self) -> None:
        """Initialize the PSO optimizer."""
        try:
            from pyswarms.discrete import BinaryPSO
            self.BinaryPSO = BinaryPSO
        except ImportError:
            raise ImportError("pyswarms is required for PSOOptimizer. Install with: pip install pyswarms")

        # Cost function for pyswarms (minimizes cost)
        def cost_function(positions):
            # positions is (n_particles, dimensions)
            costs = []
            for pos in positions:
                fitness = self.evaluate(pos)
                # Minimize negative fitness (since we maximize fitness)
                cost = -(fitness[0] + fitness[1] + fitness[2] - fitness[3])
                costs.append(cost)
            return np.array(costs)

        dimensions = self.chromosome_info['total_length']
        self.optimizer = self.BinaryPSO(
            n_particles=self.n_particles,
            dimensions=dimensions,
            options={
                'c1': self.c1,
                'c2': self.c2,
                'w': self.w,
                'k': self.k,
                'p': self.p
            },
            init_pos=None,  # Random initialization
            random_state=self.random_state
        )
        self.cost_function = cost_function

    def evaluate(self, candidate: np.ndarray) -> Tuple[float, float, float, float]:
        """Evaluate a candidate hypothesis."""
        return evaluate_hypothesis(candidate, self.data, self.target, self.chromosome_info)

    def optimize(self) -> List[List[int]]:
        """Run the PSO optimization."""
        if not hasattr(self, 'optimizer'):
            self.initialize()

        # Run optimization
        cost, pos = self.optimizer.optimize(self.cost_function, iters=self.iters)

        # Return the best position as a list of lists (single best solution)
        return [pos.tolist()]
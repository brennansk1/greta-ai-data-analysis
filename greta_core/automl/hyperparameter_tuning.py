"""
Hyperparameter Tuning for AutoML

This module provides various hyperparameter optimization strategies including
grid search, random search, and Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import warnings
from dataclasses import dataclass
from enum import Enum
import time

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not available. Bayesian optimization will not be available.")

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    warnings.warn("hyperopt not available. Tree-structured Parzen Estimator optimization will not be available.")

from .base import BaseModel
from .evaluation import ModelEvaluator, CrossValidationStrategy


class TuningStrategy(Enum):
    """Enumeration of hyperparameter tuning strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    TPE_OPTIMIZATION = "tpe_optimization"


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: BaseModel
    all_scores: List[float]
    all_params: List[Dict[str, Any]]
    tuning_time: float
    n_iterations: int
    strategy: str


class ParameterSpace:
    """Parameter space definition for hyperparameter tuning."""

    def __init__(self):
        self.parameters = {}

    def add_real(self, name: str, low: float, high: float, prior: str = 'uniform'):
        """Add a real-valued parameter."""
        self.parameters[name] = {
            'type': 'real',
            'low': low,
            'high': high,
            'prior': prior
        }

    def add_integer(self, name: str, low: int, high: int):
        """Add an integer-valued parameter."""
        self.parameters[name] = {
            'type': 'integer',
            'low': low,
            'high': high
        }

    def add_categorical(self, name: str, choices: List[Any]):
        """Add a categorical parameter."""
        self.parameters[name] = {
            'type': 'categorical',
            'choices': choices
        }

    def get_skopt_space(self):
        """Convert to scikit-optimize parameter space."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")

        space = []
        for name, config in self.parameters.items():
            if config['type'] == 'real':
                space.append(Real(config['low'], config['high'],
                                prior=config['prior'], name=name))
            elif config['type'] == 'integer':
                space.append(Integer(config['low'], config['high'], name=name))
            elif config['type'] == 'categorical':
                space.append(Categorical(config['choices'], name=name))
        return space

    def get_hyperopt_space(self):
        """Convert to hyperopt parameter space."""
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for TPE optimization")

        space = {}
        for name, config in self.parameters.items():
            if config['type'] == 'real':
                space[name] = hp.uniform(name, config['low'], config['high'])
            elif config['type'] == 'integer':
                space[name] = hp.quniform(name, config['low'], config['high'], 1)
            elif config['type'] == 'categorical':
                space[name] = hp.choice(name, config['choices'])
        return space

    def sample_random(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """Sample random parameter combinations."""
        samples = []
        for _ in range(n_samples):
            params = {}
            for name, config in self.parameters.items():
                if config['type'] == 'real':
                    params[name] = np.random.uniform(config['low'], config['high'])
                elif config['type'] == 'integer':
                    params[name] = np.random.randint(config['low'], config['high'] + 1)
                elif config['type'] == 'categorical':
                    params[name] = np.random.choice(config['choices'])
            samples.append(params)
        return samples

    def get_grid(self) -> List[Dict[str, Any]]:
        """Generate grid of all parameter combinations."""
        if not self.parameters:
            return [{}]

        # For simplicity, limit grid size
        param_names = list(self.parameters.keys())
        param_values = []

        for name in param_names:
            config = self.parameters[name]
            if config['type'] == 'categorical':
                param_values.append(config['choices'])
            elif config['type'] == 'integer':
                # Sample 5 values for integers
                values = np.linspace(config['low'], config['high'], 5, dtype=int)
                param_values.append(values.tolist())
            elif config['type'] == 'real':
                # Sample 5 values for reals
                values = np.linspace(config['low'], config['high'], 5)
                param_values.append(values.tolist())

        # Generate all combinations
        from itertools import product
        combinations = list(product(*param_values))

        grid = []
        for combo in combinations:
            params = {}
            for i, name in enumerate(param_names):
                params[name] = combo[i]
            grid.append(params)

        return grid


class HyperparameterTuner(ABC):
    """Abstract base class for hyperparameter tuners."""

    def __init__(self, model_class: type, parameter_space: ParameterSpace,
                 cv_strategy: Optional[CrossValidationStrategy] = None,
                 scoring: Optional[Union[str, Callable]] = None,
                 random_state: int = 42):
        self.model_class = model_class
        self.parameter_space = parameter_space
        self.cv_strategy = cv_strategy
        self.scoring = scoring
        self.random_state = random_state
        self.evaluator = ModelEvaluator(cv_strategy)

    @abstractmethod
    def tune(self, X: pd.DataFrame, y: pd.Series, n_iterations: int = 50) -> TuningResult:
        """Perform hyperparameter tuning."""
        pass

    def _evaluate_params(self, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate a parameter set."""
        try:
            model = self.model_class(**params)
            model.fit(X, y)

            # Use cross-validation for evaluation
            if self.cv_strategy:
                splits = self.cv_strategy.get_splits(X, y)
                scores = []

                for train_idx, test_idx in splits:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    model_copy = self.model_class(**params)
                    model_copy.fit(X_train, y_train)
                    y_pred = model_copy.predict(X_test)

                    # Calculate score based on model type
                    if hasattr(model, 'predict_proba') and len(np.unique(y)) > 2:
                        # Multi-class classification
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_test, y_pred)
                    elif hasattr(model, 'predict_proba'):
                        # Binary classification
                        from sklearn.metrics import roc_auc_score
                        try:
                            y_prob = model_copy.predict_proba(X_test)[:, 1]
                            score = roc_auc_score(y_test, y_prob)
                        except:
                            from sklearn.metrics import accuracy_score
                            score = accuracy_score(y_test, y_pred)
                    else:
                        # Regression
                        from sklearn.metrics import r2_score
                        score = r2_score(y_test, y_pred)

                    scores.append(score)

                return np.mean(scores)
            else:
                # Simple train/test split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if hasattr(model, 'predict_proba') and len(np.unique(y)) > 2:
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y_test, y_pred)
                elif hasattr(model, 'predict_proba'):
                    from sklearn.metrics import roc_auc_score
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        return roc_auc_score(y_test, y_prob)
                    except:
                        from sklearn.metrics import accuracy_score
                        return accuracy_score(y_test, y_pred)
                else:
                    from sklearn.metrics import r2_score
                    return r2_score(y_test, y_pred)

        except Exception as e:
            warnings.warn(f"Error evaluating parameters {params}: {e}")
            return -np.inf  # Return worst possible score on error


class GridSearchTuner(HyperparameterTuner):
    """Grid search hyperparameter tuner."""

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iterations: Optional[int] = None) -> TuningResult:
        """Perform grid search."""
        start_time = time.time()

        param_grid = self.parameter_space.get_grid()
        if n_iterations and n_iterations < len(param_grid):
            # Randomly sample from grid if too large
            np.random.seed(self.random_state)
            indices = np.random.choice(len(param_grid), n_iterations, replace=False)
            param_grid = [param_grid[i] for i in indices]

        all_scores = []
        all_params = []

        best_score = -np.inf
        best_params = None
        best_model = None

        for params in param_grid:
            score = self._evaluate_params(params, X, y)
            all_scores.append(score)
            all_params.append(params)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = self.model_class(**best_params)
                best_model.fit(X, y)

        tuning_time = time.time() - start_time

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            all_scores=all_scores,
            all_params=all_params,
            tuning_time=tuning_time,
            n_iterations=len(param_grid),
            strategy="grid_search"
        )


class RandomSearchTuner(HyperparameterTuner):
    """Random search hyperparameter tuner."""

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iterations: int = 50) -> TuningResult:
        """Perform random search."""
        start_time = time.time()

        all_scores = []
        all_params = []

        best_score = -np.inf
        best_params = None
        best_model = None

        for _ in range(n_iterations):
            params = self.parameter_space.sample_random(1)[0]
            score = self._evaluate_params(params, X, y)
            all_scores.append(score)
            all_params.append(params)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = self.model_class(**best_params)
                best_model.fit(X, y)

        tuning_time = time.time() - start_time

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            all_scores=all_scores,
            all_params=all_params,
            tuning_time=tuning_time,
            n_iterations=n_iterations,
            strategy="random_search"
        )


class BayesianOptimizationTuner(HyperparameterTuner):
    """Bayesian optimization tuner using scikit-optimize."""

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iterations: int = 50) -> TuningResult:
        """Perform Bayesian optimization."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")

        start_time = time.time()

        # Convert parameter space
        space = self.parameter_space.get_skopt_space()

        # Define objective function
        @use_named_args(space)
        def objective(**params):
            score = self._evaluate_params(params, X, y)
            return -score  # Minimize (since skopt minimizes)

        # Perform optimization
        res = gp_minimize(
            objective,
            space,
            n_calls=n_iterations,
            random_state=self.random_state,
            verbose=False
        )

        # Extract results
        best_params = {}
        param_names = [dim.name for dim in space]
        for i, name in enumerate(param_names):
            best_params[name] = res.x[i]

        best_score = -res.fun  # Convert back to maximization
        best_model = self.model_class(**best_params)
        best_model.fit(X, y)

        # Get all evaluations
        all_scores = [-y for y in res.func_vals]
        all_params = []
        for x in res.x_iters:
            params = {}
            for i, name in enumerate(param_names):
                params[name] = x[i]
            all_params.append(params)

        tuning_time = time.time() - start_time

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            all_scores=all_scores,
            all_params=all_params,
            tuning_time=tuning_time,
            n_iterations=n_iterations,
            strategy="bayesian_optimization"
        )


class TPEOptimizationTuner(HyperparameterTuner):
    """TPE (Tree-structured Parzen Estimator) optimization tuner."""

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iterations: int = 50) -> TuningResult:
        """Perform TPE optimization."""
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for TPE optimization")

        start_time = time.time()

        # Convert parameter space
        space = self.parameter_space.get_hyperopt_space()

        # Define objective function
        def objective(params):
            score = self._evaluate_params(params, X, y)
            return {'loss': -score, 'status': STATUS_OK}  # Minimize

        # Perform optimization
        trials = Trials()
        best = fmin(
            objective,
            space,
            algo=tpe.suggest,
            max_evals=n_iterations,
            trials=trials,
            rstate=np.random.RandomState(self.random_state)
        )

        # Extract results
        best_params = best
        best_score = -trials.best_trial['result']['loss']
        best_model = self.model_class(**best_params)
        best_model.fit(X, y)

        # Get all evaluations
        all_scores = [-trial['result']['loss'] for trial in trials.trials]
        all_params = [trial['misc']['vals'] for trial in trials.trials]

        # Clean up parameter dictionaries
        cleaned_params = []
        for params in all_params:
            clean_params = {}
            for key, value in params.items():
                if isinstance(value, list) and len(value) == 1:
                    clean_params[key] = value[0]
                else:
                    clean_params[key] = value
            cleaned_params.append(clean_params)
        all_params = cleaned_params

        tuning_time = time.time() - start_time

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            all_scores=all_scores,
            all_params=all_params,
            tuning_time=tuning_time,
            n_iterations=n_iterations,
            strategy="tpe_optimization"
        )


class AutoTuner:
    """Automated hyperparameter tuner that selects the best strategy."""

    def __init__(self, model_class: type, parameter_space: ParameterSpace,
                 cv_strategy: Optional[CrossValidationStrategy] = None,
                 scoring: Optional[Union[str, Callable]] = None,
                 random_state: int = 42):
        self.model_class = model_class
        self.parameter_space = parameter_space
        self.cv_strategy = cv_strategy
        self.scoring = scoring
        self.random_state = random_state

        # Initialize available tuners
        self.tuners = {
            TuningStrategy.GRID_SEARCH: GridSearchTuner,
            TuningStrategy.RANDOM_SEARCH: RandomSearchTuner,
        }

        if SKOPT_AVAILABLE:
            self.tuners[TuningStrategy.BAYESIAN_OPTIMIZATION] = BayesianOptimizationTuner

        if HYPEROPT_AVAILABLE:
            self.tuners[TuningStrategy.TPE_OPTIMIZATION] = TPEOptimizationTuner

    def tune(self, X: pd.DataFrame, y: pd.Series,
             strategy: TuningStrategy = TuningStrategy.RANDOM_SEARCH,
             n_iterations: int = 50) -> TuningResult:
        """Perform hyperparameter tuning with specified strategy."""
        if strategy not in self.tuners:
            available_strategies = list(self.tuners.keys())
            raise ValueError(f"Strategy {strategy} not available. Available: {available_strategies}")

        tuner_class = self.tuners[strategy]
        tuner = tuner_class(
            self.model_class,
            self.parameter_space,
            self.cv_strategy,
            self.scoring,
            self.random_state
        )

        return tuner.tune(X, y, n_iterations)

    def tune_auto(self, X: pd.DataFrame, y: pd.Series, n_iterations: int = 50) -> TuningResult:
        """Automatically select and perform the best tuning strategy."""
        n_params = len(self.parameter_space.parameters)

        # For small parameter spaces, use grid search
        if n_params <= 3:
            param_grid = self.parameter_space.get_grid()
            if len(param_grid) <= 100:
                return self.tune(X, y, TuningStrategy.GRID_SEARCH, min(n_iterations, len(param_grid)))

        # For larger spaces, prefer Bayesian optimization if available
        if TuningStrategy.BAYESIAN_OPTIMIZATION in self.tuners:
            return self.tune(X, y, TuningStrategy.BAYESIAN_OPTIMIZATION, n_iterations)
        elif TuningStrategy.TPE_OPTIMIZATION in self.tuners:
            return self.tune(X, y, TuningStrategy.TPE_OPTIMIZATION, n_iterations)
        else:
            # Fallback to random search
            return self.tune(X, y, TuningStrategy.RANDOM_SEARCH, n_iterations)
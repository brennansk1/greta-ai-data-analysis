"""
Hyperparameter Tuning Framework

This module provides automated hyperparameter optimization using grid search,
random search, and Bayesian optimization techniques.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import uniform, randint, loguniform
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Automated hyperparameter tuning framework.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the hyperparameter tuner.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Define common hyperparameter spaces
        self.param_spaces = {
            'RandomForestClassifier': {
                'n_estimators': randint(50, 500),
                'max_depth': [None] + list(range(5, 50, 5)),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None] + list(np.arange(0.1, 1.1, 0.1)),
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            },
            'RandomForestRegressor': {
                'n_estimators': randint(50, 500),
                'max_depth': [None] + list(range(5, 50, 5)),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None] + list(np.arange(0.1, 1.1, 0.1)),
                'bootstrap': [True, False]
            },
            'XGBClassifier': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 5),
                'reg_alpha': loguniform(1e-5, 1),
                'reg_lambda': loguniform(1e-5, 1)
            },
            'XGBRegressor': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 5),
                'reg_alpha': loguniform(1e-5, 1),
                'reg_lambda': loguniform(1e-5, 1)
            },
            'LGBMClassifier': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3),
                'num_leaves': randint(20, 100),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': loguniform(1e-5, 1),
                'reg_lambda': loguniform(1e-5, 1)
            },
            'LGBMRegressor': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3),
                'num_leaves': randint(20, 100),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': loguniform(1e-5, 1),
                'reg_lambda': loguniform(1e-5, 1)
            },
            'SVC': {
                'C': loguniform(0.1, 100),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': randint(2, 5),
                'gamma': ['scale', 'auto'] + list(loguniform(0.001, 1).rvs(5)),
                'coef0': uniform(-1, 2)
            },
            'SVR': {
                'C': loguniform(0.1, 100),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': randint(2, 5),
                'gamma': ['scale', 'auto'] + list(loguniform(0.001, 1).rvs(5)),
                'epsilon': loguniform(0.01, 1)
            },
            'LogisticRegression': {
                'C': loguniform(0.001, 100),
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': randint(100, 1000)
            },
            'LinearRegression': {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            'KNeighborsClassifier': {
                'n_neighbors': randint(1, 20),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2]
            },
            'KNeighborsRegressor': {
                'n_neighbors': randint(1, 20),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2]
            }
        }

    def tune_hyperparameters(self, model, X: pd.DataFrame, y: pd.Series,
                           method: str = 'random', n_iter: int = 50,
                           scoring: Optional[str] = None, task_type: str = 'auto') -> Dict[str, Any]:
        """
        Tune hyperparameters for a given model.

        Args:
            model: Model instance to tune
            X: Feature matrix
            y: Target vector
            method: Tuning method ('grid', 'random', 'bayesian')
            n_iter: Number of iterations for random/bayesian search
            scoring: Scoring metric
            task_type: Type of ML task

        Returns:
            Dictionary containing tuning results
        """
        results = {}

        try:
            model_name = model.__class__.__name__

            # Get parameter space
            param_space = self._get_param_space(model_name, method)

            if not param_space:
                logger.warning(f"No parameter space defined for {model_name}, using default")
                param_space = {}

            # Get scoring function
            if scoring is None:
                scoring = self._get_default_scoring(task_type)

            scorer = make_scorer(scoring)

            # Perform tuning based on method
            if method == 'grid':
                results = self._grid_search(model, param_space, X, y, scorer)
            elif method == 'random':
                results = self._random_search(model, param_space, X, y, scorer, n_iter)
            elif method == 'bayesian':
                results = self._bayesian_search(model, param_space, X, y, scorer, n_iter)
            else:
                raise ValueError(f"Unknown tuning method: {method}")

            results['model_name'] = model_name
            results['method'] = method
            results['scoring'] = scoring

        except Exception as e:
            logger.error(f"Error tuning hyperparameters for {model.__class__.__name__}: {e}")
            results['error'] = str(e)

        return results

    def _grid_search(self, model, param_grid: Dict[str, List], X: pd.DataFrame,
                    y: pd.Series, scorer) -> Dict[str, Any]:
        """
        Perform grid search hyperparameter tuning.

        Args:
            model: Model to tune
            param_grid: Parameter grid
            X: Feature matrix
            y: Target vector
            scorer: Scoring function

        Returns:
            Dictionary containing grid search results
        """
        results = {}

        try:
            grid_search = GridSearchCV(
                model, param_grid, cv=self.cv_folds, scoring=scorer,
                n_jobs=self.n_jobs, verbose=1, return_train_score=False
            )

            grid_search.fit(X, y)

            results['best_params'] = grid_search.best_params_
            results['best_score'] = grid_search.best_score_
            results['cv_results'] = grid_search.cv_results_
            results['best_estimator'] = grid_search.best_estimator_

            # Calculate parameter importance (simplified)
            results['param_importance'] = self._calculate_param_importance(grid_search.cv_results_)

        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            results['error'] = str(e)

        return results

    def _random_search(self, model, param_distributions: Dict[str, Any], X: pd.DataFrame,
                      y: pd.Series, scorer, n_iter: int) -> Dict[str, Any]:
        """
        Perform random search hyperparameter tuning.

        Args:
            model: Model to tune
            param_distributions: Parameter distributions
            X: Feature matrix
            y: Target vector
            scorer: Scoring function
            n_iter: Number of iterations

        Returns:
            Dictionary containing random search results
        """
        results = {}

        try:
            random_search = RandomizedSearchCV(
                model, param_distributions, n_iter=n_iter, cv=self.cv_folds,
                scoring=scorer, n_jobs=self.n_jobs, verbose=1,
                random_state=self.random_state, return_train_score=False
            )

            random_search.fit(X, y)

            results['best_params'] = random_search.best_params_
            results['best_score'] = random_search.best_score_
            results['cv_results'] = random_search.cv_results_
            results['best_estimator'] = random_search.best_estimator_

            # Calculate parameter importance
            results['param_importance'] = self._calculate_param_importance(random_search.cv_results_)

        except Exception as e:
            logger.error(f"Error in random search: {e}")
            results['error'] = str(e)

        return results

    def _bayesian_search(self, model, param_space: Dict[str, Any], X: pd.DataFrame,
                        y: pd.Series, scorer, n_trials: int) -> Dict[str, Any]:
        """
        Perform Bayesian optimization hyperparameter tuning.

        Args:
            model: Model to tune
            param_space: Parameter space
            X: Feature matrix
            y: Target vector
            scorer: Scoring function
            n_trials: Number of trials

        Returns:
            Dictionary containing Bayesian optimization results
        """
        results = {}

        try:
            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state),
                pruner=MedianPruner()
            )

            # Define objective function
            def objective(trial):
                params = self._suggest_params(trial, param_space)
                model_copy = model.__class__(**params)

                # Perform cross-validation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(
                    model_copy, X, y, cv=self.cv_folds,
                    scoring=scorer, n_jobs=self.n_jobs
                )
                return np.mean(scores)

            # Run optimization
            study.optimize(objective, n_trials=n_trials, timeout=600)  # 10 minute timeout

            results['best_params'] = study.best_params
            results['best_score'] = study.best_value
            results['n_trials'] = len(study.trials)
            results['study'] = study

            # Get parameter importance
            try:
                results['param_importance'] = optuna.importance.get_param_importances(study)
            except Exception as e:
                logger.warning(f"Could not calculate parameter importance: {e}")

        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            results['error'] = str(e)

        return results

    def _get_param_space(self, model_name: str, method: str) -> Dict[str, Any]:
        """
        Get parameter space for a model.

        Args:
            model_name: Name of the model
            method: Tuning method

        Returns:
            Parameter space dictionary
        """
        if model_name in self.param_spaces:
            param_space = self.param_spaces[model_name].copy()

            # For grid search, convert distributions to lists
            if method == 'grid':
                param_space = self._distributions_to_lists(param_space)

            return param_space
        else:
            return {}

    def _distributions_to_lists(self, param_distributions: Dict[str, Any]) -> Dict[str, List]:
        """
        Convert parameter distributions to lists for grid search.

        Args:
            param_distributions: Parameter distributions

        Returns:
            Parameter grid
        """
        param_grid = {}

        for param_name, distribution in param_distributions.items():
            if hasattr(distribution, 'rvs'):
                # It's a scipy distribution, sample some values
                try:
                    if hasattr(distribution, 'a') and hasattr(distribution, 'b'):
                        # Uniform distribution
                        start, end = distribution.a, distribution.a + distribution.b
                        param_grid[param_name] = np.linspace(start, end, 5).tolist()
                    elif hasattr(distribution, 'a') and hasattr(distribution, 'b'):
                        # Integer distribution
                        start, end = distribution.a, distribution.a + distribution.b
                        param_grid[param_name] = list(range(int(start), int(end), max(1, int((end-start)/5))))
                    else:
                        # Sample from distribution
                        param_grid[param_name] = distribution.rvs(5, random_state=self.random_state).tolist()
                except Exception:
                    # Fallback to original
                    param_grid[param_name] = [distribution]
            else:
                # Already a list
                param_grid[param_name] = distribution

        return param_grid

    def _suggest_params(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest parameters for Optuna trial.

        Args:
            trial: Optuna trial
            param_space: Parameter space

        Returns:
            Suggested parameters
        """
        params = {}

        for param_name, distribution in param_space.items():
            if isinstance(distribution, list):
                # Categorical
                params[param_name] = trial.suggest_categorical(param_name, distribution)
            elif hasattr(distribution, 'rvs'):
                # Distribution
                if isinstance(distribution, randint):
                    params[param_name] = trial.suggest_int(param_name, distribution.a, distribution.b-1)
                elif isinstance(distribution, uniform):
                    params[param_name] = trial.suggest_float(param_name, distribution.a, distribution.a + distribution.b)
                elif isinstance(distribution, loguniform):
                    params[param_name] = trial.suggest_float(param_name, distribution.a, distribution.b, log=True)
                else:
                    # Default to uniform
                    params[param_name] = trial.suggest_float(param_name, 0, 1)
            else:
                # Default
                params[param_name] = distribution

        return params

    def _get_default_scoring(self, task_type: str) -> str:
        """
        Get default scoring metric for task type.

        Args:
            task_type: Type of ML task

        Returns:
            Scoring metric name
        """
        if task_type == 'classification':
            return 'f1_weighted'
        elif task_type == 'regression':
            return 'neg_mean_squared_error'
        elif task_type == 'clustering':
            return 'silhouette'
        else:
            return 'accuracy'

    def _calculate_param_importance(self, cv_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate parameter importance from CV results.

        Args:
            cv_results: Cross-validation results

        Returns:
            Dictionary of parameter importances
        """
        importance = {}

        try:
            # Simple importance calculation based on correlation with scores
            if 'mean_test_score' in cv_results and 'params' in cv_results:
                scores = cv_results['mean_test_score']
                params = cv_results['params']

                # Get all parameter names
                param_names = set()
                for param_dict in params:
                    param_names.update(param_dict.keys())

                for param_name in param_names:
                    param_values = []
                    corresponding_scores = []

                    for param_dict, score in zip(params, scores):
                        if param_name in param_dict:
                            param_values.append(param_dict[param_name])
                            corresponding_scores.append(score)

                    # Calculate correlation if we have numeric values
                    if param_values and all(isinstance(v, (int, float)) for v in param_values):
                        correlation = np.corrcoef(param_values, corresponding_scores)[0, 1]
                        importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0

        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")

        return importance

    def tune_multiple_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                           method: str = 'random', n_iter: int = 50,
                           task_type: str = 'auto') -> Dict[str, Dict[str, Any]]:
        """
        Tune hyperparameters for multiple models.

        Args:
            models: Dictionary of model instances
            X: Feature matrix
            y: Target vector
            method: Tuning method
            n_iter: Number of iterations
            task_type: Type of ML task

        Returns:
            Dictionary containing tuning results for each model
        """
        results = {}

        for model_name, model in models.items():
            logger.info(f"Tuning hyperparameters for {model_name}")
            results[model_name] = self.tune_hyperparameters(
                model, X, y, method=method, n_iter=n_iter, task_type=task_type
            )

        return results
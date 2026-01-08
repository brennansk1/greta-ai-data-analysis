"""
Model registry for managing ML models in the AutoML framework.
"""

from typing import Dict, Any, List, Optional, Type
import inspect

from .models import (
    BaseModel, SupervisedModel, UnsupervisedModel,
    RandomForestModel, XGBoostModel, LightGBMModel,
    NeuralNetworkModel, SVMModel, LogisticRegressionModel,
    ARIMAModel, ProphetModel, LSTMModel, GRUModel,
    KMeansModel, DBSCANModel, GaussianMixtureModel, PCAModel,
    BayesianLinearRegression, BayesianLogisticRegression, BayesianHypothesisTesting
)


class ModelRegistry:
    """
    Registry for ML models with metadata and capabilities.
    """

    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_models()

    def _register_builtin_models(self):
        """Register all built-in models with their metadata."""

        # Supervised Learning Models
        supervised_models = [
            (RandomForestModel, {
                'task_types': ['classification', 'regression'],
                'description': 'Ensemble method using decision trees',
                'strengths': ['Handles missing values', 'Feature importance', 'Non-parametric'],
                'weaknesses': ['Can overfit', 'Less interpretable than single trees'],
                'computational_cost': 'medium',
                'memory_usage': 'medium'
            }),
            (XGBoostModel, {
                'task_types': ['classification', 'regression'],
                'description': 'Gradient boosting with regularization',
                'strengths': ['High performance', 'Handles missing values', 'Regularization'],
                'weaknesses': ['Can overfit if not tuned', 'Slower training'],
                'computational_cost': 'high',
                'memory_usage': 'medium'
            }),
            (LightGBMModel, {
                'task_types': ['classification', 'regression'],
                'description': 'Lightweight gradient boosting',
                'strengths': ['Fast training', 'Low memory usage', 'Good accuracy'],
                'weaknesses': ['Can overfit', 'Less mature than XGBoost'],
                'computational_cost': 'medium',
                'memory_usage': 'low'
            }),
            (NeuralNetworkModel, {
                'task_types': ['classification', 'regression'],
                'description': 'Deep neural network',
                'strengths': ['High capacity', 'Feature learning', 'Scalable'],
                'weaknesses': ['Requires large datasets', 'Black box', 'Computationally intensive'],
                'computational_cost': 'very_high',
                'memory_usage': 'high'
            }),
            (SVMModel, {
                'task_types': ['classification', 'regression'],
                'description': 'Support Vector Machine',
                'strengths': ['Effective in high dimensions', 'Memory efficient'],
                'weaknesses': ['Slow on large datasets', 'Requires feature scaling'],
                'computational_cost': 'high',
                'memory_usage': 'medium'
            }),
            (LogisticRegressionModel, {
                'task_types': ['classification'],
                'description': 'Linear classification model',
                'strengths': ['Interpretable', 'Fast training', 'Probabilistic output'],
                'weaknesses': ['Assumes linear relationship', 'Sensitive to outliers'],
                'computational_cost': 'low',
                'memory_usage': 'low'
            })
        ]

        # Time Series Models
        time_series_models = [
            (ARIMAModel, {
                'task_types': ['forecasting'],
                'description': 'AutoRegressive Integrated Moving Average',
                'strengths': ['Interpretable', 'Good for stationary series'],
                'weaknesses': ['Assumes stationarity', 'Limited to univariate'],
                'computational_cost': 'low',
                'memory_usage': 'low'
            }),
            (ProphetModel, {
                'task_types': ['forecasting'],
                'description': 'Facebook Prophet for time series',
                'strengths': ['Handles seasonality', 'Missing data tolerant', 'Interpretable'],
                'weaknesses': ['Less flexible than ARIMA', 'Requires datetime index'],
                'computational_cost': 'medium',
                'memory_usage': 'medium'
            }),
            (LSTMModel, {
                'task_types': ['forecasting'],
                'description': 'Long Short-Term Memory neural network',
                'strengths': ['Captures long dependencies', 'Flexible architecture'],
                'weaknesses': ['Requires large datasets', 'Computationally intensive', 'Black box'],
                'computational_cost': 'very_high',
                'memory_usage': 'high'
            }),
            (GRUModel, {
                'task_types': ['forecasting'],
                'description': 'Gated Recurrent Unit neural network',
                'strengths': ['Similar to LSTM but simpler', 'Good performance'],
                'weaknesses': ['Requires large datasets', 'Less proven than LSTM'],
                'computational_cost': 'high',
                'memory_usage': 'high'
            })
        ]

        # Unsupervised Learning Models
        unsupervised_models = [
            (KMeansModel, {
                'task_types': ['clustering'],
                'description': 'K-means clustering algorithm',
                'strengths': ['Fast', 'Scalable', 'Simple to understand'],
                'weaknesses': ['Requires specifying k', 'Sensitive to initialization'],
                'computational_cost': 'low',
                'memory_usage': 'low'
            }),
            (DBSCANModel, {
                'task_types': ['clustering'],
                'description': 'Density-based clustering',
                'strengths': ['Doesn\'t require k', 'Handles arbitrary shapes', 'Robust to outliers'],
                'weaknesses': ['Slow on large datasets', 'Tuning eps and min_samples'],
                'computational_cost': 'medium',
                'memory_usage': 'medium'
            }),
            (GaussianMixtureModel, {
                'task_types': ['clustering'],
                'description': 'Gaussian mixture model clustering',
                'strengths': ['Probabilistic', 'Soft assignments', 'Flexible covariance'],
                'weaknesses': ['Assumes Gaussian distributions', 'Can converge to local optima'],
                'computational_cost': 'medium',
                'memory_usage': 'medium'
            }),
            (PCAModel, {
                'task_types': ['dimensionality_reduction'],
                'description': 'Principal Component Analysis',
                'strengths': ['Linear transformation', 'Preserves variance', 'Fast'],
                'weaknesses': ['Linear method', 'May not capture non-linear relationships'],
                'computational_cost': 'low',
                'memory_usage': 'low'
            })
        ]

        # Bayesian Models
        bayesian_models = [
            (BayesianLinearRegression, {
                'task_types': ['regression'],
                'description': 'Bayesian linear regression',
                'strengths': ['Uncertainty quantification', 'Regularization', 'Probabilistic'],
                'weaknesses': ['Assumes linear relationship', 'May be slower'],
                'computational_cost': 'medium',
                'memory_usage': 'medium'
            }),
            (BayesianLogisticRegression, {
                'task_types': ['classification'],
                'description': 'Bayesian logistic regression',
                'strengths': ['Uncertainty quantification', 'Probabilistic output'],
                'weaknesses': ['Assumes linear decision boundary'],
                'computational_cost': 'medium',
                'memory_usage': 'medium'
            }),
            (BayesianHypothesisTesting, {
                'task_types': ['hypothesis_testing'],
                'description': 'Bayesian hypothesis testing',
                'strengths': ['Quantifies uncertainty', 'No p-values', 'Intuitive interpretation'],
                'weaknesses': ['Computationally intensive', 'Requires MCMC sampling'],
                'computational_cost': 'high',
                'memory_usage': 'high'
            })
        ]

        # Register all models
        all_models = supervised_models + time_series_models + unsupervised_models + bayesian_models

        for model_class, metadata in all_models:
            self.register_model(model_class, metadata)

    def register_model(self, model_class: Type[BaseModel], metadata: Dict[str, Any]) -> None:
        """
        Register a model with its metadata.

        Args:
            model_class: The model class to register
            metadata: Dictionary containing model metadata
        """
        model_name = model_class.__name__

        # Get model signature for parameter inspection
        try:
            sig = inspect.signature(model_class.__init__)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'
        except Exception:
            params = []

        self._models[model_name] = {
            'class': model_class,
            'metadata': metadata,
            'parameters': params
        }

    def get_models_by_task(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve models suitable for a specific task type.

        Args:
            task_type: Type of ML task ('classification', 'regression', 'forecasting', etc.)

        Returns:
            List of model information dictionaries
        """
        matching_models = []

        for model_name, model_info in self._models.items():
            if task_type in model_info['metadata']['task_types']:
                matching_models.append({
                    'name': model_name,
                    'class': model_info['class'],
                    'metadata': model_info['metadata'],
                    'parameters': model_info['parameters']
                })

        return matching_models

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary or None if not found
        """
        if model_name in self._models:
            model_info = self._models[model_name]
            return {
                'name': model_name,
                'class': model_info['class'],
                'metadata': model_info['metadata'],
                'parameters': model_info['parameters']
            }
        return None

    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered models.

        Returns:
            List of all model information dictionaries
        """
        return [self.get_model_info(name) for name in self._models.keys()]

    def create_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create an instance of a registered model.

        Args:
            model_name: Name of the model to create
            config: Configuration parameters for the model

        Returns:
            Instantiated model

        Raises:
            ValueError: If model is not registered
        """
        model_info = self.get_model_info(model_name)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' is not registered")

        model_class = model_info['class']
        return model_class(config=config)

    def get_models_by_cost(self, max_cost: str = 'high') -> List[Dict[str, Any]]:
        """
        Get models filtered by computational cost.

        Args:
            max_cost: Maximum acceptable cost ('low', 'medium', 'high', 'very_high')

        Returns:
            List of models within cost constraints
        """
        cost_levels = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
        max_level = cost_levels.get(max_cost, 3)

        filtered_models = []
        for model_info in self.get_all_models():
            model_cost = model_info['metadata']['computational_cost']
            cost_level = cost_levels.get(model_cost, 3)
            if cost_level <= max_level:
                filtered_models.append(model_info)

        return filtered_models
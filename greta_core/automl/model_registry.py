"""
Model Registry Module

This module provides a centralized registry for all machine learning models
in the AutoML framework, including supervised, unsupervised, time series,
and Bayesian models.
"""

from typing import Dict, Type, Any, Optional, List
from abc import ABC, abstractmethod
import logging

from .supervised_models import SUPERVISED_MODEL_REGISTRY, BaseSupervisedModel
from .unsupervised_models import UNSUPERVISED_MODEL_REGISTRY, BaseUnsupervisedModel
from .time_series_models import TIME_SERIES_MODEL_REGISTRY, BaseTimeSeriesModel
from .bayesian_models import BAYESIAN_MODEL_REGISTRY, BaseBayesianModel, BayesianHypothesisTesting

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for all machine learning models."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Type]] = {
            'supervised': SUPERVISED_MODEL_REGISTRY,
            'unsupervised': UNSUPERVISED_MODEL_REGISTRY,
            'time_series': TIME_SERIES_MODEL_REGISTRY,
            'bayesian': BAYESIAN_MODEL_REGISTRY
        }
        self._model_categories = list(self._registry.keys())

    def get_model(self, model_name: str, model_type: str = 'supervised', **kwargs) -> Any:
        """Get a model instance by name and type."""
        if model_type not in self._registry:
            raise ValueError(f"Model type '{model_type}' not found. Available types: {self._model_categories}")

        registry = self._registry[model_type]
        if model_name not in registry:
            available_models = list(registry.keys())
            raise ValueError(f"Model '{model_name}' not found in {model_type} registry. Available models: {available_models}")

        model_class = registry[model_name]
        return model_class(**kwargs)

    def list_models(self, model_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all available models, optionally filtered by type."""
        if model_type:
            if model_type not in self._registry:
                raise ValueError(f"Model type '{model_type}' not found")
            return {model_type: list(self._registry[model_type].keys())}

        return {category: list(models.keys()) for category, models in self._registry.items()}

    def get_model_info(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model = self.get_model(model_name, model_type)
        return {
            'name': model_name,
            'type': model_type,
            'class': model.__class__.__name__,
            'module': model.__class__.__module__,
            'description': getattr(model, 'description', 'No description available'),
            'parameters': getattr(model, 'get_default_params', lambda: {})()
        }

    def register_model(self, model_class: Type, model_name: str, model_type: str) -> None:
        """Register a new model in the registry."""
        if model_type not in self._registry:
            self._registry[model_type] = {}
            self._model_categories.append(model_type)

        if model_name in self._registry[model_type]:
            logger.warning(f"Model '{model_name}' already exists in {model_type} registry. Overwriting.")

        self._registry[model_type][model_name] = model_class
        logger.info(f"Registered model '{model_name}' in {model_type} category")

    def unregister_model(self, model_name: str, model_type: str) -> None:
        """Remove a model from the registry."""
        if model_type not in self._registry:
            raise ValueError(f"Model type '{model_type}' not found")

        if model_name not in self._registry[model_type]:
            raise ValueError(f"Model '{model_name}' not found in {model_type} registry")

        del self._registry[model_type][model_name]
        logger.info(f"Unregistered model '{model_name}' from {model_type} category")

    def get_model_types(self) -> List[str]:
        """Get list of available model types."""
        return self._model_categories.copy()

    def validate_model_compatibility(self, model_name: str, model_type: str,
                                   data_shape: tuple, target_type: Optional[str] = None) -> bool:
        """Validate if a model is compatible with given data characteristics."""
        try:
            model = self.get_model(model_name, model_type)

            # Basic validation based on model type
            if model_type == 'supervised':
                if not isinstance(model, BaseSupervisedModel):
                    return False
                if target_type and hasattr(model, 'supported_target_types'):
                    return target_type in model.supported_target_types

            elif model_type == 'unsupervised':
                if not isinstance(model, BaseUnsupervisedModel):
                    return False

            elif model_type == 'time_series':
                if not isinstance(model, BaseTimeSeriesModel):
                    return False

            elif model_type == 'bayesian':
                if not isinstance(model, (BaseBayesianModel, BayesianHypothesisTesting)):
                    return False

            return True
        except Exception as e:
            logger.warning(f"Model validation failed for {model_name}: {e}")
            return False


# Global model registry instance
model_registry = ModelRegistry()


def get_model(model_name: str, model_type: str = 'supervised', **kwargs) -> Any:
    """Convenience function to get a model from the global registry."""
    return model_registry.get_model(model_name, model_type, **kwargs)


def list_available_models(model_type: Optional[str] = None) -> Dict[str, List[str]]:
    """Convenience function to list available models."""
    return model_registry.list_models(model_type)


def register_custom_model(model_class: Type, model_name: str, model_type: str) -> None:
    """Convenience function to register a custom model."""
    model_registry.register_model(model_class, model_name, model_type)
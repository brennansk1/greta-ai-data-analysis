"""
AutoML Module for GRETA

This module provides automated machine learning capabilities including:
- Automated model selection and training
- Hyperparameter tuning
- Cross-validation and evaluation
- Time series forecasting
- Unsupervised learning
- Bayesian inference
"""

from .models import (
    ModelRegistry,
    ModelType,
    BaseModel,
    SupervisedModel,
    UnsupervisedModel,
    TimeSeriesModel
)

from .pipeline import (
    AutoMLPipeline,
    AutoMLConfig,
    AutoMLResult,
    run_automl
)

from .evaluation import ModelEvaluator
from .tuning import AutoTuner

__all__ = [
    # Models
    'ModelRegistry',
    'ModelType',
    'BaseModel',
    'SupervisedModel',
    'UnsupervisedModel',
    'TimeSeriesModel',

    # Pipeline
    'AutoMLPipeline',
    'AutoMLConfig',
    'AutoMLResult',
    'run_automl',

    # Utilities
    'ModelEvaluator',
    'AutoTuner'
]
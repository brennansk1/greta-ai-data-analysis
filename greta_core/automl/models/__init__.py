"""
ML Model implementations for AutoML.

This module contains base classes and specific implementations for various
machine learning models used in the AutoML pipeline.
"""

from .base import BaseModel, SupervisedModel, UnsupervisedModel
from .supervised import (
    RandomForestModel, XGBoostModel, LightGBMModel,
    NeuralNetworkModel, SVMModel, LogisticRegressionModel
)
from .time_series import ARIMAModel, ProphetModel, LSTMModel, GRUModel
from .unsupervised import KMeansModel, DBSCANModel, GaussianMixtureModel, PCAModel
from .bayesian import BayesianLinearRegression, BayesianLogisticRegression, BayesianHypothesisTesting

__all__ = [
    # Base classes
    'BaseModel', 'SupervisedModel', 'UnsupervisedModel',
    # Supervised models
    'RandomForestModel', 'XGBoostModel', 'LightGBMModel',
    'NeuralNetworkModel', 'SVMModel', 'LogisticRegressionModel',
    # Time series models
    'ARIMAModel', 'ProphetModel', 'LSTMModel', 'GRUModel',
    # Unsupervised models
    'KMeansModel', 'DBSCANModel', 'GaussianMixtureModel', 'PCAModel',
    # Bayesian models
    'BayesianLinearRegression', 'BayesianLogisticRegression', 'BayesianHypothesisTesting'
]
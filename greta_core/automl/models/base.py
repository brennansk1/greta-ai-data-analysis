"""
Base classes for ML models in the AutoML framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the AutoML framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model: Optional[BaseEstimator] = None
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'BaseModel':
        """
        Fit the model to the training data.

        Args:
            X: Training features
            y: Target values (None for unsupervised models)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        """
        Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of current parameters
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.

        Args:
            **params: Parameter name-value pairs

        Returns:
            Self for method chaining
        """
        pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'config': self.config,
            'metadata': self.metadata
        }


class SupervisedModel(BaseModel):
    """
    Base class for supervised learning models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_type: Optional[str] = None  # 'classification' or 'regression'

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]) -> 'SupervisedModel':
        """
        Fit the supervised model.

        Args:
            X: Training features
            y: Target values

        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        # Determine task type
        if hasattr(y, 'dtype') and y.dtype in ['object', 'category', 'bool']:
            self.task_type = 'classification'
        elif len(np.unique(y)) < 20:  # heuristic for classification
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'

        self._fit(X, y)
        self.is_fitted = True
        return self

    @abstractmethod
    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        """Internal fit method to be implemented by subclasses."""
        pass

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Predict class probabilities for classification tasks.

        Args:
            X: Input features

        Returns:
            Class probabilities, or None if not applicable
        """
        return None


class UnsupervisedModel(BaseModel):
    """
    Base class for unsupervised learning models.
    """

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'UnsupervisedModel':
        """
        Fit the unsupervised model.

        Args:
            X: Training features
            y: Ignored for unsupervised learning

        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        self._fit(X)
        self.is_fitted = True
        return self

    @abstractmethod
    def _fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Internal fit method to be implemented by subclasses."""
        pass

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        """
        Make predictions (e.g., cluster assignments).

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self._predict(X)

    @abstractmethod
    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        """Internal predict method to be implemented by subclasses."""
        pass

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data (e.g., dimensionality reduction).

        Args:
            X: Input features

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming data")
        return self._transform(X)

    def _transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Internal transform method. Default implementation returns predictions."""
        return self._predict(X)
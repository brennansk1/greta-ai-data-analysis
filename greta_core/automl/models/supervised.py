"""
Supervised learning model implementations.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .base import SupervisedModel


class RandomForestModel(SupervisedModel):
    """Random Forest model for classification and regression."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.is_fitted or self.task_type != 'classification':
            return None
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'RandomForestModel':
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None
        if self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(enumerate(self.model.feature_importances_))


class XGBoostModel(SupervisedModel):
    """XGBoost model for classification and regression."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        self.default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        if self.task_type == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.is_fitted or self.task_type != 'classification':
            return None
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'XGBoostModel':
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None
        if self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(enumerate(self.model.feature_importances_))


class LightGBMModel(SupervisedModel):
    """LightGBM model for classification and regression."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        self.default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        if self.task_type == 'classification':
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.is_fitted or self.task_type != 'classification':
            return None
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'LightGBMModel':
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None
        if self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(enumerate(self.model.feature_importances_))


class NeuralNetworkModel(SupervisedModel):
    """Neural Network model using TensorFlow/Keras."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

        self.default_params = {
            'hidden_layers': [64, 32],
            'activation': 'relu',
            'output_activation': 'sigmoid' if self.task_type == 'classification' else 'linear',
            'optimizer': 'adam',
            'loss': 'binary_crossentropy' if self.task_type == 'classification' else 'mse',
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'verbose': 0
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        # Build model architecture
        self.model = keras.Sequential()

        # Input layer
        input_dim = X.shape[1]
        self.model.add(keras.layers.Input(shape=(input_dim,)))

        # Hidden layers
        for units in self.params['hidden_layers']:
            self.model.add(keras.layers.Dense(units, activation=self.params['activation']))

        # Output layer
        if self.task_type == 'classification':
            if len(np.unique(y)) > 2:
                # Multi-class
                output_units = len(np.unique(y))
                self.model.add(keras.layers.Dense(output_units, activation='softmax'))
                self.params['loss'] = 'sparse_categorical_crossentropy'
            else:
                # Binary
                self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        else:
            self.model.add(keras.layers.Dense(1, activation='linear'))

        # Compile model
        self.model.compile(
            optimizer=self.params['optimizer'],
            loss=self.params['loss'],
            metrics=['accuracy'] if self.task_type == 'classification' else ['mse']
        )

        # Fit model
        self.model.fit(
            X, y,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=self.params['validation_split'],
            verbose=self.params['verbose']
        )

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        predictions = self.model.predict(X, verbose=0)
        if self.task_type == 'classification':
            if predictions.shape[1] > 1:
                return np.argmax(predictions, axis=1)
            else:
                return (predictions > 0.5).astype(int).flatten()
        return predictions.flatten()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.is_fitted or self.task_type != 'classification':
            return None
        return self.model.predict(X, verbose=0)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'NeuralNetworkModel':
        self.params.update(params)
        return self


class SVMModel(SupervisedModel):
    """Support Vector Machine model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        if self.task_type == 'classification':
            self.model = SVC(**self.params)
        else:
            self.model = SVR(**self.params)
        self.model.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.is_fitted or self.task_type != 'classification':
            return None
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'SVMModel':
        self.params.update(params)
        return self


class LogisticRegressionModel(SupervisedModel):
    """Logistic Regression model for classification."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        if self.task_type != 'classification':
            raise ValueError("LogisticRegressionModel is only for classification tasks")
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.is_fitted:
            return None
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'LogisticRegressionModel':
        self.params.update(params)
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted or not hasattr(self.model, 'coef_'):
            return None
        # For binary classification, take absolute values of coefficients
        coef = np.abs(self.model.coef_).flatten()
        if self.feature_names:
            return dict(zip(self.feature_names, coef))
        return dict(enumerate(coef))
"""
Model classes for AutoML functionality.

This module defines the base model interfaces and concrete implementations
for supervised, unsupervised, and time series models.
"""

import abc
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration of supported model types."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    TIME_SERIES = "time_series"
    BAYESIAN = "bayesian"

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    name: str
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    random_state: Optional[int] = 42

class BaseModel(abc.ABC):
    """Abstract base class for all ML models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None

    @abc.abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'BaseModel':
        """Fit the model to training data."""
        pass

    @abc.abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        """Make predictions on new data."""
        pass

    @abc.abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available."""
        pass

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True

class SupervisedModel(BaseModel):
    """Base class for supervised learning models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.model_type != ModelType.SUPERVISED:
            raise ValueError("Model type must be SUPERVISED")

    @abc.abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities for classification tasks."""
        pass

    def score(self, X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series]) -> float:
        """Calculate model score."""
        from sklearn.metrics import accuracy_score, r2_score
        predictions = self.predict(X)

        if hasattr(self, 'classes_'):  # Classification
            return accuracy_score(y, predictions)
        else:  # Regression
            return r2_score(y, predictions)

class UnsupervisedModel(BaseModel):
    """Base class for unsupervised learning models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.model_type != ModelType.UNSUPERVISED:
            raise ValueError("Model type must be UNSUPERVISED")

    @abc.abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using the fitted model."""
        pass

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit model and return predictions."""
        return self.fit(X).predict(X)

class TimeSeriesModel(BaseModel):
    """Base class for time series models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.model_type != ModelType.TIME_SERIES:
            raise ValueError("Model type must be TIME_SERIES")

    @abc.abstractmethod
    def forecast(self, steps: int) -> pd.Series:
        """Forecast future values."""
        pass

    @abc.abstractmethod
    def detect_anomalies(self, X: Union[np.ndarray, pd.DataFrame],
                        threshold: float = 0.95) -> pd.Series:
        """Detect anomalies in time series data."""
        pass

# Concrete Model Implementations

class RandomForestClassifier(SupervisedModel):
    """Random Forest Classifier implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            **config.hyperparameters,
            random_state=config.random_state
        )

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

class RandomForestRegressor(SupervisedModel):
    """Random Forest Regressor implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            **config.hyperparameters,
            random_state=config.random_state
        )

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError("Regression models don't support predict_proba")

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

class XGBoostClassifier(SupervisedModel):
    """XGBoost Classifier implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                **config.hyperparameters,
                random_state=config.random_state
            )
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

class XGBoostRegressor(SupervisedModel):
    """XGBoost Regressor implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                **config.hyperparameters,
                random_state=config.random_state
            )
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError("Regression models don't support predict_proba")

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

class LightGBMClassifier(SupervisedModel):
    """LightGBM Classifier implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                **config.hyperparameters,
                random_state=config.random_state
            )
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

class LightGBMRegressor(SupervisedModel):
    """LightGBM Regressor implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(
                **config.hyperparameters,
                random_state=config.random_state
            )
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError("Regression models don't support predict_proba")

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

class SVMClassifier(SupervisedModel):
    """Support Vector Machine Classifier implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.svm import SVC
        self.model = SVC(
            **config.hyperparameters,
            random_state=config.random_state,
            probability=True
        )

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        # SVM doesn't have built-in feature importance
        return None

class LogisticRegression(SupervisedModel):
    """Logistic Regression implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            **config.hyperparameters,
            random_state=config.random_state
        )

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, np.abs(self.model.coef_[0])))

class LinearRegression(SupervisedModel):
    """Linear Regression implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(**config.hyperparameters)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError("Regression models don't support predict_proba")

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, np.abs(self.model.coef_)))

class NeuralNetworkClassifier(SupervisedModel):
    """Neural Network Classifier implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(
            **config.hyperparameters,
            random_state=config.random_state
        )

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        # Neural networks don't have straightforward feature importance
        return None

class NeuralNetworkRegressor(SupervisedModel):
    """Neural Network Regressor implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.neural_network import MLPRegressor
        self.model = MLPRegressor(
            **config.hyperparameters,
            random_state=config.random_state
        )

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")
        self.model.fit(X, y)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError("Regression models don't support predict_proba")

    def get_feature_importance(self):
        # Neural networks don't have straightforward feature importance
        return None

# Unsupervised Models

class KMeans(UnsupervisedModel):
    """K-Means clustering implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.cluster import KMeans
        self.model = KMeans(
            **config.hyperparameters,
            random_state=config.random_state
        )

    def fit(self, X, y=None):
        self.model.fit(X)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def transform(self, X):
        return self.model.transform(X)

    def get_feature_importance(self):
        return None

class DBSCAN(UnsupervisedModel):
    """DBSCAN clustering implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.cluster import DBSCAN
        self.model = DBSCAN(**config.hyperparameters)

    def fit(self, X, y=None):
        self.model.fit(X)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.fit_predict(X)

    def transform(self, X):
        # DBSCAN doesn't have a transform method
        raise NotImplementedError("DBSCAN doesn't support transform")

    def get_feature_importance(self):
        return None

class GaussianMixture(UnsupervisedModel):
    """Gaussian Mixture Model implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.mixture import GaussianMixture
        self.model = GaussianMixture(
            **config.hyperparameters,
            random_state=config.random_state
        )

    def fit(self, X, y=None):
        self.model.fit(X)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def transform(self, X):
        return self.model.predict_proba(X)  # Return probabilities as transform

    def get_feature_importance(self):
        return None

class PCA(UnsupervisedModel):
    """Principal Component Analysis implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from sklearn.decomposition import PCA
        self.model = PCA(**config.hyperparameters)

    def fit(self, X, y=None):
        self.model.fit(X)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        return self.model.transform(X)

    def transform(self, X):
        return self.model.transform(X)

    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        return dict(zip(self.feature_names, self.model.explained_variance_ratio_))

# Time Series Models

class ARIMAModel(TimeSeriesModel):
    """ARIMA time series model implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.ARIMA = ARIMA
        except ImportError:
            raise ImportError("statsmodels not installed. Install with: pip install statsmodels")

    def fit(self, X, y=None):
        # X should be a time series (1D array-like)
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("ARIMA requires univariate time series")
            X = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            pass
        else:
            X = pd.Series(X)

        self.model = self.ARIMA(X, **self.config.hyperparameters)
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self

    def predict(self, X):
        # For ARIMA, predict returns in-sample predictions
        return self.fitted_model.predict()

    def forecast(self, steps: int):
        return self.fitted_model.forecast(steps=steps)

    def detect_anomalies(self, X, threshold=0.95):
        # Simple anomaly detection based on prediction intervals
        predictions = self.fitted_model.predict()
        residuals = X - predictions
        threshold_value = np.percentile(np.abs(residuals), threshold * 100)
        return pd.Series(np.abs(residuals) > threshold_value, index=X.index)

    def get_feature_importance(self):
        return None

class ProphetModel(TimeSeriesModel):
    """Facebook Prophet time series model implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            raise ImportError("prophet not installed. Install with: pip install prophet")

    def fit(self, X, y=None):
        # Prophet expects a DataFrame with 'ds' and 'y' columns
        if isinstance(X, pd.DataFrame):
            if 'ds' not in X.columns or 'y' not in X.columns:
                raise ValueError("Prophet requires DataFrame with 'ds' and 'y' columns")
            df = X.copy()
        else:
            # Assume X is a time series, create ds column
            df = pd.DataFrame({'ds': pd.date_range(start='2020-01-01', periods=len(X)), 'y': X})

        self.model = self.Prophet(**self.config.hyperparameters)
        self.model.fit(df)
        self.is_fitted = True
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and 'ds' in X.columns:
            return self.model.predict(X)['yhat']
        else:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=len(X))
            forecast = self.model.predict(future)
            return forecast['yhat'].tail(len(X))

    def forecast(self, steps: int):
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(steps)

    def detect_anomalies(self, X, threshold=0.95):
        # Use Prophet's built-in anomaly detection
        if isinstance(X, pd.DataFrame) and 'ds' in X.columns:
            forecast = self.model.predict(X)
        else:
            future = self.model.make_future_dataframe(periods=len(X))
            forecast = self.model.predict(future)

        residuals = X['y'] - forecast['yhat'] if isinstance(X, pd.DataFrame) else X - forecast['yhat']
        threshold_value = np.percentile(np.abs(residuals), threshold * 100)
        return pd.Series(np.abs(residuals) > threshold_value)

    def get_feature_importance(self):
        return None

class LSTMModel(TimeSeriesModel):
    """LSTM time series model implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
        except ImportError:
            raise ImportError("tensorflow not installed. Install with: pip install tensorflow")

    def fit(self, X, y=None):
        # Simple LSTM implementation
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        # Reshape for LSTM [samples, time steps, features]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        self.model = self.Sequential()
        self.model.add(self.LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        self.model.add(self.Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(X, X[:, -1, :], epochs=self.config.hyperparameters.get('epochs', 100), verbose=0)
        self.is_fitted = True
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X).flatten()

    def forecast(self, steps: int):
        # Simple forecasting by repeating last prediction
        last_value = self.model.predict(np.array([self.last_input]))[0][0]
        forecast = [last_value] * steps
        return pd.Series(forecast)

    def detect_anomalies(self, X, threshold=0.95):
        predictions = self.predict(X)
        residuals = X - predictions
        threshold_value = np.percentile(np.abs(residuals), threshold * 100)
        return pd.Series(np.abs(residuals) > threshold_value)

    def get_feature_importance(self):
        return None

# Bayesian Models

class BayesianLinearRegression(SupervisedModel):
    """Bayesian Linear Regression implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import pymc3 as pm
            self.pm = pm
        except ImportError:
            raise ImportError("pymc3 not installed. Install with: pip install pymc3")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")

        with self.pm.Model() as model:
            # Priors
            alpha = self.pm.Normal('alpha', mu=0, sd=10)
            beta = self.pm.Normal('beta', mu=0, sd=10, shape=X.shape[1])
            sigma = self.pm.HalfNormal('sigma', sd=1)

            # Expected value
            mu = alpha + self.pm.math.dot(X, beta)

            # Likelihood
            y_obs = self.pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)

            # Inference
            self.trace = self.pm.sample(**self.config.hyperparameters)

        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        # Use posterior predictive mean
        with self.pm.Model():
            alpha = self.trace['alpha'].mean()
            beta = self.trace['beta'].mean()
            return alpha + np.dot(X, beta)

    def predict_proba(self, X):
        raise NotImplementedError("Bayesian models don't support predict_proba")

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, np.abs(self.trace['beta'].mean(axis=0))))

class BayesianLogisticRegression(SupervisedModel):
    """Bayesian Logistic Regression implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import pymc3 as pm
            self.pm = pm
        except ImportError:
            raise ImportError("pymc3 not installed. Install with: pip install pymc3")

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required for supervised learning")

        with self.pm.Model() as model:
            # Priors
            alpha = self.pm.Normal('alpha', mu=0, sd=10)
            beta = self.pm.Normal('beta', mu=0, sd=10, shape=X.shape[1])

            # Expected value
            mu = alpha + self.pm.math.dot(X, beta)
            p = self.pm.math.sigmoid(mu)

            # Likelihood
            y_obs = self.pm.Bernoulli('y_obs', p=p, observed=y)

            # Inference
            self.trace = self.pm.sample(**self.config.hyperparameters)

        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        # Use posterior predictive mean
        with self.pm.Model():
            alpha = self.trace['alpha'].mean()
            beta = self.trace['beta'].mean()
            mu = alpha + np.dot(X, beta)
            p = 1 / (1 + np.exp(-mu))
            return (p > 0.5).astype(int)

    def predict_proba(self, X):
        with self.pm.Model():
            alpha = self.trace['alpha'].mean()
            beta = self.trace['beta'].mean()
            mu = alpha + np.dot(X, beta)
            p = 1 / (1 + np.exp(-mu))
            return np.column_stack([1-p, p])

    def get_feature_importance(self):
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, np.abs(self.trace['beta'].mean(axis=0))))

class ModelRegistry:
    """Registry for managing available models."""

    def __init__(self):
        self.models = {
            # Supervised Models
            'random_forest_classifier': RandomForestClassifier,
            'random_forest_regressor': RandomForestRegressor,
            'xgboost_classifier': XGBoostClassifier,
            'xgboost_regressor': XGBoostRegressor,
            'lightgbm_classifier': LightGBMClassifier,
            'lightgbm_regressor': LightGBMRegressor,
            'svm_classifier': SVMClassifier,
            'logistic_regression': LogisticRegression,
            'linear_regression': LinearRegression,
            'neural_network_classifier': NeuralNetworkClassifier,
            'neural_network_regressor': NeuralNetworkRegressor,

            # Unsupervised Models
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'gaussian_mixture': GaussianMixture,
            'pca': PCA,

            # Time Series Models
            'arima': ARIMAModel,
            'prophet': ProphetModel,
            'lstm': LSTMModel,

            # Bayesian Models
            'bayesian_linear_regression': BayesianLinearRegression,
            'bayesian_logistic_regression': BayesianLogisticRegression,
        }

    def get_model_class(self, model_name: str):
        """Get model class by name."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]

    def create_model(self, model_name: str, config: ModelConfig):
        """Create a model instance."""
        model_class = self.get_model_class(model_name)
        return model_class(config)

    def list_models(self, model_type: Optional[ModelType] = None) -> List[str]:
        """List available models, optionally filtered by type."""
        if model_type is None:
            return list(self.models.keys())

        type_mapping = {
            ModelType.SUPERVISED: ['random_forest_classifier', 'random_forest_regressor',
                                  'xgboost_classifier', 'xgboost_regressor',
                                  'lightgbm_classifier', 'lightgbm_regressor',
                                  'svm_classifier', 'logistic_regression', 'linear_regression',
                                  'neural_network_classifier', 'neural_network_regressor',
                                  'bayesian_linear_regression', 'bayesian_logistic_regression'],
            ModelType.UNSUPERVISED: ['kmeans', 'dbscan', 'gaussian_mixture', 'pca'],
            ModelType.TIME_SERIES: ['arima', 'prophet', 'lstm'],
            ModelType.BAYESIAN: ['bayesian_linear_regression', 'bayesian_logistic_regression']
        }

        return type_mapping.get(model_type, [])
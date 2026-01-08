"""
Time Series Models Module

This module contains implementations of time series forecasting models
including ARIMA, Prophet, LSTM, and anomaly detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class BaseTimeSeriesModel(ABC):
    """Abstract base class for time series models."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> None:
        """Fit the model to the time series data."""
        pass

    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Make predictions for future steps."""
        pass

    @abstractmethod
    def detect_anomalies(self, data: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect anomalies in the time series data."""
        pass


class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA model for time series forecasting."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        super().__init__('arima')
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, data: pd.Series, **kwargs) -> None:
        """Fit the ARIMA model."""
        try:
            if self.seasonal_order:
                self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(data, order=self.order)
            self.results = self.model.fit()
            self.is_fitted = True
            logger.info("ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            raise

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Make predictions for future steps."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            forecast = self.results.forecast(steps=steps)
            return forecast.values
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise

    def detect_anomalies(self, data: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect anomalies using residuals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")

        # Get fitted values and residuals
        fitted_values = self.results.fittedvalues
        residuals = data - fitted_values

        # Calculate z-scores of residuals
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        z_scores = (residuals - residual_mean) / residual_std

        # Identify anomalies
        anomalies = np.abs(z_scores) > threshold
        return pd.Series(anomalies, index=data.index)


class ProphetModel(BaseTimeSeriesModel):
    """Facebook Prophet model for time series forecasting."""

    def __init__(self, seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0):
        super().__init__('prophet')
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale

    def fit(self, data: pd.Series, **kwargs) -> None:
        """Fit the Prophet model."""
        try:
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            df = pd.DataFrame({'ds': data.index, 'y': data.values})
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale
            )
            self.model.fit(df)
            self.is_fitted = True
            logger.info("Prophet model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit Prophet model: {e}")
            raise

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Make predictions for future steps."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            future = self.model.make_future_dataframe(periods=steps)
            forecast = self.model.predict(future)
            return forecast['yhat'].tail(steps).values
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise

    def detect_anomalies(self, data: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect anomalies using Prophet's uncertainty intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")

        try:
            # Create future dataframe for historical predictions
            df = pd.DataFrame({'ds': data.index, 'y': data.values})
            forecast = self.model.predict(df)

            # Calculate residuals
            residuals = data.values - forecast['yhat'].values

            # Calculate z-scores
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            z_scores = (residuals - residual_mean) / residual_std

            # Identify anomalies
            anomalies = np.abs(z_scores) > threshold
            return pd.Series(anomalies, index=data.index)
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise


class LSTMModel(BaseTimeSeriesModel):
    """LSTM model for time series forecasting."""

    def __init__(self, units: int = 50, dropout: float = 0.2,
                 epochs: int = 100, batch_size: int = 32,
                 sequence_length: int = 10):
        super().__init__('lstm')
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def fit(self, data: pd.Series, **kwargs) -> None:
        """Fit the LSTM model."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

            # Create sequences
            X, y = self._create_sequences(scaled_data)

            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build LSTM model
            self.model = Sequential([
                LSTM(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(self.dropout),
                LSTM(self.units // 2),
                Dropout(self.dropout),
                Dense(1)
            ])

            self.model.compile(optimizer='adam', loss='mse')

            # Early stopping
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

            # Fit the model
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                          callbacks=[early_stopping], verbose=0)

            self.is_fitted = True
            logger.info("LSTM model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit LSTM model: {e}")
            raise

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Make predictions for future steps."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            predictions = []
            current_sequence = kwargs.get('last_sequence')

            if current_sequence is None:
                raise ValueError("last_sequence must be provided for LSTM prediction")

            for _ in range(steps):
                # Reshape for prediction
                current_sequence = current_sequence.reshape((1, self.sequence_length, 1))

                # Make prediction
                pred = self.model.predict(current_sequence, verbose=0)[0][0]

                # Append prediction
                predictions.append(pred)

                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred

            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()

            return predictions
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise

    def detect_anomalies(self, data: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect anomalies using reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")

        try:
            # Scale the data
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1)).flatten()

            # Create sequences
            X, y = self._create_sequences(scaled_data)

            # Get predictions
            predictions = self.model.predict(X, verbose=0).flatten()

            # Calculate reconstruction error
            errors = np.abs(y - predictions)

            # Calculate threshold
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            anomaly_threshold = mean_error + threshold * std_error

            # Identify anomalies
            anomalies = errors > anomaly_threshold

            # Create full-length anomaly series
            anomaly_series = pd.Series(False, index=data.index)
            anomaly_series.iloc[self.sequence_length:] = anomalies

            return anomaly_series
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise


class GRUModel(BaseTimeSeriesModel):
    """GRU model for time series forecasting."""

    def __init__(self, units: int = 50, dropout: float = 0.2,
                 epochs: int = 100, batch_size: int = 32,
                 sequence_length: int = 10):
        super().__init__('gru')
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def fit(self, data: pd.Series, **kwargs) -> None:
        """Fit the GRU model."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

            # Create sequences
            X, y = self._create_sequences(scaled_data)

            # Reshape for GRU [samples, time steps, features]
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build GRU model
            self.model = Sequential([
                GRU(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(self.dropout),
                GRU(self.units // 2),
                Dropout(self.dropout),
                Dense(1)
            ])

            self.model.compile(optimizer='adam', loss='mse')

            # Early stopping
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

            # Fit the model
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                          callbacks=[early_stopping], verbose=0)

            self.is_fitted = True
            logger.info("GRU model fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit GRU model: {e}")
            raise

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Make predictions for future steps."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            predictions = []
            current_sequence = kwargs.get('last_sequence')

            if current_sequence is None:
                raise ValueError("last_sequence must be provided for GRU prediction")

            for _ in range(steps):
                # Reshape for prediction
                current_sequence = current_sequence.reshape((1, self.sequence_length, 1))

                # Make prediction
                pred = self.model.predict(current_sequence, verbose=0)[0][0]

                # Append prediction
                predictions.append(pred)

                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred

            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()

            return predictions
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise

    def detect_anomalies(self, data: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect anomalies using reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")

        try:
            # Scale the data
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1)).flatten()

            # Create sequences
            X, y = self._create_sequences(scaled_data)

            # Get predictions
            predictions = self.model.predict(X, verbose=0).flatten()

            # Calculate reconstruction error
            errors = np.abs(y - predictions)

            # Calculate threshold
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            anomaly_threshold = mean_error + threshold * std_error

            # Identify anomalies
            anomalies = errors > anomaly_threshold

            # Create full-length anomaly series
            anomaly_series = pd.Series(False, index=data.index)
            anomaly_series.iloc[self.sequence_length:] = anomalies

            return anomaly_series
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise


# Time series model registry
TIME_SERIES_MODEL_REGISTRY = {
    'arima': ARIMAModel,
    'prophet': ProphetModel,
    'lstm': LSTMModel,
    'gru': GRUModel,
}


def get_time_series_model(model_name: str, **kwargs) -> BaseTimeSeriesModel:
    """Factory function to create a time series model instance."""
    if model_name not in TIME_SERIES_MODEL_REGISTRY:
        raise ValueError(f"Time series model '{model_name}' not found in registry")
    return TIME_SERIES_MODEL_REGISTRY[model_name](**kwargs)
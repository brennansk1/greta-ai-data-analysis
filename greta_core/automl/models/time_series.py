"""
Time series forecasting model implementations.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .base import SupervisedModel


class ARIMAModel(SupervisedModel):
    """ARIMA model for time series forecasting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_type = 'forecasting'
        self.default_params = {
            'order': (1, 1, 1),  # (p, d, q)
            'seasonal_order': (0, 0, 0, 0),  # (P, D, Q, s)
            'trend': 'c'  # constant trend
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        # For time series, X is typically the time series data itself
        if isinstance(y, pd.Series) and hasattr(y.index, 'freq'):
            # Use SARIMAX if seasonal component is specified
            if any(self.params['seasonal_order']):
                self.model = SARIMAX(y, order=self.params['order'],
                                   seasonal_order=self.params['seasonal_order'],
                                   trend=self.params['trend'])
            else:
                self.model = ARIMA(y, order=self.params['order'])
        else:
            # Assume X contains the time series
            if isinstance(X, pd.DataFrame):
                ts_data = X.iloc[:, 0] if X.shape[1] == 1 else y
            else:
                ts_data = X.flatten() if X.ndim > 1 else X

            if any(self.params['seasonal_order']):
                self.model = SARIMAX(ts_data, order=self.params['order'],
                                   seasonal_order=self.params['seasonal_order'],
                                   trend=self.params['trend'])
            else:
                self.model = ARIMA(ts_data, order=self.params['order'])

        self.fitted_model = self.model.fit()

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # For forecasting, X represents the number of steps to forecast
        if isinstance(X, (int, np.integer)):
            steps = int(X)
        elif hasattr(X, '__len__') and len(X) == 1:
            steps = int(X[0])
        else:
            raise ValueError("For forecasting, X should be the number of steps to forecast")

        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'ARIMAModel':
        self.params.update(params)
        return self


class ProphetModel(SupervisedModel):
    """Facebook Prophet model for time series forecasting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")

        self.task_type = 'forecasting'
        self.default_params = {
            'yearly_seasonality': 'auto',
            'weekly_seasonality': 'auto',
            'daily_seasonality': 'auto',
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        # Prophet requires a DataFrame with 'ds' (date) and 'y' (value) columns
        if isinstance(X, pd.DataFrame) and 'ds' in X.columns:
            df = X.copy()
        elif isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            df = pd.DataFrame({'ds': y.index, 'y': y.values})
        else:
            # Assume X is datetime and y is values
            if hasattr(X, '__iter__') and hasattr(y, '__iter__'):
                df = pd.DataFrame({'ds': pd.to_datetime(X), 'y': y})
            else:
                raise ValueError("Prophet requires datetime index or 'ds' column")

        self.model = Prophet(**self.params)
        self.model.fit(df)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Create future dataframe
        if isinstance(X, (int, np.integer)):
            # Number of periods to forecast
            future = self.model.make_future_dataframe(periods=int(X))
        elif isinstance(X, pd.DataFrame) and 'ds' in X.columns:
            future = X
        else:
            # Assume X contains future dates
            future = pd.DataFrame({'ds': pd.to_datetime(X)})

        forecast = self.model.predict(future)
        return forecast['yhat'].values

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'ProphetModel':
        self.params.update(params)
        return self


class LSTMModel(SupervisedModel):
    """LSTM neural network for time series forecasting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

        self.task_type = 'forecasting'
        self.default_params = {
            'units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 10,
            'optimizer': 'adam',
            'loss': 'mse',
            'validation_split': 0.2,
            'verbose': 0
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        # Prepare sequences for LSTM
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = np.array(X)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Create sequences
        sequences = []
        targets = []
        seq_len = self.params['sequence_length']

        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
            if isinstance(y, (pd.Series, np.ndarray)):
                targets.append(y[i+seq_len] if i+seq_len < len(y) else y[-1])
            else:
                targets.append(data[i+seq_len, 0])

        X_seq = np.array(sequences)
        y_seq = np.array(targets)

        # Build LSTM model
        self.model = keras.Sequential([
            keras.layers.LSTM(self.params['units'],
                             dropout=self.params['dropout'],
                             recurrent_dropout=self.params['recurrent_dropout'],
                             input_shape=(seq_len, data.shape[1])),
            keras.layers.Dense(1)
        ])

        self.model.compile(optimizer=self.params['optimizer'], loss=self.params['loss'])

        # Fit model
        self.model.fit(X_seq, y_seq,
                      epochs=self.params['epochs'],
                      batch_size=self.params['batch_size'],
                      validation_split=self.params['validation_split'],
                      verbose=self.params['verbose'])

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # For forecasting, use the last sequence to predict next values
        if isinstance(X, (int, np.integer)):
            # Forecast multiple steps ahead
            steps = int(X)
            predictions = []
            current_sequence = self.last_sequence.copy()

            for _ in range(steps):
                pred = self.model.predict(current_sequence.reshape(1, -1, current_sequence.shape[1]),
                                        verbose=0)
                predictions.append(pred[0, 0])
                # Update sequence (shift and add prediction)
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = pred[0]

            return np.array(predictions)
        else:
            # Direct prediction on provided sequences
            return self.model.predict(X, verbose=0).flatten()

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'LSTMModel':
        self.params.update(params)
        return self


class GRUModel(SupervisedModel):
    """GRU neural network for time series forecasting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

        self.task_type = 'forecasting'
        self.default_params = {
            'units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 10,
            'optimizer': 'adam',
            'loss': 'mse',
            'validation_split': 0.2,
            'verbose': 0
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        # Similar to LSTM but using GRU layers
        if isinstance(X, pd.DataFrame):
            data = X.values
        else:
            data = np.array(X)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Create sequences
        sequences = []
        targets = []
        seq_len = self.params['sequence_length']

        for i in range(len(data) - seq_len):
            sequences.append(data[i:i+seq_len])
            if isinstance(y, (pd.Series, np.ndarray)):
                targets.append(y[i+seq_len] if i+seq_len < len(y) else y[-1])
            else:
                targets.append(data[i+seq_len, 0])

        X_seq = np.array(sequences)
        y_seq = np.array(targets)

        # Build GRU model
        self.model = keras.Sequential([
            keras.layers.GRU(self.params['units'],
                            dropout=self.params['dropout'],
                            recurrent_dropout=self.params['recurrent_dropout'],
                            input_shape=(seq_len, data.shape[1])),
            keras.layers.Dense(1)
        ])

        self.model.compile(optimizer=self.params['optimizer'], loss=self.params['loss'])

        # Fit model
        self.model.fit(X_seq, y_seq,
                      epochs=self.params['epochs'],
                      batch_size=self.params['batch_size'],
                      validation_split=self.params['validation_split'],
                      verbose=self.params['verbose'])

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if isinstance(X, (int, np.integer)):
            # Forecasting logic similar to LSTM
            steps = int(X)
            predictions = []
            current_sequence = self.last_sequence.copy()

            for _ in range(steps):
                pred = self.model.predict(current_sequence.reshape(1, -1, current_sequence.shape[1]),
                                        verbose=0)
                predictions.append(pred[0, 0])
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = pred[0]

            return np.array(predictions)
        else:
            return self.model.predict(X, verbose=0).flatten()

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'GRUModel':
        self.params.update(params)
        return self
"""
AutoML Pipeline Framework

This module provides an automated machine learning pipeline that handles
model selection, training, hyperparameter tuning, and evaluation.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, regression_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from .models import ModelRegistry
from .evaluation import ModelEvaluator
from .tuning import HyperparameterTuner

logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """
    Automated Machine Learning Pipeline.

    Handles end-to-end ML workflow from data preprocessing to model deployment.
    """

    def __init__(self, task_type: str = 'auto', cv_folds: int = 5,
                 random_state: int = 42, max_models: int = 10,
                 tuning_method: str = 'random', tuning_iterations: int = 50):
        """
        Initialize the AutoML pipeline.

        Args:
            task_type: Type of ML task ('classification', 'regression', 'clustering', 'time_series', 'auto')
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            max_models: Maximum number of models to evaluate
            tuning_method: Hyperparameter tuning method ('grid', 'random', 'bayesian')
            tuning_iterations: Number of tuning iterations
        """
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_models = max_models
        self.tuning_method = tuning_method
        self.tuning_iterations = tuning_iterations

        # Initialize components
        self.model_registry = ModelRegistry()
        self.evaluator = ModelEvaluator(cv_folds=cv_folds, random_state=random_state)
        self.tuner = HyperparameterTuner(cv_folds=cv_folds, random_state=random_state)

        # Pipeline state
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.pipeline_results = {}

        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
            time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Fit the AutoML pipeline.

        Args:
            X: Feature matrix
            y: Target vector (None for unsupervised learning)
            time_column: Name of time column for time series tasks

        Returns:
            Dictionary containing pipeline results
        """
        try:
            # Determine task type if auto
            if self.task_type == 'auto':
                self.task_type = self._infer_task_type(X, y, time_column)

            logger.info(f"Starting AutoML pipeline for {self.task_type} task")

            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)

            # Select candidate models
            candidate_models = self._select_candidate_models()

            # Train and evaluate models
            self.pipeline_results = self._train_and_evaluate_models(
                candidate_models, X_processed, y_processed
            )

            # Select best model
            self._select_best_model()

            self.is_fitted = True

            logger.info(f"AutoML pipeline completed. Best model: {self.best_model}")

            return self.pipeline_results

        except Exception as e:
            logger.error(f"Error in AutoML pipeline: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_fitted or self.best_model is None:
            raise ValueError("Pipeline must be fitted before making predictions")

        # Preprocess input data
        X_processed, _ = self._preprocess_data(X, None, fit=False)

        # Make predictions
        predictions = self.best_model.predict(X_processed)

        # Inverse transform predictions if needed
        if self.label_encoder is not None and hasattr(self.best_model, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the best model.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions array
        """
        if not self.is_fitted or self.best_model is None:
            raise ValueError("Pipeline must be fitted before making predictions")

        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError("Best model does not support probability predictions")

        # Preprocess input data
        X_processed, _ = self._preprocess_data(X, None, fit=False)

        return self.best_model.predict_proba(X_processed)

    def forecast(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Make time series forecasts.

        Args:
            X: Historical data
            steps: Number of steps to forecast

        Returns:
            Forecast predictions
        """
        if self.task_type != 'time_series':
            raise ValueError("Forecasting is only available for time series tasks")

        if not self.is_fitted or self.best_model is None:
            raise ValueError("Pipeline must be fitted before forecasting")

        # For time series models, use their specific forecasting methods
        if hasattr(self.best_model, 'forecast'):
            return self.best_model.forecast(X, steps)
        else:
            # Use predict for non-time-series models adapted to forecasting
            return self.predict(X)

    def get_model_report(self) -> Dict[str, Any]:
        """
        Get comprehensive report of the AutoML pipeline results.

        Returns:
            Dictionary containing model performance metrics and details
        """
        if not self.is_fitted:
            return {}

        report = {
            'task_type': self.task_type,
            'best_model_name': self.best_model.__class__.__name__ if self.best_model else None,
            'best_score': self.best_score,
            'models_evaluated': len(self.models),
            'pipeline_results': self.pipeline_results,
            'model_details': {}
        }

        # Add details for each model
        for model_name, model_info in self.models.items():
            report['model_details'][model_name] = {
                'score': model_info.get('score', None),
                'params': model_info.get('params', {}),
                'training_time': model_info.get('training_time', None),
                'cv_scores': model_info.get('cv_scores', None)
            }

        return report

    def _infer_task_type(self, X: pd.DataFrame, y: Optional[pd.Series],
                        time_column: Optional[str]) -> str:
        """
        Infer the task type from data.

        Args:
            X: Feature matrix
            y: Target vector
            time_column: Time column name

        Returns:
            Inferred task type
        """
        if y is None:
            return 'clustering'
        elif time_column is not None:
            return 'time_series'
        elif y.dtype in ['object', 'category'] or len(y.unique()) < 20:
            return 'classification'
        else:
            return 'regression'

    def _preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series],
                        fit: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess the input data.

        Args:
            X: Feature matrix
            y: Target vector
            fit: Whether to fit transformers

        Returns:
            Tuple of processed X and y
        """
        X_processed = X.copy()

        # Handle categorical features
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)

        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean())

        # Scale features
        if fit:
            X_processed_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        else:
            X_processed_scaled = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )

        # Process target for classification
        y_processed = y
        if y is not None and self.task_type == 'classification':
            if fit:
                self.label_encoder = LabelEncoder()
                y_processed = pd.Series(
                    self.label_encoder.fit_transform(y),
                    index=y.index,
                    name=y.name
                )
            elif self.label_encoder is not None:
                y_processed = pd.Series(
                    self.label_encoder.transform(y),
                    index=y.index,
                    name=y.name
                )

        return X_processed_scaled, y_processed

    def _select_candidate_models(self) -> Dict[str, Any]:
        """
        Select candidate models based on task type.

        Returns:
            Dictionary of candidate models
        """
        if self.task_type == 'classification':
            model_types = ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier',
                          'SVC', 'LogisticRegression']
        elif self.task_type == 'regression':
            model_types = ['RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor',
                          'SVR', 'LinearRegression']
        elif self.task_type == 'clustering':
            model_types = ['KMeans', 'DBSCAN', 'GaussianMixture']
        elif self.task_type == 'time_series':
            model_types = ['ARIMA', 'Prophet', 'LSTM']
        else:
            model_types = ['RandomForestClassifier', 'RandomForestRegressor']

        # Get model instances
        candidate_models = {}
        for model_type in model_types[:self.max_models]:
            try:
                model = self.model_registry.get_model(model_type)
                candidate_models[model_type] = model
            except Exception as e:
                logger.warning(f"Could not load model {model_type}: {e}")

        return candidate_models

    def _train_and_evaluate_models(self, candidate_models: Dict[str, Any],
                                 X: pd.DataFrame, y: Optional[pd.Series]) -> Dict[str, Any]:
        """
        Train and evaluate candidate models.

        Args:
            candidate_models: Dictionary of candidate models
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary containing evaluation results
        """
        results = {}

        for model_name, model in candidate_models.items():
            try:
                logger.info(f"Training and evaluating {model_name}")

                # Tune hyperparameters
                tuning_results = self.tuner.tune_hyperparameters(
                    model, X, y, method=self.tuning_method,
                    n_iter=self.tuning_iterations, task_type=self.task_type
                )

                # Get tuned model
                tuned_model = tuning_results.get('best_estimator', model)

                # Evaluate model
                eval_results = self.evaluator.evaluate_model(
                    tuned_model, X, y, task_type=self.task_type
                )

                # Store results
                self.models[model_name] = {
                    'model': tuned_model,
                    'score': eval_results.get('score', 0),
                    'params': tuning_results.get('best_params', {}),
                    'tuning_results': tuning_results,
                    'eval_results': eval_results
                }

                results[model_name] = eval_results

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        return results

    def _select_best_model(self):
        """
        Select the best performing model.
        """
        best_score = -np.inf
        best_model_info = None

        for model_name, model_info in self.models.items():
            score = model_info.get('score', -np.inf)
            if score > best_score:
                best_score = score
                best_model_info = model_info

        if best_model_info:
            self.best_model = best_model_info['model']
            self.best_score = best_score
            logger.info(f"Selected {self.best_model.__class__.__name__} as best model with score {best_score}")

    def save_pipeline(self, filepath: str):
        """
        Save the trained pipeline to disk.

        Args:
            filepath: Path to save the pipeline
        """
        import joblib

        pipeline_state = {
            'task_type': self.task_type,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'models': self.models,
            'pipeline_results': self.pipeline_results,
            'is_fitted': self.is_fitted
        }

        joblib.dump(pipeline_state, filepath)
        logger.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """
        Load a trained pipeline from disk.

        Args:
            filepath: Path to load the pipeline from
        """
        import joblib

        pipeline_state = joblib.load(filepath)

        self.task_type = pipeline_state['task_type']
        self.best_model = pipeline_state['best_model']
        self.best_score = pipeline_state['best_score']
        self.scaler = pipeline_state['scaler']
        self.label_encoder = pipeline_state['label_encoder']
        self.models = pipeline_state['models']
        self.pipeline_results = pipeline_state['pipeline_results']
        self.is_fitted = pipeline_state['is_fitted']

        logger.info(f"Pipeline loaded from {filepath}")

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance from the best model.

        Returns:
            Series of feature importances
        """
        if not self.is_fitted or self.best_model is None:
            return None

        if hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(
                self.best_model.feature_importances_,
                index=self.scaler.feature_names_in_
            )
        elif hasattr(self.best_model, 'coef_'):
            return pd.Series(
                np.abs(self.best_model.coef_.flatten()),
                index=self.scaler.feature_names_in_
            )
        else:
            return None
"""
AutoML Pipeline Framework

This module provides an end-to-end automated machine learning pipeline that
handles data preprocessing, model selection, hyperparameter tuning, and evaluation.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_registry import ModelRegistry
from .hyperparameter_tuning import AutomatedTuner
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """
    End-to-end automated machine learning pipeline with intelligent model selection
    and hyperparameter optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AutoML pipeline.

        Args:
            config: Configuration dictionary for pipeline settings
        """
        self.config = config or self._get_default_config()
        self.model_registry = ModelRegistry()
        self.tuner = AutomatedTuner(self.config.get('tuning', {}))
        self.evaluator = ModelEvaluator(self.config.get('evaluation', {}))
        self.trained_models = {}
        self.best_model = None
        self.best_score = None
        self.preprocessing_pipeline = None

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'problem_type': 'auto',  # 'classification', 'regression', 'clustering', 'time_series'
            'max_models': 5,
            'time_budget': 600,  # 10 minutes
            'metric': 'auto',  # 'accuracy', 'f1', 'auc', 'mse', 'mae', 'r2', 'silhouette'
            'preprocessing': {
                'scale_numeric': True,
                'encode_categorical': True,
                'handle_missing': True
            },
            'tuning': {
                'enabled': True,
                'time_budget': 300
            },
            'evaluation': {
                'detailed_metrics': True,
                'feature_importance': True,
                'learning_curves': False
            }
        }

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
            problem_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Fit AutoML pipeline to data.

        Args:
            X: Feature matrix
            y: Target variable (None for unsupervised learning)
            problem_type: Override problem type detection

        Returns:
            Dictionary containing pipeline results
        """
        start_time = time.time()

        try:
            logger.info("Starting AutoML pipeline fitting")

            # Detect problem type if not specified
            if problem_type is None:
                problem_type = self._detect_problem_type(X, y)
            self.config['problem_type'] = problem_type

            logger.info(f"Detected problem type: {problem_type}")

            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)

            # Split data
            X_train, X_test, y_train, y_test = self._split_data(X_processed, y_processed)

            # Select candidate models
            candidate_models = self._select_candidate_models(problem_type)

            # Train and evaluate models
            model_results = self._train_and_evaluate_models(
                candidate_models, X_train, y_train, X_test, y_test
            )

            # Select best model
            self._select_best_model(model_results)

            # Final evaluation on test set
            final_results = self._final_evaluation(X_test, y_test)

            pipeline_time = time.time() - start_time

            results = {
                'problem_type': problem_type,
                'best_model': self.best_model,
                'best_score': self.best_score,
                'model_results': model_results,
                'final_results': final_results,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'pipeline_time': pipeline_time,
                'config': self.config
            }

            logger.info(f"AutoML pipeline completed in {pipeline_time:.2f} seconds")
            logger.info(f"Best model: {self.best_model}, Score: {self.best_score:.4f}")

            return results

        except Exception as e:
            logger.error(f"AutoML pipeline failed: {e}")
            return {'error': str(e)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best trained model.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if self.best_model is None:
            raise ValueError("Pipeline must be fitted before making predictions")

        # Preprocess input data
        X_processed = self._apply_preprocessing(X)

        # Make predictions
        model_instance = self.trained_models[self.best_model]
        return model_instance.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the best trained model.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions array
        """
        if self.best_model is None:
            raise ValueError("Pipeline must be fitted before making predictions")

        # Preprocess input data
        X_processed = self._apply_preprocessing(X)

        # Make probability predictions
        model_instance = self.trained_models[self.best_model]
        if hasattr(model_instance, 'predict_proba'):
            return model_instance.predict_proba(X_processed)
        else:
            raise AttributeError(f"Model {self.best_model} does not support probability predictions")

    def save_pipeline(self, filepath: str) -> None:
        """
        Save the trained pipeline to disk.

        Args:
            filepath: Path to save the pipeline
        """
        pipeline_data = {
            'config': self.config,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'trained_models': self.trained_models,
            'preprocessing_pipeline': self.preprocessing_pipeline
        }

        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str) -> None:
        """
        Load a trained pipeline from disk.

        Args:
            filepath: Path to load the pipeline from
        """
        pipeline_data = joblib.load(filepath)

        self.config = pipeline_data['config']
        self.best_model = pipeline_data['best_model']
        self.best_score = pipeline_data['best_score']
        self.trained_models = pipeline_data['trained_models']
        self.preprocessing_pipeline = pipeline_data['preprocessing_pipeline']

        logger.info(f"Pipeline loaded from {filepath}")

    def _detect_problem_type(self, X: pd.DataFrame, y: Optional[pd.Series]) -> str:
        """Automatically detect the type of ML problem."""
        if y is None:
            return 'clustering'

        # Check if target is numeric
        if y.dtype in ['int64', 'float64']:
            # Check if it's a regression problem (continuous) or classification (few unique values)
            n_unique = y.nunique()
            if n_unique <= 20:  # Arbitrary threshold
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    def _preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess the input data."""
        logger.info("Preprocessing data")

        X_processed = X.copy()
        y_processed = y.copy() if y is not None else None

        # Handle missing values
        if self.config['preprocessing']['handle_missing']:
            X_processed = self._handle_missing_values(X_processed)

        # Encode categorical variables
        if self.config['preprocessing']['encode_categorical']:
            X_processed, y_processed = self._encode_categorical(X_processed, y_processed)

        # Scale numeric features
        if self.config['preprocessing']['scale_numeric']:
            X_processed = self._scale_numeric_features(X_processed)

        # Store preprocessing pipeline for later use
        self.preprocessing_pipeline = {
            'missing_handled': self.config['preprocessing']['handle_missing'],
            'categorical_encoded': self.config['preprocessing']['encode_categorical'],
            'numeric_scaled': self.config['preprocessing']['scale_numeric']
        }

        return X_processed, y_processed

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # For now, use simple imputation
        # In a full implementation, this could be more sophisticated
        return X.fillna(X.mean(numeric_only=True))

    def _encode_categorical(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Encode categorical variables."""
        X_encoded = X.copy()

        # Encode categorical features
        for col in X_encoded.select_dtypes(include=['object', 'category']):
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

        # Encode target if it's categorical
        if y is not None and y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y_encoded = pd.Series(y_encoded, index=y.index, name=y.name)
            return X_encoded, y_encoded

        return X_encoded, y

    def _scale_numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features."""
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        return X_scaled

    def _split_data(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """Split data into train and test sets."""
        if y is not None:
            return train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y if self.config['problem_type'] == 'classification' else None
            )
        else:
            # For unsupervised learning
            n_samples = len(X)
            n_test = int(n_samples * self.config['test_size'])
            indices = np.random.RandomState(self.config['random_state']).permutation(n_samples)
            train_indices = indices[n_test:]
            test_indices = indices[:n_test]

            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]

            return X_train, X_test, None, None

    def _select_candidate_models(self, problem_type: str) -> List[str]:
        """Select candidate models based on problem type."""
        model_mapping = {
            'classification': ['RandomForest', 'XGBoost', 'LogisticRegression', 'SVM'],
            'regression': ['RandomForest', 'XGBoost', 'LinearRegression', 'SVR'],
            'clustering': ['KMeans', 'DBSCAN', 'GaussianMixture'],
            'time_series': ['ARIMA', 'Prophet', 'LSTM']
        }

        candidates = model_mapping.get(problem_type, ['RandomForest'])
        return candidates[:self.config['max_models']]

    def _train_and_evaluate_models(self, candidate_models: List[str],
                                 X_train: pd.DataFrame, y_train: Optional[pd.Series],
                                 X_test: pd.DataFrame, y_test: Optional[pd.Series]) -> Dict[str, Any]:
        """Train and evaluate candidate models."""
        logger.info(f"Training and evaluating {len(candidate_models)} models")

        model_results = {}

        for model_name in candidate_models:
            try:
                logger.info(f"Training {model_name}")

                # Get model class
                model_class = self.model_registry.get_model(model_name)
                if model_class is None:
                    logger.warning(f"Model {model_name} not found in registry")
                    continue

                # Train model with hyperparameter tuning
                if self.config['tuning']['enabled']:
                    tuning_results = self.tuner.auto_tune(
                        model_class, X_train, y_train, model_name,
                        self.config['problem_type'],
                        self.config['tuning']['time_budget']
                    )

                    if 'error' in tuning_results:
                        logger.warning(f"Tuning failed for {model_name}: {tuning_results['error']}")
                        # Fall back to default parameters
                        model_instance = model_class()
                    else:
                        model_instance = model_class(**tuning_results['best_params'])
                        logger.info(f"Best params for {model_name}: {tuning_results['best_params']}")
                else:
                    model_instance = model_class()

                # Train the model
                if y_train is not None:
                    model_instance.fit(X_train, y_train)
                else:
                    model_instance.fit(X_train)

                # Store trained model
                self.trained_models[model_name] = model_instance

                # Evaluate model
                evaluation_results = self.evaluator.evaluate_model(
                    model_instance, X_test, y_test, self.config['problem_type']
                )

                model_results[model_name] = {
                    'model': model_instance,
                    'evaluation': evaluation_results,
                    'tuning': tuning_results if self.config['tuning']['enabled'] else None
                }

                logger.info(f"{model_name} - Score: {evaluation_results.get('score', 'N/A')}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                model_results[model_name] = {'error': str(e)}

        return model_results

    def _select_best_model(self, model_results: Dict[str, Any]) -> None:
        """Select the best performing model."""
        best_score = -np.inf
        best_model_name = None

        for model_name, results in model_results.items():
            if 'error' in results:
                continue

            score = results['evaluation'].get('score')
            if score is not None and score > best_score:
                best_score = score
                best_model_name = model_name

        self.best_model = best_model_name
        self.best_score = best_score

        if best_model_name:
            logger.info(f"Selected best model: {best_model_name} with score {best_score:.4f}")
        else:
            logger.warning("No suitable model found")

    def _final_evaluation(self, X_test: pd.DataFrame, y_test: Optional[pd.Series]) -> Dict[str, Any]:
        """Perform final evaluation on test set."""
        if self.best_model is None:
            return {'error': 'No best model selected'}

        model_instance = self.trained_models[self.best_model]

        return self.evaluator.evaluate_model(
            model_instance, X_test, y_test, self.config['problem_type']
        )

    def _apply_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to new data."""
        X_processed = X.copy()

        # Apply the same preprocessing steps
        if self.preprocessing_pipeline['missing_handled']:
            X_processed = self._handle_missing_values(X_processed)

        if self.preprocessing_pipeline['categorical_encoded']:
            # For prediction, we need to handle categorical encoding consistently
            # This is a simplified version - in practice, you'd store the encoders
            for col in X_processed.select_dtypes(include=['object', 'category']):
                # Try to map to existing categories, otherwise use most frequent
                if hasattr(self, 'categorical_encoders') and col in self.categorical_encoders:
                    encoder = self.categorical_encoders[col]
                    # Handle unknown categories
                    X_processed[col] = X_processed[col].map(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_
                        else encoder.transform([encoder.classes_[0]])[0]
                    )
                else:
                    # Simple label encoding for unknown data
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        if self.preprocessing_pipeline['numeric_scaled']:
            # Apply the same scaling - in practice, you'd store the scaler
            if hasattr(self, 'scaler'):
                numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                X_processed[numeric_cols] = self.scaler.transform(X_processed[numeric_cols])

        return X_processed

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the best model.

        Returns:
            Dictionary of feature names and importance scores
        """
        if self.best_model is None:
            return None

        model_instance = self.trained_models[self.best_model]

        if hasattr(model_instance, 'feature_importances_'):
            # For tree-based models
            return dict(zip(self.feature_names, model_instance.feature_importances_))
        elif hasattr(model_instance, 'coef_'):
            # For linear models
            return dict(zip(self.feature_names, np.abs(model_instance.coef_)))
        else:
            return None

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the AutoML pipeline results.

        Returns:
            Dictionary containing pipeline summary
        """
        return {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'problem_type': self.config['problem_type'],
            'trained_models': list(self.trained_models.keys()),
            'preprocessing': self.preprocessing_pipeline,
            'config': self.config
        }


class AutoMLConfig:
    """
    Configuration class for AutoML pipeline settings.
    """

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default AutoML configuration."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'problem_type': 'auto',
            'max_models': 5,
            'time_budget': 600,
            'metric': 'auto',
            'preprocessing': {
                'scale_numeric': True,
                'encode_categorical': True,
                'handle_missing': True
            },
            'tuning': {
                'enabled': True,
                'time_budget': 300
            },
            'evaluation': {
                'detailed_metrics': True,
                'feature_importance': True,
                'learning_curves': False
            }
        }

    @staticmethod
    def get_config_for_problem_type(problem_type: str) -> Dict[str, Any]:
        """Get configuration optimized for specific problem type."""
        base_config = AutoMLConfig.get_default_config()
        base_config['problem_type'] = problem_type

        if problem_type == 'classification':
            base_config['metric'] = 'accuracy'
            base_config['max_models'] = 6
        elif problem_type == 'regression':
            base_config['metric'] = 'r2'
            base_config['max_models'] = 5
        elif problem_type == 'clustering':
            base_config['metric'] = 'silhouette'
            base_config['max_models'] = 4
        elif problem_type == 'time_series':
            base_config['metric'] = 'mae'
            base_config['max_models'] = 4

        return base_config
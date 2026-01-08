"""
AutoML Evaluation Framework

This module provides comprehensive evaluation capabilities for AutoML models,
including cross-validation, performance metrics, and model comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_percentage_error
)
from scipy import stats
import warnings
from dataclasses import dataclass
from enum import Enum

from .base import BaseModel


class MetricType(Enum):
    """Enumeration of metric types for different ML tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    metrics: Dict[str, float]
    cv_scores: Optional[Dict[str, List[float]]] = None
    predictions: Optional[np.ndarray] = None
    true_values: Optional[np.ndarray] = None
    fold_results: Optional[List[Dict[str, Any]]] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None


class CrossValidationStrategy(ABC):
    """Abstract base class for cross-validation strategies."""

    @abstractmethod
    def get_splits(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return list of (train_idx, test_idx) tuples."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass


class KFoldCV(CrossValidationStrategy):
    """Standard k-fold cross-validation."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_splits(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(kf.split(X, y))

    def get_name(self) -> str:
        return f"kfold_{self.n_splits}"


class StratifiedKFoldCV(CrossValidationStrategy):
    """Stratified k-fold cross-validation for classification."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_splits(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("StratifiedKFold requires target variable y")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return list(skf.split(X, y))

    def get_name(self) -> str:
        return f"stratified_kfold_{self.n_splits}"


class TimeSeriesCV(CrossValidationStrategy):
    """Time series cross-validation."""

    def __init__(self, n_splits: int = 5, gap: int = 0, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

    def get_splits(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap, test_size=self.test_size)
        return list(tscv.split(X))

    def get_name(self) -> str:
        return f"time_series_{self.n_splits}"


class MetricsCalculator:
    """Calculator for various performance metrics."""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                      y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC-AUC for binary classification
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                warnings.warn(f"Could not calculate ROC-AUC: {e}")

        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}

        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE (avoid division by zero)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            metrics['mape'] = np.nan

        return metrics

    @staticmethod
    def calculate_time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate time series specific metrics."""
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)

        # Add time series specific metrics
        # Mean Absolute Scaled Error (MASE)
        try:
            naive_forecast = np.roll(y_true, 1)
            naive_forecast[0] = y_true[0]  # First value can't be predicted naively
            naive_mae = mean_absolute_error(y_true[1:], naive_forecast[1:])
            if naive_mae > 0:
                metrics['mase'] = metrics['mae'] / naive_mae
            else:
                metrics['mase'] = np.nan
        except:
            metrics['mase'] = np.nan

        return metrics

    @staticmethod
    def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray,
                                   true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate clustering metrics."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        metrics = {}

        try:
            metrics['silhouette'] = silhouette_score(X, labels)
        except:
            metrics['silhouette'] = np.nan

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz'] = np.nan

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin'] = np.nan

        # Adjusted Rand Index and Adjusted Mutual Information if true labels available
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
            try:
                metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, labels)
                metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(true_labels, labels)
            except:
                metrics['adjusted_rand_index'] = np.nan
                metrics['adjusted_mutual_info'] = np.nan

        return metrics


class ModelEvaluator:
    """Main evaluator class for AutoML models."""

    def __init__(self, cv_strategy: Optional[CrossValidationStrategy] = None):
        self.cv_strategy = cv_strategy or KFoldCV()
        self.metrics_calculator = MetricsCalculator()

    def evaluate_model(self, model: BaseModel, X: pd.DataFrame, y: pd.Series,
                      cv: bool = True, n_splits: Optional[int] = None) -> EvaluationResult:
        """Evaluate a single model."""
        import time

        model_name = model.__class__.__name__
        start_time = time.time()

        # Fit model
        model.fit(X, y)
        training_time = time.time() - start_time

        # Get predictions
        start_time = time.time()
        predictions = model.predict(X)
        inference_time = time.time() - start_time

        # Calculate metrics based on model type
        if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
            # Classification model
            try:
                probas = model.predict_proba(X)
                metrics = self.metrics_calculator.calculate_classification_metrics(y.values, predictions, probas)
            except:
                metrics = self.metrics_calculator.calculate_classification_metrics(y.values, predictions)
        elif hasattr(model, 'task_type') and model.task_type == 'regression':
            # Regression model
            metrics = self.metrics_calculator.calculate_regression_metrics(y.values, predictions)
        elif hasattr(model, 'task_type') and model.task_type == 'time_series':
            # Time series model
            metrics = self.metrics_calculator.calculate_time_series_metrics(y.values, predictions)
        elif hasattr(model, 'task_type') and model.task_type == 'clustering':
            # Clustering model
            metrics = self.metrics_calculator.calculate_clustering_metrics(X.values, predictions, y.values if y is not None else None)
        else:
            # Default to regression metrics
            metrics = self.metrics_calculator.calculate_regression_metrics(y.values, predictions)

        # Cross-validation if requested
        cv_scores = None
        fold_results = None
        if cv:
            cv_scores, fold_results = self._perform_cross_validation(model, X, y, n_splits)

        return EvaluationResult(
            model_name=model_name,
            metrics=metrics,
            cv_scores=cv_scores,
            predictions=predictions,
            true_values=y.values,
            fold_results=fold_results,
            training_time=training_time,
            inference_time=inference_time
        )

    def _perform_cross_validation(self, model: BaseModel, X: pd.DataFrame, y: pd.Series,
                                n_splits: Optional[int] = None) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
        """Perform cross-validation and return scores."""
        if n_splits:
            if isinstance(self.cv_strategy, KFoldCV):
                self.cv_strategy.n_splits = n_splits
            elif isinstance(self.cv_strategy, StratifiedKFoldCV):
                self.cv_strategy.n_splits = n_splits
            elif isinstance(self.cv_strategy, TimeSeriesCV):
                self.cv_strategy.n_splits = n_splits

        splits = self.cv_strategy.get_splits(X, y)
        cv_scores = {}
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model on training fold
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)

            # Predict on test fold
            y_pred = model_copy.predict(X_test)

            # Calculate metrics for this fold
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model_copy.predict_proba(X_test)
                    fold_metrics = self.metrics_calculator.calculate_classification_metrics(y_test.values, y_pred, y_prob)
                except:
                    fold_metrics = self.metrics_calculator.calculate_classification_metrics(y_test.values, y_pred)
            elif hasattr(model, 'task_type') and model.task_type == 'regression':
                fold_metrics = self.metrics_calculator.calculate_regression_metrics(y_test.values, y_pred)
            elif hasattr(model, 'task_type') and model.task_type == 'time_series':
                fold_metrics = self.metrics_calculator.calculate_time_series_metrics(y_test.values, y_pred)
            else:
                fold_metrics = self.metrics_calculator.calculate_regression_metrics(y_test.values, y_pred)

            fold_results.append({
                'fold': fold_idx + 1,
                'metrics': fold_metrics,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })

            # Accumulate scores
            for metric_name, value in fold_metrics.items():
                if metric_name not in cv_scores:
                    cv_scores[metric_name] = []
                cv_scores[metric_name].append(value)

        return cv_scores, fold_results

    def compare_models(self, results: List[EvaluationResult],
                      metric: str = 'accuracy') -> pd.DataFrame:
        """Compare multiple models based on a specific metric."""
        comparison_data = []

        for result in results:
            row = {'model': result.model_name}

            # Add main metric
            if metric in result.metrics:
                row[metric] = result.metrics[metric]
            else:
                row[metric] = np.nan

            # Add CV statistics if available
            if result.cv_scores and metric in result.cv_scores:
                scores = result.cv_scores[metric]
                row[f'{metric}_mean'] = np.mean(scores)
                row[f'{metric}_std'] = np.std(scores)
                row[f'{metric}_min'] = np.min(scores)
                row[f'{metric}_max'] = np.max(scores)
            else:
                row[f'{metric}_mean'] = row[metric] if not np.isnan(row[metric]) else np.nan
                row[f'{metric}_std'] = 0.0
                row[f'{metric}_min'] = row[metric] if not np.isnan(row[metric]) else np.nan
                row[f'{metric}_max'] = row[metric] if not np.isnan(row[metric]) else np.nan

            # Add training and inference times
            row['training_time'] = result.training_time or np.nan
            row['inference_time'] = result.inference_time or np.nan

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def statistical_significance_test(self, result1: EvaluationResult, result2: EvaluationResult,
                                    metric: str = 'accuracy', alpha: float = 0.05) -> Dict[str, Any]:
        """Perform statistical significance test between two model results."""
        if not (result1.cv_scores and result2.cv_scores):
            return {'error': 'Cross-validation scores required for statistical testing'}

        if metric not in result1.cv_scores or metric not in result2.cv_scores:
            return {'error': f'Metric {metric} not found in both results'}

        scores1 = result1.cv_scores[metric]
        scores2 = result2.cv_scores[metric]

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(scores1) - np.mean(scores2)
        pooled_std = np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            'metric': metric,
            'model1': result1.model_name,
            'model2': result2.model_name,
            'model1_mean': np.mean(scores1),
            'model2_mean': np.mean(scores2),
            'mean_difference': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': effect_size,
            'alpha': alpha
        }


class AutoMLEvaluator:
    """High-level evaluator for AutoML pipelines."""

    def __init__(self, cv_strategy: Optional[CrossValidationStrategy] = None):
        self.evaluator = ModelEvaluator(cv_strategy)

    def evaluate_pipeline_results(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate results from an AutoML pipeline."""
        evaluation_results = {}

        if 'best_model' in pipeline_results:
            best_model = pipeline_results['best_model']
            X = pipeline_results.get('X_test')
            y = pipeline_results.get('y_test')

            if X is not None and y is not None:
                evaluation_results['best_model_eval'] = self.evaluator.evaluate_model(
                    best_model, X, y, cv=True
                )

        if 'model_candidates' in pipeline_results:
            candidate_results = []
            for model_info in pipeline_results['model_candidates']:
                model = model_info.get('model')
                X = pipeline_results.get('X_test')
                y = pipeline_results.get('y_test')

                if model and X is not None and y is not None:
                    result = self.evaluator.evaluate_model(model, X, y, cv=False)
                    candidate_results.append(result)

            evaluation_results['candidate_results'] = candidate_results

            if candidate_results:
                evaluation_results['model_comparison'] = self.evaluator.compare_models(
                    candidate_results, metric=pipeline_results.get('metric', 'accuracy')
                )

        return evaluation_results
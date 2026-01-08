"""
Tests for AutoML functionality.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from greta_core.automl import AutoMLPipeline, ModelRegistry
from greta_core.automl.models import RandomForestModel, XGBoostModel
from greta_core.automl.evaluation import AutoMLEvaluator
from greta_core.automl.tuning import HyperparameterTuner


class TestModelRegistry:
    """Test the ModelRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes with built-in models."""
        registry = ModelRegistry()
        models = registry.get_all_models()
        assert len(models) > 0

    def test_get_models_by_task(self):
        """Test filtering models by task type."""
        registry = ModelRegistry()
        classification_models = registry.get_models_by_task('classification')
        assert len(classification_models) > 0
        assert all('classification' in model['metadata']['task_types'] for model in classification_models)

    def test_create_model(self):
        """Test creating a model instance."""
        registry = ModelRegistry()
        model = registry.create_model('RandomForestModel')
        assert isinstance(model, RandomForestModel)


class TestAutoMLPipeline:
    """Test the AutoML pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return {'X': X_df, 'y': y}

    def test_pipeline_initialization(self, sample_data):
        """Test pipeline initialization."""
        config = {
            'enabled': True,
            'task_type': 'classification',
            'models': ['RandomForestModel'],
            'max_models': 3
        }
        pipeline = AutoMLPipeline(config, sample_data)
        assert pipeline.task_type == 'classification'
        assert len(pipeline.models_to_try) > 0

    def test_pipeline_run(self, sample_data):
        """Test running the complete pipeline."""
        config = {
            'enabled': True,
            'task_type': 'classification',
            'models': ['RandomForestModel'],
            'tuning': {'enabled': False},  # Disable tuning for faster tests
            'ensemble': {'enabled': False}
        }
        pipeline = AutoMLPipeline(config, sample_data)
        results = pipeline.run_automl()

        assert 'task_type' in results
        assert 'models_evaluated' in results
        assert results['models_evaluated'] > 0

    def test_pipeline_with_hypotheses(self, sample_data):
        """Test pipeline with hypothesis-based feature selection."""
        hypotheses = [
            {'features': ['feature_0', 'feature_1'], 'fitness': 0.1},
            {'features': ['feature_2', 'feature_3'], 'fitness': 0.2}
        ]

        config = {
            'enabled': True,
            'task_type': 'classification',
            'models': ['RandomForestModel'],
            'tuning': {'enabled': False},
            'ensemble': {'enabled': False}
        }
        pipeline = AutoMLPipeline(config, sample_data, hypotheses)
        selected_features = pipeline._select_features_from_hypotheses()

        assert selected_features == ['feature_0', 'feature_1']


class TestAutoMLEvaluator:
    """Test the AutoML evaluator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return X, y

    def test_evaluate_classification(self, sample_data):
        """Test evaluating a classification model."""
        X, y = sample_data
        model = RandomForestModel()
        model.fit(X, y)

        evaluator = AutoMLEvaluator()
        scores = evaluator.evaluate_model(model, X, y, 'classification')

        assert 'cv_accuracy' in scores
        assert isinstance(scores['cv_accuracy'], (int, float))

    def test_evaluate_regression(self):
        """Test evaluating a regression model."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model = RandomForestModel()
        model.fit(X, y)

        evaluator = AutoMLEvaluator()
        scores = evaluator.evaluate_model(model, X, y, 'regression')

        assert 'cv_neg_mean_squared_error' in scores


class TestHyperparameterTuner:
    """Test the hyperparameter tuner."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        return X, y

    def test_tuner_initialization(self):
        """Test tuner initialization."""
        tuner = HyperparameterTuner()
        assert hasattr(tuner, 'tuning_strategies')

    def test_random_tuning(self, sample_data):
        """Test random search tuning."""
        X, y = sample_data
        model = RandomForestModel()

        tuner = HyperparameterTuner()
        tuned_model = tuner.tune_model(model, X, y, 'classification',
                                     {'method': 'random', 'max_evals': 2})

        # Check that the model was updated with a fitted estimator
        assert hasattr(tuned_model, 'model')
        assert tuned_model.model is not None


class TestModels:
    """Test individual ML models."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Create regression data."""
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        return X, y

    def test_random_forest_classification(self, classification_data):
        """Test RandomForest for classification."""
        X, y = classification_data
        model = RandomForestModel()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

        # Test feature importance
        importance = model.get_feature_importance()
        assert importance is not None

    def test_random_forest_regression(self, regression_data):
        """Test RandomForest for regression."""
        X, y = regression_data
        model = RandomForestModel()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_model_parameters(self):
        """Test model parameter getting/setting."""
        model = RandomForestModel({'n_estimators': 50})
        params = model.get_params()
        assert params['n_estimators'] == 50

        model.set_params(n_estimators=100)
        assert model.get_params()['n_estimators'] == 100


class TestIntegration:
    """Test integration with existing system."""

    def test_backward_compatibility(self):
        """Test that existing functionality works when AutoML is disabled."""
        # This would require testing the full pipeline, but for now just check imports
        from greta_core.automl import AutoMLPipeline
        assert AutoMLPipeline is not None

    def test_config_integration(self):
        """Test that AutoML config integrates with main config."""
        from greta_cli.config import GretaConfig

        config = GretaConfig(data={'type': 'csv', 'source': 'test.csv'})
        assert hasattr(config, 'automl')
        assert config.automl.enabled == False  # Default should be disabled
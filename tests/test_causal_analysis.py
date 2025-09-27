"""
Tests for causal analysis module using pytest.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Try to import causal analysis functions
try:
    from greta_core.causal_analysis import (
        define_causal_model, identify_confounders, estimate_causal_effect,
        perform_causal_analysis, validate_causal_assumptions
    )
    from greta_core.causal_narratives import generate_causal_narrative, generate_causal_insight
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False


@pytest.mark.skipif(not DOWHY_AVAILABLE, reason="DoWhy not available")
class TestDefineCausalModel:
    """Test causal model definition functionality."""

    def test_define_causal_model_basic(self):
        """Test basic causal model definition."""
        data = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'outcome': [1, 2, 1, 3],
            'confounder': [0.5, 1.0, 0.3, 0.8]
        })

        model = define_causal_model(
            data=data,
            treatment='treatment',
            outcome='outcome',
            confounders=['confounder']
        )

        assert model is not None
        # Check that the model has the expected attributes
        assert hasattr(model, '_treatment')
        assert hasattr(model, '_outcome')

    def test_define_causal_model_no_confounders(self):
        """Test causal model without confounders."""
        data = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'outcome': [1, 2, 1, 3]
        })

        model = define_causal_model(
            data=data,
            treatment='treatment',
            outcome='outcome'
        )

        assert model is not None


@pytest.mark.skipif(not DOWHY_AVAILABLE, reason="DoWhy not available")
class TestIdentifyConfounders:
    """Test confounder identification functionality."""

    @patch('greta_core.causal_analysis.CausalModel')
    def test_identify_confounders(self, mock_model_class):
        """Test confounder identification."""
        # Mock the CausalModel and its methods
        mock_model = MagicMock()
        mock_estimand = MagicMock()
        mock_estimand.get_backdoor_variables.return_value = ['confounder1']
        mock_estimand.get_instrumental_variables.return_value = []
        mock_estimand.get_frontdoor_variables.return_value = []
        mock_model.identify_effect.return_value = mock_estimand
        mock_model_class.return_value = mock_model

        data = pd.DataFrame({
            'treatment': [0, 1],
            'outcome': [1, 2],
            'confounder1': [0.5, 1.0]
        })

        model = define_causal_model(data, 'treatment', 'outcome', ['confounder1'])
        results = identify_confounders(model)

        assert 'estimand' in results
        assert 'backdoor_variables' in results
        assert results['backdoor_variables'] == ['confounder1']


@pytest.mark.skipif(not DOWHY_AVAILABLE, reason="DoWhy not available")
class TestEstimateCausalEffect:
    """Test causal effect estimation functionality."""

    @patch('greta_core.causal_analysis.CausalModel')
    def test_estimate_causal_effect(self, mock_model_class):
        """Test causal effect estimation."""
        # Mock the model and estimate
        mock_model = MagicMock()
        mock_estimand = MagicMock()
        mock_estimate = MagicMock()
        mock_estimate.value = 0.5
        mock_estimate.get_confidence_intervals.return_value = (0.2, 0.8)

        mock_model.identify_effect.return_value = mock_estimand
        mock_model.estimate_effect.return_value = mock_estimate
        mock_model_class.return_value = mock_model

        data = pd.DataFrame({
            'treatment': [0, 1],
            'outcome': [1, 2]
        })

        model = define_causal_model(data, 'treatment', 'outcome')
        results = estimate_causal_effect(model)

        assert 'estimate' in results
        assert results['estimate'] == 0.5
        assert 'confidence_intervals' in results


@pytest.mark.skipif(not DOWHY_AVAILABLE, reason="DoWhy not available")
class TestPerformCausalAnalysis:
    """Test complete causal analysis pipeline."""

    @patch('greta_core.causal_analysis.CausalModel')
    def test_perform_causal_analysis(self, mock_model_class):
        """Test full causal analysis."""
        # Mock all components
        mock_model = MagicMock()
        mock_estimand = MagicMock()
        mock_estimand.get_backdoor_variables.return_value = []
        mock_estimate = MagicMock()
        mock_estimate.value = 0.3
        mock_estimate.get_confidence_intervals.return_value = (0.1, 0.5)

        mock_model.identify_effect.return_value = mock_estimand
        mock_model.estimate_effect.return_value = mock_estimate
        mock_model_class.return_value = mock_model

        data = pd.DataFrame({
            'treatment': [0, 1, 0, 1],
            'outcome': [1, 2, 1, 3],
            'confounder': [0.5, 1.0, 0.3, 0.8]
        })

        results = perform_causal_analysis(
            data=data,
            treatment='treatment',
            outcome='outcome',
            confounders=['confounder']
        )

        assert 'model' in results
        assert 'identification' in results
        assert 'estimation' in results
        assert results['treatment'] == 'treatment'
        assert results['outcome'] == 'outcome'


@pytest.mark.skipif(not DOWHY_AVAILABLE, reason="DoWhy not available")
class TestCausalNarratives:
    """Test causal narrative generation."""

    def test_generate_causal_narrative(self):
        """Test causal narrative generation."""
        causal_results = {
            'estimation': {
                'estimate': 0.4,
                'method': 'backdoor.linear_regression',
                'confidence_intervals': (0.1, 0.7)
            },
            'treatment': 'feature_A',
            'outcome': 'target',
            'confounders': ['feature_B']
        }

        narrative = generate_causal_narrative(causal_results)

        assert isinstance(narrative, str)
        assert 'feature_A' in narrative
        assert 'target' in narrative
        assert 'strong causal effect' in narrative

    def test_generate_causal_narrative_no_results(self):
        """Test narrative generation with no results."""
        narrative = generate_causal_narrative(None)
        assert "No causal analysis was performed" in narrative

    def test_generate_causal_insight(self):
        """Test causal insight generation."""
        causal_results = {
            'identification': {
                'backdoor_variables': ['confounder1'],
                'instrumental_variables': [],
                'frontdoor_variables': []
            }
        }

        insight = generate_causal_insight(causal_results)

        assert isinstance(insight, str)
        assert 'confounding' in insight.lower()


@pytest.mark.skipif(not DOWHY_AVAILABLE, reason="DoWhy not available")
class TestCausalAnalysisIntegration:
    """Integration tests for causal analysis."""

    @patch('greta_core.causal_analysis.CausalModel')
    def test_causal_analysis_pipeline_integration(self, mock_model_class):
        """Test integration of causal analysis with mocked DoWhy."""
        # Mock the entire DoWhy pipeline
        mock_model = MagicMock()
        mock_estimand = MagicMock()
        mock_estimand.get_backdoor_variables.return_value = ['age']
        mock_estimate = MagicMock()
        mock_estimate.value = 0.25
        mock_estimate.get_confidence_intervals.return_value = (0.05, 0.45)

        mock_model.identify_effect.return_value = mock_estimand
        mock_model.estimate_effect.return_value = mock_estimate
        mock_model_class.return_value = mock_model

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'treatment': np.random.binomial(1, 0.5, 100),
            'outcome': np.random.randn(100),
            'age': np.random.randn(100),
            'income': np.random.randn(100)
        })

        # Run causal analysis
        results = perform_causal_analysis(
            data=data,
            treatment='treatment',
            outcome='outcome',
            confounders=['age', 'income']
        )

        # Verify structure
        assert isinstance(results, dict)
        assert 'estimation' in results
        assert 'identification' in results

        # Test narrative generation
        narrative = generate_causal_narrative(results)
        assert isinstance(narrative, str)
        assert len(narrative) > 0
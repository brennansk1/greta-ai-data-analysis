"""
Tests for enhanced feature engineering capabilities.
"""

import pytest
import pandas as pd
import numpy as np
from greta_core.preprocessing import (
    prepare_features_for_modeling, basic_feature_engineering,
    encode_one_hot, encode_label, encode_ordinal, encode_frequency, encode_target_mean,
    generate_polynomial_features, generate_trigonometric_features,
    select_features, generate_features_automated, analyze_feature_importance
)


class TestEnhancedFeatureEngineering:
    """Test enhanced feature engineering functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100) * 2,
            'categorical': ['A', 'B', 'C'] * 33 + ['A'],
            'ordinal': ['low', 'medium', 'high'] * 33 + ['low'],
            'target': np.random.randint(0, 2, 100)
        })

    def test_prepare_features_for_modeling_basic(self):
        """Test basic feature preparation."""
        df_prepared, metadata = prepare_features_for_modeling(self.df, self.df['target'])

        assert df_prepared is not None
        assert 'encoding_info' in metadata
        assert 'engineering_info' in metadata
        assert metadata['success'] is True

    def test_prepare_features_for_modeling_with_encoding(self):
        """Test feature preparation with specific encoding."""
        df_prepared, metadata = prepare_features_for_modeling(
            self.df, self.df['target'],
            encoding_method='one_hot',
            numeric_transforms=['polynomial', 'trigonometric']
        )

        assert df_prepared.shape[1] > self.df.shape[1]  # Should have more features
        assert 'one_hot' in str(metadata['encoding_info'])

    def test_backward_compatibility_basic_feature_engineering(self):
        """Test backward compatibility of basic_feature_engineering."""
        # Old interface should return just DataFrame
        result = basic_feature_engineering(self.df)
        assert isinstance(result, pd.DataFrame)

        # New interface should return tuple
        result_tuple = basic_feature_engineering(self.df, numeric_transforms=['polynomial'])
        assert isinstance(result_tuple, tuple)
        assert len(result_tuple) == 2

    def test_encoding_functions(self):
        """Test individual encoding functions."""
        # One-hot encoding
        df_encoded, info = encode_one_hot(self.df, ['categorical'])
        assert 'categorical_A' in df_encoded.columns
        assert 'categorical_B' in df_encoded.columns
        assert 'categorical' not in df_encoded.columns

        # Label encoding
        df_encoded, info = encode_label(self.df, ['categorical'])
        assert 'categorical_encoded' in df_encoded.columns
        assert 'categorical' not in df_encoded.columns

        # Frequency encoding
        df_encoded, info = encode_frequency(self.df, ['categorical'])
        assert 'categorical_freq_encoded' in df_encoded.columns

    def test_target_mean_encoding(self):
        """Test target mean encoding."""
        df_encoded, info = encode_target_mean(self.df, ['categorical'], self.df['target'])
        assert 'categorical_target_encoded' in df_encoded.columns
        assert info['categorical']['method'] == 'target_mean'

    def test_feature_generation_functions(self):
        """Test feature generation functions."""
        numeric_df = self.df[['numeric1', 'numeric2']]

        # Polynomial features
        df_poly, info = generate_polynomial_features(numeric_df, ['numeric1'])
        assert 'numeric1_pow_2' in df_poly.columns
        assert info['degree'] == 2

        # Trigonometric features
        df_trig, info = generate_trigonometric_features(numeric_df, ['numeric1'])
        assert 'numeric1_sin' in df_trig.columns
        assert 'numeric1_cos' in df_trig.columns

    def test_error_handling(self):
        """Test error handling in feature engineering."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            prepare_features_for_modeling(empty_df)

        # Invalid encoding method
        with pytest.raises(ValueError):
            prepare_features_for_modeling(self.df, encoding_method='invalid')

        # Invalid numeric transforms
        with pytest.raises(ValueError):
            prepare_features_for_modeling(self.df, numeric_transforms=['invalid_transform'])


class TestFeatureSelection:
    """Test feature selection functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'noise1': np.random.randn(100) * 0.1,
            'noise2': np.random.randn(100) * 0.1
        })
        self.y = self.X['feature1'] + self.X['feature2'] + np.random.randn(100) * 0.1

    def test_select_by_correlation(self):
        """Test correlation-based feature selection."""
        X_selected, features, info = select_features(self.X, self.y, method='correlation')

        assert len(features) > 0
        assert 'method' in info
        assert info['method'] == 'correlation'

    def test_select_by_mutual_info(self):
        """Test mutual information-based feature selection."""
        X_selected, features, info = select_features(self.X, self.y, method='mutual_info', k=3)

        assert len(features) <= 3
        assert info['method'] == 'mutual_info'

    def test_select_by_univariate(self):
        """Test univariate feature selection."""
        X_selected, features, info = select_features(self.X, self.y, method='univariate', k=3)

        assert len(features) <= 3
        assert 'feature_scores' in info

    def test_select_by_rfe(self):
        """Test RFE feature selection."""
        X_selected, features, info = select_features(self.X, self.y, method='rfe', n_features_to_select=3)

        assert len(features) <= 3
        assert info['method'] == 'rfe'


class TestAutomatedFeatureGeneration:
    """Test automated feature generation."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'numeric1': np.random.randn(50),
            'numeric2': np.random.randn(50),
            'categorical': ['A', 'B', 'C'] * 16 + ['A', 'B'],
            'target': np.random.randint(0, 2, 50)
        })

    def test_automated_generation(self):
        """Test automated feature generation."""
        df_generated, info = generate_features_automated(self.df, self.df['target'])

        assert df_generated.shape[1] >= self.df.shape[1]
        assert 'characteristics' in info
        assert 'total_features_generated' in info

    def test_automated_generation_no_target(self):
        """Test automated generation without target."""
        df_generated, info = generate_features_automated(self.df)

        assert df_generated.shape[1] >= self.df.shape[1]
        assert 'characteristics' in info


class TestImportanceAnalysis:
    """Test importance analysis functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'important1': np.random.randn(100),
            'important2': np.random.randn(100),
            'noise': np.random.randn(100) * 0.1
        })
        self.y = self.X['important1'] * 2 + self.X['important2'] + np.random.randn(100) * 0.1

    def test_analyze_feature_importance(self):
        """Test comprehensive importance analysis."""
        results = analyze_feature_importance(self.X, self.y, methods=['tree_importance', 'permutation'])

        assert 'tree_importance' in results
        assert 'permutation' in results
        assert 'consensus' in results

        # Check consensus ranking
        consensus = results['consensus']
        assert 'importance_ranking' in consensus

    def test_get_top_important_features(self):
        """Test extracting top important features."""
        results = analyze_feature_importance(self.X, self.y)

        top_features = get_top_important_features(results, method='consensus', top_k=2)
        assert len(top_features) <= 2
        assert isinstance(top_features, list)
"""
Comprehensive tests for preprocessing module using pytest.
"""

import pytest
import pandas as pd
import numpy as np
from greta_core.preprocessing import (
    profile_data, handle_missing_values, detect_outliers,
    normalize_data_types, remove_outliers, basic_feature_engineering
)


class TestProfileData:
    """Test data profiling functionality."""

    def test_profile_data_basic(self, messy_data):
        """Test basic data profiling."""
        profile = profile_data(messy_data)

        assert 'shape' in profile
        assert 'columns' in profile
        assert profile['shape'] == (5, 6)
        assert len(profile['columns']) == 6

    def test_profile_data_numeric_columns(self, messy_data):
        """Test profiling of numeric columns."""
        profile = profile_data(messy_data)

        # Check numeric column profiling
        numeric_col = profile['columns']['numeric_clean']
        assert 'mean' in numeric_col
        assert 'std' in numeric_col
        assert 'min' in numeric_col
        assert 'max' in numeric_col
        assert numeric_col['null_count'] == 0
        assert numeric_col['unique_count'] == 5

    def test_profile_data_categorical_columns(self, messy_data):
        """Test profiling of categorical columns."""
        profile = profile_data(messy_data)

        # Check categorical column profiling
        cat_col = profile['columns']['categorical_clean']
        assert 'most_common' in cat_col
        assert 'least_common' in cat_col
        assert cat_col['dtype'] == 'object'
        assert cat_col['unique_count'] == 2

    def test_profile_data_empty_dataframe(self):
        """Test profiling empty DataFrame."""
        empty_df = pd.DataFrame()
        profile = profile_data(empty_df)

        assert profile['shape'] == (0, 0)
        assert len(profile['columns']) == 0

    def test_profile_data_single_column(self):
        """Test profiling DataFrame with single column."""
        single_df = pd.DataFrame({'col': [1, 2, 3]})
        profile = profile_data(single_df)

        assert profile['shape'] == (3, 1)
        assert len(profile['columns']) == 1
        assert 'col' in profile['columns']


class TestHandleMissingValues:
    """Test missing value handling functionality."""

    def test_handle_missing_values_mean(self, messy_data):
        """Test mean imputation for missing values."""
        cleaned = handle_missing_values(messy_data, strategy='mean')

        # Should not drop rows
        assert cleaned.shape[0] == 5
        # Should fill nulls in numeric_with_nulls
        assert not cleaned['numeric_with_nulls'].isnull().any()
        # Should preserve non-null values
        assert cleaned.loc[0, 'numeric_with_nulls'] == 1.0

    def test_handle_missing_values_median(self, messy_data):
        """Test median imputation."""
        cleaned = handle_missing_values(messy_data, strategy='median')

        assert not cleaned['numeric_with_nulls'].isnull().any()
        # Median of [1.0, 2.0, 4.0, 5.0] should be 3.0
        assert cleaned.loc[2, 'numeric_with_nulls'] == 3.0

    def test_handle_missing_values_mode(self, messy_data):
        """Test mode imputation."""
        cleaned = handle_missing_values(messy_data, strategy='mode')

        assert not cleaned['categorical_with_nulls'].isnull().any()
        # Mode should be 'X' (appears twice)
        assert cleaned.loc[2, 'categorical_with_nulls'] == 'X'

    def test_handle_missing_values_drop(self, messy_data):
        """Test dropping rows with missing values."""
        cleaned = handle_missing_values(messy_data, strategy='drop')

        # Should drop row with null in categorical_with_nulls
        assert cleaned.shape[0] == 4
        assert not cleaned.isnull().any().any()

    def test_handle_missing_values_threshold(self, messy_data):
        """Test column dropping based on missing threshold."""
        # Create data with high missing rate
        high_missing_df = messy_data.copy()
        high_missing_df['mostly_null'] = [1, None, None, None, None]

        cleaned = handle_missing_values(high_missing_df, threshold=0.6)

        # Should drop the mostly_null column
        assert 'mostly_null' not in cleaned.columns
        assert cleaned.shape[1] == 6  # Original 6 + 1 - 1 dropped

    def test_handle_missing_values_no_nulls(self):
        """Test handling when no missing values exist."""
        clean_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        cleaned = handle_missing_values(clean_df)

        pd.testing.assert_frame_equal(cleaned, clean_df)


class TestDetectOutliers:
    """Test outlier detection functionality."""

    def test_detect_outliers_iqr(self, messy_data):
        """Test IQR-based outlier detection."""
        outliers = detect_outliers(messy_data, method='iqr')

        # Should detect outlier in numeric_with_outliers
        assert 'numeric_with_outliers' in outliers
        assert len(outliers['numeric_with_outliers']) > 0
        # The value 1000 should be detected as outlier
        assert 4 in outliers['numeric_with_outliers']  # Index 4 has 1000

    def test_detect_outliers_zscore(self, messy_data):
        """Test Z-score-based outlier detection."""
        outliers = detect_outliers(messy_data, method='zscore', threshold=2.0)

        assert isinstance(outliers, dict)
        # Should have entries for numeric columns
        assert 'numeric_with_outliers' in outliers

    def test_detect_outliers_no_numeric(self):
        """Test outlier detection on non-numeric data."""
        text_df = pd.DataFrame({
            'text_col': ['a', 'b', 'c', 'd', 'e']
        })
        outliers = detect_outliers(text_df)

        assert len(outliers) == 0  # No numeric columns

    def test_detect_outliers_empty_dataframe(self):
        """Test outlier detection on empty DataFrame."""
        empty_df = pd.DataFrame()
        outliers = detect_outliers(empty_df)

        assert outliers == {}

    def test_detect_outliers_single_value(self):
        """Test outlier detection with single value."""
        single_df = pd.DataFrame({'col': [5]})
        outliers = detect_outliers(single_df)

        assert 'col' in outliers
        assert len(outliers['col']) == 0  # No outliers with single value


class TestNormalizeDataTypes:
    """Test data type normalization functionality."""

    def test_normalize_data_types_categorical(self):
        """Test conversion of low-cardinality object columns to category."""
        df = pd.DataFrame({
            'high_card': [f'val_{i}' for i in range(100)],  # High cardinality
            'low_card': ['A', 'B', 'A', 'B'] * 25  # Low cardinality
        })

        normalized = normalize_data_types(df)

        # Low cardinality should become category
        assert pd.api.types.is_categorical_dtype(normalized['low_card'])
        # High cardinality should remain object
        assert normalized['high_card'].dtype == 'object'

    def test_normalize_data_types_numeric_strings(self):
        """Test conversion of numeric strings to numbers."""
        df = pd.DataFrame({
            'numeric_str': ['1', '2', '3'],
            'text_str': ['a', 'b', 'c'],
            'mixed': ['1', 'text', '3']
        })

        normalized = normalize_data_types(df)

        # Pure numeric strings should become numeric
        assert pd.api.types.is_numeric_dtype(normalized['numeric_str'])
        # Mixed should remain object
        assert normalized['mixed'].dtype == 'object'

    def test_normalize_data_types_already_normalized(self, messy_data):
        """Test normalization on already well-typed data."""
        normalized = normalize_data_types(messy_data)

        # Should not change much for already good data
        assert normalized['numeric_clean'].dtype == messy_data['numeric_clean'].dtype


class TestRemoveOutliers:
    """Test outlier removal functionality."""

    def test_remove_outliers_basic(self, messy_data):
        """Test basic outlier removal."""
        outliers = {'numeric_with_outliers': [4]}  # Remove index 4

        cleaned = remove_outliers(messy_data, outliers)

        assert cleaned.shape[0] == 4  # One row removed
        assert 1000 not in cleaned['numeric_with_outliers'].values

    def test_remove_outliers_multiple_columns(self, messy_data):
        """Test removing outliers from multiple columns."""
        outliers = {
            'numeric_with_outliers': [4],
            'numeric_with_nulls': [2]
        }

        cleaned = remove_outliers(messy_data, outliers)

        # Should remove rows that have outliers in any column
        assert cleaned.shape[0] == 3  # Two rows removed

    def test_remove_outliers_no_outliers(self, messy_data):
        """Test removal when no outliers specified."""
        outliers = {}
        cleaned = remove_outliers(messy_data, outliers)

        pd.testing.assert_frame_equal(cleaned, messy_data)


class TestBasicFeatureEngineering:
    """Test basic feature engineering functionality."""

    def test_basic_feature_engineering_numeric(self):
        """Test feature engineering on numeric columns."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z']  # Non-numeric
        })

        engineered = basic_feature_engineering(df)

        # Should add squared terms
        assert 'A_squared' in engineered.columns
        assert 'B_squared' in engineered.columns
        assert engineered['A_squared'].tolist() == [1, 4, 9]

        # Should add interaction terms
        assert 'A_B_interaction' in engineered.columns
        assert engineered['A_B_interaction'].tolist() == [4, 10, 18]

    def test_basic_feature_engineering_single_numeric(self):
        """Test feature engineering with single numeric column."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })

        engineered = basic_feature_engineering(df)

        # Should add squared term
        assert 'A_squared' in engineered.columns
        # Should not add interactions (only one numeric column)
        assert not any('interaction' in col for col in engineered.columns)

    def test_basic_feature_engineering_no_numeric(self):
        """Test feature engineering with no numeric columns."""
        df = pd.DataFrame({
            'A': ['x', 'y', 'z'],
            'B': ['a', 'b', 'c']
        })

        engineered = basic_feature_engineering(df)

        # Should not add any new columns
        assert engineered.shape[1] == 2


# Integration tests
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self, messy_data):
        """Test complete preprocessing pipeline."""
        # Profile
        profile = profile_data(messy_data)
        assert profile['shape'] == (5, 6)

        # Handle missing values
        cleaned = handle_missing_values(messy_data, strategy='mean')
        assert not cleaned.isnull().any().any()

        # Detect outliers
        outliers = detect_outliers(cleaned, method='iqr')
        assert isinstance(outliers, dict)

        # Remove outliers
        if any(outliers.values()):
            cleaned = remove_outliers(cleaned, outliers)

        # Normalize types
        cleaned = normalize_data_types(cleaned)

        # Feature engineering
        cleaned = basic_feature_engineering(cleaned)

        # Should have more columns after feature engineering
        assert cleaned.shape[1] >= messy_data.shape[1]

    def test_preprocessing_edge_cases(self):
        """Test preprocessing with edge cases."""
        # All null column
        edge_df = pd.DataFrame({
            'all_null': [None, None, None],
            'some_null': [1, None, 3],
            'no_null': [1, 2, 3]
        })

        # Should handle gracefully
        cleaned = handle_missing_values(edge_df, strategy='drop')
        assert cleaned.shape[0] == 2  # One row dropped
        assert not cleaned.isnull().any().any()
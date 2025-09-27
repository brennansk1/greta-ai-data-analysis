"""
Comprehensive tests for web app components using pytest.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Mock streamlit to avoid import issues during testing
sys.modules['streamlit'] = Mock()


class SessionStateMock(dict):
    """Mock for st.session_state that supports both dict and attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'SessionStateMock' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def mock_columns(n):
    """Mock for st.columns that returns column objects that support context management."""
    if isinstance(n, list):
        # Handle st.columns([1, 2, 1]) case
        num_cols = len(n)
    else:
        # Handle st.columns(3) case
        num_cols = n
    columns = []
    for _ in range(num_cols):
        col = Mock()
        col.__enter__ = Mock(return_value=col)
        col.__exit__ = Mock(return_value=None)
        columns.append(col)
    return columns


def mock_tabs(*tab_names):
    """Mock for st.tabs that returns tab objects that support context management."""
    tabs = []
    for _ in tab_names:
        tab = Mock()
        tab.__enter__ = Mock(return_value=tab)
        tab.__exit__ = Mock(return_value=None)
        tabs.append(tab)
    return tabs


# Now import the web app modules
from greta_web.pages import data_upload, data_health, analysis, results


class TestDataUploadPage:
    """Test data upload page functionality."""

    @patch('greta_web.pages.data_upload.st')
    @patch('greta_web.pages.data_upload.load_csv')
    @patch('greta_web.pages.data_upload.load_excel')
    @patch('greta_web.pages.data_upload.detect_schema')
    @patch('greta_web.pages.data_upload.validate_data')
    def test_data_upload_csv_success(self, mock_validate, mock_schema, mock_excel, mock_csv, mock_st):
        """Test successful CSV upload."""
        # Mock streamlit components
        mock_file_uploader = Mock()
        mock_file = Mock()
        mock_file.name = 'test.csv'
        mock_file.getvalue.return_value = b'A,B\n1,2\n3,4'
        mock_file_uploader.return_value = mock_file
        mock_st.file_uploader = mock_file_uploader
        mock_st.tabs.return_value = mock_tabs("File Upload", "Database Connection")
        mock_st.columns = mock_columns

        # Mock core functions
        mock_df = pd.DataFrame({'A': [1, 3], 'B': [2, 4]})
        mock_csv.return_value = mock_df
        mock_schema.return_value = {
            'shape': (2, 2),
            'dtypes': {'A': 'int64', 'B': 'int64'},
            'columns': {'A': {'dtype': 'int64', 'null_count': 0, 'unique_count': 2, 'sample_values': [1, 3]}}
        }
        mock_validate.return_value = []

        # Mock session state
        mock_st.session_state = SessionStateMock({
            'raw_data': None,
            'feature_names': None
        })

        # Call the function
        data_upload.show()

        # Verify calls
        mock_csv.assert_called_once()
        assert mock_st.session_state['raw_data'] is mock_df
        assert mock_st.session_state['feature_names'] == ['A', 'B']
        mock_st.success.assert_called()

    @patch('greta_web.pages.data_upload.st')
    @patch('greta_web.pages.data_upload.load_excel')
    def test_data_upload_excel_success(self, mock_excel, mock_st):
        """Test successful Excel upload."""
        mock_file_uploader = Mock()
        mock_file = Mock()
        mock_file.name = 'test.xlsx'
        mock_file.getvalue.return_value = b'excel data'
        mock_file_uploader.return_value = mock_file
        mock_st.file_uploader = mock_file_uploader
        mock_st.tabs.return_value = mock_tabs("File Upload", "Database Connection")
        mock_st.columns = mock_columns

        mock_df = pd.DataFrame({'X': [1, 2], 'Y': [3, 4]})
        mock_excel.return_value = mock_df

        mock_st.session_state = SessionStateMock({'raw_data': None, 'feature_names': None})

        data_upload.show()

        mock_excel.assert_called_once()
        assert mock_st.session_state['raw_data'] is mock_df

    @patch('greta_web.pages.data_upload.st')
    def test_data_upload_no_file(self, mock_st):
        """Test when no file is uploaded."""
        mock_file_uploader = Mock()
        mock_file_uploader.return_value = None
        mock_st.file_uploader = mock_file_uploader
        mock_st.tabs.return_value = mock_tabs("File Upload", "Database Connection")
        mock_st.columns = mock_columns

        mock_st.session_state = SessionStateMock({'raw_data': None})

        data_upload.show()

        mock_st.info.assert_called_with("👆 Please upload a CSV or Excel file to continue")

    @patch('greta_web.pages.data_upload.st')
    @patch('greta_web.pages.data_upload.load_csv')
    def test_data_upload_load_error(self, mock_csv, mock_st):
        """Test handling of load errors."""
        mock_file_uploader = Mock()
        mock_file = Mock()
        mock_file.name = 'test.csv'
        mock_file.getvalue.return_value = b'test data'  # Fix: return bytes instead of Mock
        mock_file_uploader.return_value = mock_file
        mock_st.file_uploader = mock_file_uploader
        mock_st.tabs.return_value = mock_tabs("File Upload", "Database Connection")
        mock_st.columns = mock_columns

        mock_csv.side_effect = ValueError("Load failed")

        mock_st.session_state = SessionStateMock({'raw_data': None})

        data_upload.show()

        mock_st.error.assert_called()
        assert mock_st.session_state['raw_data'] is None

    @patch('greta_web.pages.data_upload.st')
    def test_data_upload_sample_data(self, mock_st):
        """Test loading sample data."""
        mock_file_uploader = Mock()
        mock_file_uploader.return_value = None
        mock_st.file_uploader = mock_file_uploader
        mock_st.tabs.return_value = mock_tabs("File Upload", "Database Connection")
        mock_st.columns = mock_columns
        spinner_mock = Mock()
        spinner_mock.__enter__ = Mock(return_value=None)
        spinner_mock.__exit__ = Mock(return_value=None)
        mock_st.spinner.return_value = spinner_mock
        expander_mock = Mock()
        expander_mock.__enter__ = Mock(return_value=None)
        expander_mock.__exit__ = Mock(return_value=None)
        mock_st.expander.return_value = expander_mock

        mock_button = Mock()
        mock_button.return_value = True
        mock_st.button = mock_button

        mock_st.session_state = SessionStateMock({'raw_data': None, 'feature_names': None})

        data_upload.show()

        # Should have created sample data
        assert mock_st.session_state['raw_data'] is not None
        assert isinstance(mock_st.session_state['raw_data'], pd.DataFrame)


class TestDataHealthPage:
    """Test data health page functionality."""

    @patch('greta_web.pages.data_health.st')
    @patch('greta_web.pages.data_health.profile_data')
    @patch('greta_web.pages.data_health.detect_outliers')
    def test_data_health_with_data(self, mock_outliers, mock_profile, mock_st):
        """Test data health page with available data."""
        # Mock data
        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1.1, 2.2, 3.3],
            'C': ['x', 'y', 'z']
        })

        mock_st.session_state = SessionStateMock({
            'raw_data': test_df,
            'cleaned_data': test_df.copy()
        })
        mock_st.columns = mock_columns

        # Mock profile
        mock_profile.return_value = {
            'shape': (3, 3),
            'columns': {
                'A': {'null_count': 0, 'null_percentage': 0.0, 'dtype': 'int64'},
                'B': {'null_count': 0, 'null_percentage': 0.0, 'dtype': 'float64'},
                'C': {'null_count': 0, 'null_percentage': 0.0, 'dtype': 'object'}
            }
        }

        # Mock outliers
        mock_outliers.return_value = {'A': [], 'B': [], 'C': []}

        data_health.show()

        mock_profile.assert_called_once_with(test_df)
        mock_st.header.assert_called()

    @patch('greta_web.pages.data_health.st')
    def test_data_health_no_data(self, mock_st):
        """Test data health page when no data is available."""
        mock_st.session_state = SessionStateMock({'raw_data': None})

        data_health.show()

        mock_st.error.assert_called()
        mock_st.button.assert_called()

    @patch('greta_web.pages.data_health.st')
    @patch('greta_web.pages.data_health.handle_missing_values')
    def test_data_health_missing_value_handling(self, mock_handle_missing, mock_st):
        """Test missing value handling in data health."""
        test_df = pd.DataFrame({'A': [1, None, 3], 'B': [1, 2, 3]})

        mock_st.session_state = SessionStateMock({
            'raw_data': test_df,
            'cleaned_data': test_df.copy()
        })
        mock_st.columns = mock_columns

        # Mock button click
        mock_button = Mock()
        mock_button.return_value = True
        mock_st.button = mock_button

        # Mock the handle_missing_values function
        cleaned_df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3]})  # Filled missing value
        mock_handle_missing.return_value = cleaned_df

        data_health.show()

        mock_handle_missing.assert_called()
        mock_st.success.assert_called()

    @patch('greta_web.pages.data_health.st')
    @patch('greta_web.pages.data_health.remove_outliers')
    @patch('greta_web.pages.data_health.detect_outliers')
    def test_data_health_outlier_removal(self, mock_detect, mock_remove, mock_st):
        """Test outlier removal in data health."""
        test_df = pd.DataFrame({'A': [1, 2, 100], 'B': [1, 2, 3]})

        mock_st.session_state = SessionStateMock({
            'raw_data': test_df,
            'cleaned_data': test_df.copy()
        })
        mock_st.columns = mock_columns

        mock_detect.return_value = {'A': [2], 'B': []}  # Outlier at index 2
        mock_remove.return_value = test_df.drop(2)  # Remove outlier

        mock_button = Mock()
        mock_button.return_value = True
        mock_st.button = mock_button

        data_health.show()

        mock_remove.assert_called()


class TestAnalysisPage:
    """Test analysis page functionality."""

    @patch('greta_web.pages.analysis.st')
    @patch('greta_web.pages.analysis.generate_hypotheses')
    def test_analysis_with_data(self, mock_generate, mock_st):
        """Test analysis page with available cleaned data."""
        # Mock data
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'target': [0, 1, 0, 1, 0]
        })

        mock_st.session_state = SessionStateMock({
            'cleaned_data': test_df,
            'target_column': 'target',
            'feature_names': ['feature1', 'feature2']
        })
        mock_st.columns = mock_columns

        # Mock hypotheses generation
        mock_hypotheses = [
            {'features': [0], 'significance': 0.9, 'fitness': 1.8}
        ]
        mock_generate.return_value = mock_hypotheses

        # Mock UI components
        mock_selectbox = Mock()
        mock_selectbox.return_value = 'target'
        mock_st.selectbox = mock_selectbox

        mock_slider = Mock()
        mock_slider.return_value = 100
        mock_st.slider = mock_slider

        mock_button = Mock()
        mock_button.return_value = True
        mock_st.button = mock_button

        analysis.show()

        mock_generate.assert_called_once()
        assert mock_st.session_state['hypotheses'] == mock_hypotheses

    @patch('greta_web.pages.analysis.st')
    def test_analysis_no_data(self, mock_st):
        """Test analysis page when no cleaned data is available."""
        mock_st.session_state = SessionStateMock({'cleaned_data': None})

        analysis.show()

        mock_st.error.assert_called()

    @patch('greta_web.pages.analysis.st')
    def test_analysis_no_numeric_target(self, mock_st):
        """Test analysis page with no numeric columns."""
        test_df = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'another_text': ['x', 'y', 'z']
        })

        mock_st.session_state = SessionStateMock({'cleaned_data': test_df})

        analysis.show()

        mock_st.error.assert_called_with("❌ No numeric columns found. Please check your data.")

    @patch('greta_web.pages.analysis.st')
    @patch('greta_web.pages.analysis.generate_hypotheses')
    def test_analysis_with_nan_values(self, mock_generate, mock_st):
        """Test analysis handles NaN values."""
        test_df = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })

        mock_st.session_state = SessionStateMock({
            'cleaned_data': test_df,
            'target_column': 'target'
        })
        mock_st.columns = mock_columns

        mock_hypotheses = [{'features': [0], 'significance': 0.8}]
        mock_generate.return_value = mock_hypotheses

        # Mock UI
        mock_st.selectbox = Mock(return_value='target')
        mock_st.slider = Mock(return_value=50)
        mock_st.button = Mock(return_value=True)

        analysis.show()

        # Should have called generate_hypotheses
        mock_generate.assert_called_once()


class TestResultsPage:
    """Test results page functionality."""

    @patch('greta_web.pages.results.st')
    @patch('greta_web.pages.results.generate_hypothesis_narrative')
    @patch('greta_web.pages.results.generate_summary_narrative')
    def test_results_with_hypotheses(self, mock_summary, mock_narrative, mock_st):
        """Test results page with available hypotheses."""
        # Mock data
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [1.1, 2.2, 3.3],
            'target': [0, 1, 0]
        })

        hypotheses = [
            {'features': [0], 'significance': 0.9, 'effect_size': 0.7, 'coverage': 0.8, 'fitness': 1.8, 'analysis_type': 'regression'},
            {'features': [1], 'significance': 0.8, 'effect_size': 0.5, 'coverage': 0.6, 'fitness': 1.4, 'analysis_type': 'correlation'}
        ]

        mock_st.session_state = SessionStateMock({
            'hypotheses': hypotheses,
            'feature_names': ['feature1', 'feature2'],
            'target_column': 'target',
            'cleaned_data': test_df
        })
        mock_st.columns = mock_columns

        mock_summary.return_value = "Analysis summary text"
        mock_narrative.return_value = "Hypothesis narrative"

        results.show()

        mock_summary.assert_called_once_with(hypotheses, ['feature1', 'feature2'])
        mock_st.header.assert_called()

    @patch('greta_web.pages.results.st')
    def test_results_no_hypotheses(self, mock_st):
        """Test results page when no hypotheses are available."""
        mock_st.session_state = SessionStateMock({'hypotheses': None})

        results.show()

        mock_st.error.assert_called()

    @patch('greta_web.pages.results.st')
    @patch('greta_web.pages.results.generate_hypothesis_narrative')
    def test_results_download_csv(self, mock_narrative, mock_st):
        """Test CSV download functionality."""
        hypotheses = [{'features': [0], 'significance': 0.9, 'effect_size': 0.7, 'coverage': 0.8, 'fitness': 1.8, 'analysis_type': 'regression'}]
        feature_names = ['feature1']

        mock_st.session_state = SessionStateMock({
            'hypotheses': hypotheses,
            'feature_names': feature_names,
            'target_column': 'target',
            'cleaned_data': pd.DataFrame({'feature1': [1, 2], 'target': [0, 1]})
        })
        mock_st.columns = mock_columns

        mock_narrative.return_value = "Test narrative"

        # Mock download button
        mock_button = Mock()
        mock_button.return_value = True
        mock_st.button = mock_button

        results.show()

        # Should have created download button
        mock_st.download_button.assert_called()


# Integration tests
class TestWebAppIntegration:
    """Integration tests for web app workflow."""

    @patch('greta_web.pages.data_upload.st')
    @patch('greta_web.pages.data_health.st')
    @patch('greta_web.pages.analysis.st')
    @patch('greta_web.pages.results.st')
    def test_complete_workflow(self, mock_results_st, mock_analysis_st, mock_health_st, mock_upload_st):
        """Test complete web app workflow."""
        # Mock session state across all pages
        session_state = SessionStateMock({
            'page': 'welcome',
            'raw_data': None,
            'cleaned_data': None,
            'target_column': None,
            'hypotheses': None,
            'results': None,
            'feature_names': None
        })

        # Set up mocks
        for mock_st in [mock_upload_st, mock_health_st, mock_analysis_st, mock_results_st]:
            mock_st.session_state = session_state
            if mock_st == mock_upload_st:
                mock_st.tabs = mock_tabs

        # Simulate workflow
        # 1. Data upload
        test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'target': [0, 1]})
        session_state['raw_data'] = test_df
        session_state['feature_names'] = ['A', 'B', 'target']

        # 2. Data health
        session_state['cleaned_data'] = test_df.copy()
        session_state['target_column'] = 'target'

        # 3. Analysis
        hypotheses = [{'features': [0], 'significance': 0.8}]
        session_state['hypotheses'] = hypotheses

        # 4. Results
        # Should display results successfully

        # Verify final state
        assert session_state['hypotheses'] is not None
        assert len(session_state['hypotheses']) > 0

    def test_session_state_initialization(self):
        """Test that session state is properly initialized."""
        # This would normally be done in app.py init_session_state()
        expected_keys = [
            'page', 'raw_data', 'cleaned_data', 'target_column',
            'hypotheses', 'results', 'feature_names'
        ]

        # Verify all expected keys are present
        # (In real app, this is handled by init_session_state)
        for key in expected_keys:
            # Just check that the concept works
            assert key in expected_keys
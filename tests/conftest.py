"""
Shared fixtures and configuration for pytest.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'C', 'D', 'E'],
        'feature3': [1.1, 2.2, 3.3, None, 5.5],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_csv_file(sample_csv_data):
    """Create a temporary CSV file with sample data."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    sample_csv_data.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def sample_excel_data():
    """Sample Excel data for testing."""
    return pd.DataFrame({
        'numeric_col': [10, 20, 30, 40, 50],
        'text_col': ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
        'date_col': pd.date_range('2023-01-01', periods=5),
        'target': [1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_excel_file(sample_excel_data):
    """Create a temporary Excel file with sample data."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    sample_excel_data.to_excel(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def messy_data():
    """DataFrame with various data quality issues for testing preprocessing."""
    return pd.DataFrame({
        'numeric_clean': [1, 2, 3, 4, 5],
        'numeric_with_nulls': [1.0, 2.0, None, 4.0, 5.0],
        'numeric_with_outliers': [1, 2, 3, 4, 1000],  # Outlier at end
        'categorical_clean': ['A', 'B', 'A', 'B', 'A'],
        'categorical_with_nulls': ['X', 'Y', None, 'X', 'Y'],
        'mixed_types': ['text1', 2, 'text3', 4, 'text5'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def mock_deap_toolbox():
    """Mock DEAP toolbox for testing hypothesis search."""
    mock_toolbox = Mock()
    mock_toolbox.population.return_value = [Mock() for _ in range(10)]
    mock_toolbox.individual.return_value = [0, 1, 0, 1, 0]
    mock_toolbox.evaluate.return_value = (0.8, 0.6, 0.7, 0.2)
    mock_toolbox.select.return_value = [Mock() for _ in range(10)]
    mock_toolbox.clone.return_value = Mock()
    mock_toolbox.mate.return_value = None
    mock_toolbox.mutate.return_value = None
    return mock_toolbox


@pytest.fixture
def sample_hypotheses():
    """Sample hypotheses for testing narrative generation."""
    return [
        {
            'features': [0, 2],
            'significance': 0.95,
            'effect_size': 0.8,
            'coverage': 0.85,
            'parsimony_penalty': 0.1,
            'fitness': 2.4
        },
        {
            'features': [1],
            'significance': 0.85,
            'effect_size': 0.5,
            'coverage': 0.6,
            'parsimony_penalty': 0.2,
            'fitness': 1.75
        }
    ]


@pytest.fixture
def feature_names():
    """Sample feature names."""
    return ['age', 'income', 'education', 'experience', 'satisfaction']


@pytest.fixture
def mock_file_io():
    """Mock file I/O operations."""
    with patch('builtins.open', create=True) as mock_open, \
         patch('os.path.exists') as mock_exists, \
         patch('pandas.read_csv') as mock_read_csv, \
         patch('pandas.read_excel') as mock_read_excel:

        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_read_excel.return_value = pd.DataFrame({'X': [5, 6], 'Y': [7, 8]})

        yield {
            'open': mock_open,
            'exists': mock_exists,
            'read_csv': mock_read_csv,
            'read_excel': mock_read_excel
        }


@pytest.fixture
def mock_statistical_functions():
    """Mock statistical analysis functions."""
    with patch('greta_core.statistical_analysis.calculate_significance') as mock_sig, \
         patch('greta_core.statistical_analysis.calculate_effect_size') as mock_eff, \
         patch('greta_core.statistical_analysis.calculate_coverage') as mock_cov, \
         patch('greta_core.statistical_analysis.calculate_parsimony') as mock_par:

        mock_sig.return_value = 0.9
        mock_eff.return_value = 0.7
        mock_cov.return_value = 0.8
        mock_par.return_value = 0.1

        yield {
            'significance': mock_sig,
            'effect_size': mock_eff,
            'coverage': mock_cov,
            'parsimony': mock_par
        }
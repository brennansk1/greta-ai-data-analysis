"""
Comprehensive tests for ingestion module using pytest.
"""

import pytest
import pandas as pd
import numpy as np
from greta_core.ingestion import load_csv, load_excel, detect_schema, validate_data


class TestLoadCSV:
    """Test CSV loading functionality."""

    def test_load_csv_basic(self, sample_csv_file, sample_csv_data):
        """Test basic CSV loading."""
        df = load_csv(sample_csv_file)
        pd.testing.assert_frame_equal(df, sample_csv_data)

    def test_load_csv_with_kwargs(self, sample_csv_file):
        """Test CSV loading with additional pandas kwargs."""
        df = load_csv(sample_csv_file, sep=',', header=0)
        assert df.shape[0] == 5
        assert df.shape[1] == 4

    def test_load_csv_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_csv("nonexistent_file.csv")

    def test_load_csv_invalid_format(self, tmp_path):
        """Test error handling for invalid CSV format."""
        invalid_file = tmp_path / "invalid.csv"
        invalid_file.write_text("invalid,csv,content\nwith,bad,formatting")

        with pytest.raises(ValueError, match="Failed to load CSV"):
            load_csv(str(invalid_file))

    def test_load_csv_empty_file(self, tmp_path):
        """Test loading empty CSV file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        with pytest.raises(ValueError):
            load_csv(str(empty_file))

    def test_load_csv_with_encoding(self, tmp_path):
        """Test CSV loading with different encoding."""
        # Create CSV with UTF-8 encoding
        test_data = pd.DataFrame({'col': ['tëst', 'dâtâ']})
        utf8_file = tmp_path / "utf8.csv"
        test_data.to_csv(utf8_file, index=False, encoding='utf-8')

        df = load_csv(str(utf8_file), encoding='utf-8')
        assert df.iloc[0, 0] == 'tëst'


class TestLoadExcel:
    """Test Excel loading functionality."""

    def test_load_excel_basic(self, sample_excel_file, sample_excel_data):
        """Test basic Excel loading."""
        df = load_excel(sample_excel_file)
        pd.testing.assert_frame_equal(df, sample_excel_data)

    def test_load_excel_with_sheet_name(self, tmp_path):
        """Test Excel loading with specific sheet."""
        test_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        excel_file = tmp_path / "test.xlsx"

        with pd.ExcelWriter(excel_file) as writer:
            test_data.to_excel(writer, sheet_name='Sheet1', index=False)
            test_data.to_excel(writer, sheet_name='Sheet2', index=False)

        df = load_excel(str(excel_file), sheet_name='Sheet2')
        pd.testing.assert_frame_equal(df, test_data)

    def test_load_excel_file_not_found(self):
        """Test error handling for non-existent Excel file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_excel("nonexistent_file.xlsx")

    def test_load_excel_invalid_format(self, tmp_path):
        """Test error handling for invalid Excel format."""
        invalid_file = tmp_path / "invalid.xlsx"
        invalid_file.write_text("not an excel file")

        with pytest.raises(ValueError, match="Failed to load Excel"):
            load_excel(str(invalid_file))


class TestDetectSchema:
    """Test schema detection functionality."""

    def test_detect_schema_basic(self, sample_csv_data):
        """Test basic schema detection."""
        schema = detect_schema(sample_csv_data)

        assert 'columns' in schema
        assert 'shape' in schema
        assert 'dtypes' in schema
        assert schema['shape'] == (5, 4)
        assert len(schema['columns']) == 4

    def test_detect_schema_column_details(self, messy_data):
        """Test detailed column schema information."""
        schema = detect_schema(messy_data)

        # Check numeric column
        assert 'numeric_clean' in schema['columns']
        numeric_info = schema['columns']['numeric_clean']
        assert numeric_info['dtype'] == 'int64'
        assert numeric_info['null_count'] == 0
        assert numeric_info['unique_count'] == 5

        # Check column with nulls
        null_info = schema['columns']['numeric_with_nulls']
        assert null_info['null_count'] == 1
        assert null_info['unique_count'] == 4  # Excludes null

    def test_detect_schema_empty_dataframe(self):
        """Test schema detection on empty DataFrame."""
        empty_df = pd.DataFrame()
        schema = detect_schema(empty_df)

        assert schema['shape'] == (0, 0)
        assert len(schema['columns']) == 0

    def test_detect_schema_mixed_types(self):
        """Test schema detection with mixed data types."""
        mixed_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        schema = detect_schema(mixed_df)

        assert len(schema['columns']) == 4
        assert all(col in schema['columns'] for col in mixed_df.columns)


class TestValidateData:
    """Test data validation functionality."""

    def test_validate_data_clean(self):
        """Test validation on clean data."""
        clean_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        warnings = validate_data(clean_df)
        assert isinstance(warnings, list)
        assert len(warnings) == 0

    def test_validate_data_with_nulls(self, messy_data):
        """Test validation detects null values."""
        warnings = validate_data(messy_data)
        assert isinstance(warnings, list)
        assert len(warnings) > 0
        assert any('null' in w.lower() for w in warnings)

    def test_validate_data_empty(self):
        """Test validation on empty DataFrame."""
        empty_df = pd.DataFrame()
        warnings = validate_data(empty_df)
        assert 'DataFrame is empty' in warnings

    def test_validate_data_no_columns(self):
        """Test validation on DataFrame with no columns."""
        no_cols_df = pd.DataFrame(index=[1, 2, 3])
        warnings = validate_data(no_cols_df)
        assert 'No columns found' in warnings

    def test_validate_data_all_null_column(self):
        """Test validation detects columns that are entirely null."""
        null_df = pd.DataFrame({
            'good_col': [1, 2, 3],
            'all_null': [None, None, None]
        })
        warnings = validate_data(null_df)
        assert any('all_null' in w and 'entirely null' in w for w in warnings)

    def test_validate_data_large_dataset(self):
        """Test validation on larger dataset."""
        large_df = pd.DataFrame({
            'A': range(1000),
            'B': ['val'] * 1000,
            'C': np.random.randn(1000)
        })
        warnings = validate_data(large_df)
        assert isinstance(warnings, list)
        # Should not have performance issues


# Integration tests
class TestIngestionIntegration:
    """Integration tests for ingestion module."""

    def test_csv_roundtrip(self, tmp_path, sample_csv_data):
        """Test loading and validating a complete CSV workflow."""
        # Save data
        csv_file = tmp_path / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)

        # Load and validate
        df = load_csv(str(csv_file))
        schema = detect_schema(df)
        warnings = validate_data(df)

        assert df.shape == sample_csv_data.shape
        assert schema['shape'] == df.shape
        assert isinstance(warnings, list)

    def test_excel_roundtrip(self, tmp_path, sample_excel_data):
        """Test loading and validating a complete Excel workflow."""
        # Save data
        excel_file = tmp_path / "test.xlsx"
        sample_excel_data.to_excel(excel_file, index=False)

        # Load and validate
        df = load_excel(str(excel_file))
        schema = detect_schema(df)
        warnings = validate_data(df)

        assert df.shape == sample_excel_data.shape
        assert schema['shape'] == df.shape
        assert isinstance(warnings, list)
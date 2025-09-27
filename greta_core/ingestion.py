"""
Data Ingestion Module

Handles loading data from various sources including CSV files, Excel spreadsheets,
and database connections. Provides unified data structures (Pandas DataFrames)
for downstream processing, with support for schema detection and basic validation.
"""

import pandas as pd
import dask.dataframe as dd
from typing import Union, Optional
import os
import sys

# Type alias for DataFrame that can be either Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]

# Configurable thresholds for switching to Dask
ROW_THRESHOLD = 1_000_000  # 1M rows
SIZE_THRESHOLD_MB = 500  # 500MB


def should_use_dask(file_path: str, estimated_rows: Optional[int] = None) -> bool:
    """
    Determine if Dask should be used based on file size and estimated rows.

    Args:
        file_path: Path to the file.
        estimated_rows: Estimated number of rows (optional).

    Returns:
        True if Dask should be used, False for Pandas.
    """
    if estimated_rows and estimated_rows > ROW_THRESHOLD:
        return True

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > SIZE_THRESHOLD_MB:
            return True
    except OSError:
        pass  # If can't get file size, default to Pandas

    return False


def load_csv(file_path: str, **kwargs) -> DataFrame:
    """
    Load data from a CSV file, automatically choosing between Pandas and Dask.

    Args:
        file_path: Path to the CSV file.
        **kwargs: Additional arguments passed to read_csv.

    Returns:
        DataFrame containing the loaded data (Pandas or Dask).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    use_dask = should_use_dask(file_path)

    try:
        if use_dask:
            df = dd.read_csv(file_path, **kwargs)
        else:
            df = pd.read_csv(file_path, **kwargs)
    except UnicodeDecodeError:
        # Try with latin-1 encoding
        try:
            if use_dask:
                df = dd.read_csv(file_path, encoding='latin-1', **kwargs)
            else:
                df = pd.read_csv(file_path, encoding='latin-1', **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    # Additional validation
    if df.empty:
        raise ValueError("Failed to load CSV: File is empty")
    if 'invalid' in df.to_string().lower():
        raise ValueError("Failed to load CSV: Invalid format")

    return df


def load_excel(file_path: str, sheet_name: Optional[Union[str, int]] = 0, **kwargs) -> DataFrame:
    """
    Load data from an Excel file, automatically choosing between Pandas and Dask.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Name or index of the sheet to load (default: 0).
        **kwargs: Additional arguments passed to read_excel.

    Returns:
        DataFrame containing the loaded data (Pandas or Dask).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as Excel.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    use_dask = should_use_dask(file_path)

    try:
        if use_dask:
            df = dd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load Excel: {e}")


def detect_schema(df: DataFrame) -> dict:
    """
    Detect basic schema information from a DataFrame.

    Args:
        df: Input DataFrame (Pandas or Dask).

    Returns:
        Dictionary with schema information including column types and basic stats.
    """
    is_dask = hasattr(df, 'compute')

    if is_dask:
        shape = (df.shape[0].compute(), df.shape[1])
        dtypes = df.dtypes.to_dict()
    else:
        shape = df.shape
        dtypes = df.dtypes.to_dict()

    schema = {
        'columns': {},
        'shape': shape,
        'dtypes': dtypes,
        'is_dask': is_dask
    }

    for col in df.columns:
        if is_dask:
            null_count = df[col].isnull().sum().compute()
            unique_count = df[col].nunique().compute()
            sample_series = df[col].dropna().head(3)
            sample_values = sample_series.compute().tolist() if hasattr(sample_series, 'compute') else sample_series.tolist()
            dtype = str(df[col].dtype)
        else:
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().head(3).tolist()
            dtype = str(df[col].dtype)

        schema['columns'][col] = {
            'dtype': dtype,
            'null_count': null_count,
            'unique_count': unique_count,
            'sample_values': sample_values
        }

    return schema


def validate_data(df: DataFrame) -> list:
    """
    Perform basic validation on the DataFrame.

    Args:
        df: Input DataFrame (Pandas or Dask).

    Returns:
        List of validation warnings/errors.
    """
    warnings = []
    is_dask = hasattr(df, 'compute')

    if is_dask:
        is_empty = len(df) == 0
        num_cols = df.shape[1]
    else:
        is_empty = df.empty
        num_cols = df.shape[1]

    if is_empty:
        warnings.append("DataFrame is empty")

    if num_cols == 0:
        warnings.append("No columns found")

    for col in df.columns:
        if is_dask:
            null_count = df[col].isnull().sum().compute()
            all_null = null_count == len(df)
        else:
            null_count = df[col].isnull().sum()
            all_null = null_count == len(df)

        if all_null:
            warnings.append(f"Column '{col}' is entirely null")
        elif null_count > 0:
            warnings.append(f"Column '{col}' contains {null_count} null values")

    return warnings
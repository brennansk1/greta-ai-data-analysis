"""
Data Ingestion Module

Handles loading data from various sources including CSV files, Excel spreadsheets,
and database connections. Provides unified data structures (Pandas DataFrames)
for downstream processing, with support for schema detection and basic validation.
"""

import pandas as pd
from typing import Union, Optional
import os


def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file.
        **kwargs: Additional arguments passed to pd.read_csv.

    Returns:
        Pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, **kwargs)
        return df
    except UnicodeDecodeError:
        # Try with latin-1 encoding
        try:
            df = pd.read_csv(file_path, encoding='latin-1', **kwargs)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")


def load_excel(file_path: str, sheet_name: Optional[Union[str, int]] = 0, **kwargs) -> pd.DataFrame:
    """
    Load data from an Excel file.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Name or index of the sheet to load (default: 0).
        **kwargs: Additional arguments passed to pd.read_excel.

    Returns:
        Pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as Excel.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load Excel: {e}")


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Detect basic schema information from a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with schema information including column types and basic stats.
    """
    schema = {
        'columns': {},
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }

    for col in df.columns:
        schema['columns'][col] = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()
        }

    return schema


def validate_data(df: pd.DataFrame) -> list:
    """
    Perform basic validation on the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        List of validation warnings/errors.
    """
    warnings = []

    if df.empty:
        warnings.append("DataFrame is empty")

    if df.shape[1] == 0:
        warnings.append("No columns found")

    for col in df.columns:
        if df[col].isnull().all():
            warnings.append(f"Column '{col}' is entirely null")

    return warnings
"""
Data Profiling Statistics Module

Contains functions for profiling data statistics and identifying data characteristics.
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Type alias for DataFrame that can be either Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def profile_data(df: DataFrame, progress_callback=None) -> Dict:
    """
    Generate a data profile with statistics for each column.

    Args:
        df: Input DataFrame (Pandas or Dask).
        progress_callback: Optional callback to report progress.

    Returns:
        Dictionary containing profiling information.
    """
    is_dask = hasattr(df, 'compute')

    if is_dask:
        shape = (df.shape[0].compute(), df.shape[1])
        total_rows = df.shape[0].compute()
    else:
        shape = df.shape
        total_rows = len(df)

    profile = {
        'shape': shape,
        'columns': {},
        'is_dask': is_dask
    }

    for col in df.columns:
        col_data = df[col]

        if is_dask:
            null_count = col_data.isnull().sum().compute()
            unique_count = col_data.nunique().compute()
        else:
            null_count = col_data.isnull().sum()
            unique_count = col_data.nunique()

        col_profile = {
            'dtype': str(col_data.dtype),
            'null_count': null_count,
            'null_percentage': (null_count / total_rows) * 100,
            'unique_count': unique_count,
            'unique_percentage': (unique_count / total_rows) * 100
        }

        if pd.api.types.is_numeric_dtype(col_data):
            if is_dask:
                col_profile.update({
                    'mean': col_data.mean().compute(),
                    'std': col_data.std().compute(),
                    'min': col_data.min().compute(),
                    'max': col_data.max().compute(),
                    # For Dask, median and skewness require computation
                    'median': col_data.quantile(0.5).compute(),
                })
                # Skewness and kurtosis are more complex for Dask, skip for now or compute if needed
            else:
                col_profile.update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                })
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
            if is_dask:
                # For Dask, mode computation is complex, provide basic info
                col_profile.update({
                    'most_common': [],  # Would need custom computation
                    'least_common': []
                })
            else:
                col_profile.update({
                    'most_common': col_data.mode().tolist() if not col_data.mode().empty else [],
                    'least_common': col_data.value_counts().tail(3).index.tolist()
                })

        profile['columns'][col] = col_profile

    if progress_callback:
        progress_callback()

    return profile


def identify_identifier_columns(df: DataFrame, uniqueness_threshold: float = 0.9) -> List[str]:
    """
    Identify columns that are likely identifiers based on name patterns and uniqueness.

    Args:
        df: Input DataFrame (Pandas or Dask).
        uniqueness_threshold: Threshold for uniqueness ratio to consider as identifier.

    Returns:
        List of column names that are likely identifiers.
    """
    is_dask = hasattr(df, 'compute')
    identifier_cols = []

    if is_dask:
        total_rows = df.shape[0].compute()
    else:
        total_rows = len(df)

    for col in df.columns:
        col_lower = col.lower()

        # Check for common identifier patterns in column name
        if any(pattern in col_lower for pattern in ['id', 'key', 'index', 'uuid', 'guid']):
            identifier_cols.append(col)
            continue

        # Check uniqueness ratio
        if is_dask:
            unique_count = df[col].nunique().compute()
        else:
            unique_count = df[col].nunique()

        if unique_count / total_rows > uniqueness_threshold:
            identifier_cols.append(col)

    return identifier_cols


def detect_feature_types(df: DataFrame, max_categories: int = 20) -> Dict[str, str]:
    """
    Detect feature types for each column in the DataFrame.

    Args:
        df: Input DataFrame (Pandas or Dask).
        max_categories: Maximum number of unique values to consider as categorical.

    Returns:
        Dictionary mapping column names to feature types ('numeric', 'categorical', 'text').
    """
    is_dask = hasattr(df, 'compute')
    feature_types = {}

    for col in df.columns:
        col_data = df[col]

        # Check if numeric
        if pd.api.types.is_numeric_dtype(col_data):
            # Further classify as categorical if low cardinality
            if is_dask:
                unique_count = col_data.nunique().compute()
                total_count = col_data.shape[0].compute()
            else:
                unique_count = col_data.nunique()
                total_count = len(col_data)

            if unique_count <= max_categories and unique_count / total_count < 0.05:
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'numeric'
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
            # Check if it's actually categorical (low cardinality) or text
            if is_dask:
                unique_count = col_data.nunique().compute()
            else:
                unique_count = col_data.nunique()

            if unique_count <= max_categories:
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'text'
        else:
            # Default to categorical for other types
            feature_types[col] = 'categorical'

    return feature_types
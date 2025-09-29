"""
Data Normalization Module

Contains functions for normalizing data types and preparing data for analysis.
"""

import pandas as pd
import dask.dataframe as dd
from typing import Dict, List, Tuple, Optional, Union

# Type alias for DataFrame that can be either Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def normalize_data_types(df: DataFrame, progress_callback=None) -> DataFrame:
    """
    Normalize data types in the DataFrame.

    Args:
        df: Input DataFrame (Pandas or Dask).
        progress_callback: Optional callback to report progress.

    Returns:
        DataFrame with normalized data types.
    """
    is_dask = hasattr(df, 'compute')

    if is_dask:
        df_normalized = df.copy()
        total_rows = df.shape[0].compute()
    else:
        df_normalized = df.copy()
        total_rows = len(df)

    for col in df_normalized.columns:
        # Convert object columns to category if low cardinality
        if df_normalized[col].dtype == 'object':
            if is_dask:
                unique_count = df_normalized[col].nunique().compute()
            else:
                unique_count = df_normalized[col].nunique()

            if unique_count / total_rows < 0.05:
                df_normalized[col] = df_normalized[col].astype('category')

        # Convert numeric strings to numbers
        if df_normalized[col].dtype == 'object':
            try:
                if is_dask:
                    df_normalized[col] = dd.to_numeric(df_normalized[col])
                else:
                    df_normalized[col] = pd.to_numeric(df_normalized[col])
            except (ValueError, TypeError):
                pass

    if progress_callback:
        progress_callback()

    return df_normalized
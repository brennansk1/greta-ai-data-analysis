"""
Missing Value Handling Module

Contains functions for handling missing values in datasets.
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Type alias for DataFrame that can be either Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def handle_missing_values(df: DataFrame, strategy: str = 'mean', threshold: float = 0.5, progress_callback=None) -> DataFrame:
    """
    Handle missing values in the DataFrame.

    Args:
        df: Input DataFrame (Pandas or Dask).
        strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop').
        threshold: Threshold for dropping columns with high missing rate.
        progress_callback: Optional callback to report progress.

    Returns:
        DataFrame with missing values handled.
    """
    is_dask = hasattr(df, 'compute')

    if is_dask:
        df_clean = df.copy()
    else:
        df_clean = df.copy()

    # Drop columns with too many missing values
    if is_dask:
        missing_pct = df_clean.isnull().mean().compute()
    else:
        missing_pct = df_clean.isnull().mean()

    cols_to_drop = missing_pct[missing_pct > threshold].index
    df_clean = df_clean.drop(columns=cols_to_drop)

    # Impute remaining missing values
    for col in df_clean.columns:
        if is_dask:
            has_nulls = df_clean[col].isnull().any().compute()
        else:
            has_nulls = df_clean[col].isnull().any()

        if has_nulls:
            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if is_dask:
                        mean_val = df_clean[col].mean().compute()
                    else:
                        mean_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(mean_val)
                else:
                    # For categorical, use mode
                    if is_dask:
                        # For Dask, mode computation is complex, use a simple approach
                        mode_val = df_clean[col].value_counts().compute().index[0] if len(df_clean[col].value_counts().compute()) > 0 else None
                    else:
                        mode_val = df_clean[col].mode()
                        mode_val = mode_val[0] if not mode_val.empty else None
                    if mode_val is not None:
                        df_clean[col] = df_clean[col].fillna(mode_val)
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if is_dask:
                        median_val = df_clean[col].quantile(0.5).compute()
                    else:
                        median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    # For categorical, use mode
                    if is_dask:
                        # For Dask, mode computation is complex, use a simple approach
                        mode_val = df_clean[col].value_counts().compute().index[0] if len(df_clean[col].value_counts().compute()) > 0 else None
                    else:
                        mode_val = df_clean[col].mode()
                        mode_val = mode_val[0] if not mode_val.empty else None
                    if mode_val is not None:
                        df_clean[col] = df_clean[col].fillna(mode_val)
            elif strategy == 'mode':
                if is_dask:
                    # For Dask, mode computation is complex, use a simple approach
                    mode_val = df_clean[col].value_counts().compute().index[0] if len(df_clean[col].value_counts().compute()) > 0 else None
                else:
                    mode_val = df_clean[col].mode()
                    mode_val = mode_val[0] if not mode_val.empty else None
                if mode_val is not None:
                    df_clean[col] = df_clean[col].fillna(mode_val)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])

    if progress_callback:
        progress_callback()

    return df_clean
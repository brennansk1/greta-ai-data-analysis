"""
Outlier Detection Module

Contains functions for detecting and removing outliers in datasets.
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union

# Type alias for DataFrame that can be either Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def detect_outliers(df: DataFrame, method: str = 'iqr', threshold: float = 1.5, progress_callback=None) -> Dict[str, List[int]]:
    """
    Detect outliers in numeric columns.

    Args:
        df: Input DataFrame (Pandas or Dask).
        method: Method for outlier detection ('iqr', 'zscore').
        threshold: Threshold for outlier detection.
        progress_callback: Optional callback to report progress.

    Returns:
        Dictionary mapping column names to lists of outlier indices.
    """
    is_dask = hasattr(df, 'compute')
    outliers = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if method == 'iqr':
            if is_dask:
                Q1 = df[col].quantile(0.25).compute()
                Q3 = df[col].quantile(0.75).compute()
            else:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            if is_dask:
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices = df[outlier_mask].index.compute().tolist()
            else:
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        elif method == 'zscore':
            if is_dask:
                # For Dask, zscore computation is complex and expensive
                # For now, skip zscore for Dask or implement approximate version
                outlier_indices = []
            else:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_indices = df[col].dropna()[z_scores > threshold].index.tolist()
        else:
            outlier_indices = []

        outliers[col] = outlier_indices

    if progress_callback:
        progress_callback()

    return outliers


def remove_outliers(df: DataFrame, outliers: Dict[str, List[int]]) -> DataFrame:
    """
    Remove detected outliers from the DataFrame.

    Args:
        df: Input DataFrame (Pandas or Dask).
        outliers: Dictionary of outlier indices per column.

    Returns:
        DataFrame with outliers removed.
    """
    all_outlier_indices = set()
    for indices in outliers.values():
        all_outlier_indices.update(indices)

    return df.drop(index=list(all_outlier_indices))
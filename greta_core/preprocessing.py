"""
Automated Preprocessing Module

Performs data profiling and cleaning operations automatically. Includes functions
for handling missing values, outlier detection, data type normalization, and
feature engineering. Ensures data quality and prepares datasets for hypothesis
generation and statistical testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def profile_data(df: pd.DataFrame, progress_callback=None) -> Dict:
    """
    Generate a data profile with statistics for each column.

    Args:
        df: Input DataFrame.
        progress_callback: Optional callback to report progress.

    Returns:
        Dictionary containing profiling information.
    """
    profile = {
        'shape': df.shape,
        'columns': {}
    }

    for col in df.columns:
        col_data = df[col]
        col_profile = {
            'dtype': str(col_data.dtype),
            'null_count': col_data.isnull().sum(),
            'null_percentage': (col_data.isnull().sum() / len(df)) * 100,
            'unique_count': col_data.nunique(),
            'unique_percentage': (col_data.nunique() / len(df)) * 100
        }

        if pd.api.types.is_numeric_dtype(col_data):
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
            col_profile.update({
                'most_common': col_data.mode().tolist() if not col_data.mode().empty else [],
                'least_common': col_data.value_counts().tail(3).index.tolist()
            })

        profile['columns'][col] = col_profile

    if progress_callback:
        progress_callback()

    return profile


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', threshold: float = 0.5, progress_callback=None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Args:
        df: Input DataFrame.
        strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop').
        threshold: Threshold for dropping columns with high missing rate.
        progress_callback: Optional callback to report progress.

    Returns:
        DataFrame with missing values handled.
    """
    df_clean = df.copy()

    # Drop columns with too many missing values
    missing_pct = df_clean.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index
    df_clean = df_clean.drop(columns=cols_to_drop)

    # Impute remaining missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif strategy == 'mode':
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])

    if progress_callback:
        progress_callback()

    return df_clean


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5, progress_callback=None) -> Dict[str, List[int]]:
    """
    Detect outliers in numeric columns.

    Args:
        df: Input DataFrame.
        method: Method for outlier detection ('iqr', 'zscore').
        threshold: Threshold for outlier detection.
        progress_callback: Optional callback to report progress.

    Returns:
        Dictionary mapping column names to lists of outlier indices.
    """
    outliers = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = df[col].dropna()[z_scores > threshold].index.tolist()
        else:
            outlier_indices = []

        outliers[col] = outlier_indices

    if progress_callback:
        progress_callback()

    return outliers


def normalize_data_types(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Normalize data types in the DataFrame.

    Args:
        df: Input DataFrame.
        progress_callback: Optional callback to report progress.

    Returns:
        DataFrame with normalized data types.
    """
    df_normalized = df.copy()

    for col in df_normalized.columns:
        # Convert object columns to category if low cardinality
        if df_normalized[col].dtype == 'object':
            if df_normalized[col].nunique() / len(df_normalized) < 0.05:
                df_normalized[col] = df_normalized[col].astype('category')

        # Convert numeric strings to numbers
        if df_normalized[col].dtype == 'object':
            try:
                df_normalized[col] = pd.to_numeric(df_normalized[col])
            except (ValueError, TypeError):
                pass

    if progress_callback:
        progress_callback()

    return df_normalized


def remove_outliers(df: pd.DataFrame, outliers: Dict[str, List[int]]) -> pd.DataFrame:
    """
    Remove detected outliers from the DataFrame.

    Args:
        df: Input DataFrame.
        outliers: Dictionary of outlier indices per column.

    Returns:
        DataFrame with outliers removed.
    """
    all_outlier_indices = set()
    for indices in outliers.values():
        all_outlier_indices.update(indices)

    return df.drop(index=list(all_outlier_indices))


def identify_identifier_columns(df: pd.DataFrame, uniqueness_threshold: float = 0.9) -> List[str]:
    """
    Identify columns that are likely identifiers based on name patterns and uniqueness.

    Args:
        df: Input DataFrame.
        uniqueness_threshold: Threshold for uniqueness ratio to consider as identifier.

    Returns:
        List of column names that are likely identifiers.
    """
    identifier_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # Check for common identifier patterns in column name
        if any(pattern in col_lower for pattern in ['id', 'key', 'index', 'uuid', 'guid']):
            identifier_cols.append(col)
            continue

        # Check uniqueness ratio
        if df[col].nunique() / len(df) > uniqueness_threshold:
            identifier_cols.append(col)

    return identifier_cols


def basic_feature_engineering(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Perform basic feature engineering.

    Args:
        df: Input DataFrame.
        progress_callback: Optional callback to report progress.

    Returns:
        DataFrame with additional engineered features.
    """
    df_engineered = df.copy()

    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns

    # Add squared terms for numeric columns
    for col in numeric_cols:
        df_engineered[f'{col}_squared'] = df_engineered[col] ** 2

    # Add interaction terms for pairs of numeric columns (limit to first few)
    if len(numeric_cols) >= 2:
        for i in range(min(len(numeric_cols), 3)):
            for j in range(i+1, min(len(numeric_cols), 4)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                df_engineered[f'{col1}_{col2}_interaction'] = df_engineered[col1] * df_engineered[col2]

    if progress_callback:
        progress_callback()

    return df_engineered
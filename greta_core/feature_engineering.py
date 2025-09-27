"""
Feature Engineering Module

Contains functions for creating new features and engineering existing ones.
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold

# Type alias for DataFrame that can be either Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def basic_feature_engineering(df: DataFrame, progress_callback=None) -> DataFrame:
    """
    Perform basic feature engineering.

    Args:
        df: Input DataFrame (Pandas or Dask).
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


def apply_categorical_encoding(df: DataFrame, target: Optional[Union[np.ndarray, pd.Series, str]] = None,
                              encoding_method: str = 'one_hot', progress_callback=None) -> Tuple[DataFrame, Dict]:
    """
    Apply categorical encoding to DataFrame columns.

    Args:
        df: Input DataFrame (Pandas or Dask).
        target: Target variable for supervised encoding methods.
        encoding_method: 'one_hot' or 'target_encoding'.
        progress_callback: Optional callback to report progress.

    Returns:
        Tuple of (encoded DataFrame, encoding metadata dictionary).
    """
    is_dask = hasattr(df, 'compute')
    df_encoded = df.copy() if is_dask else df.copy()
    encoding_info = {}

    # Get categorical columns
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            if is_dask:
                unique_count = df[col].nunique().compute()
            else:
                unique_count = df[col].nunique()
            if unique_count <= 20:  # Only encode low-cardinality categoricals
                categorical_cols.append(col)

    if not categorical_cols:
        return df_encoded, encoding_info

    if encoding_method == 'one_hot':
        # One-hot encoding
        for col in categorical_cols:
            if is_dask:
                # For Dask, we need to handle this differently
                dummies = dd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                df_encoded = dd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            else:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

            encoding_info[col] = {
                'method': 'one_hot',
                'original_type': str(df[col].dtype),
                'categories': sorted(df[col].unique()) if not is_dask else sorted(df[col].unique().compute())
            }

    elif encoding_method == 'target_encoding' and target is not None:
        # Target encoding (mean encoding)
        if is_dask:
            target_series = target if hasattr(target, 'compute') else dd.from_pandas(pd.Series(target), npartitions=1)
            target_name = target_series.name if hasattr(target_series, 'name') else '__target__'
            if target_name not in df_encoded.columns:
                df_encoded[target_name] = target_series
        else:
            target_series = pd.Series(target) if not isinstance(target, pd.Series) else target
            target_name = '__target__'
            if target_name not in df_encoded.columns:
                df_encoded[target_name] = target_series.values

        for col in categorical_cols:
            if is_dask:
                # Group by category and compute mean target
                means = df_encoded.groupby(col)[target_name].mean().compute()
                df_encoded[f'{col}_encoded'] = df_encoded[col].map(means)
                df_encoded = df_encoded.drop(columns=[col])
            else:
                # Use cross-validation to avoid overfitting
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                encoded_values = np.zeros(len(df_encoded))

                for train_idx, val_idx in kf.split(df_encoded):
                    train_data = df_encoded.iloc[train_idx]
                    val_data = df_encoded.iloc[val_idx]

                    # Compute means on training data
                    means = train_data.groupby(col)[target_name].mean()
                    # Apply to validation data
                    encoded_values[val_idx] = val_data[col].map(means).fillna(df_encoded[target_name].mean())

                df_encoded[f'{col}_encoded'] = encoded_values
                df_encoded = df_encoded.drop(columns=[col])

        # Remove temporary target column
        if '__target__' in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=['__target__'])

            encoding_info[col] = {
                'method': 'target_encoding',
                'original_type': str(df[col].dtype),
                'categories': sorted(df[col].unique()) if not is_dask else sorted(df[col].unique().compute())
            }

    if progress_callback:
        progress_callback()

    return df_encoded, encoding_info


def prepare_features_for_modeling(df: DataFrame, target: Optional[Union[np.ndarray, pd.Series, str]] = None,
                                 encoding_method: str = 'one_hot', progress_callback=None) -> Tuple[DataFrame, Dict]:
    """
    Prepare features for modeling by applying categorical encoding and basic feature engineering.

    Args:
        df: Input DataFrame.
        target: Target variable.
        encoding_method: Encoding method for categorical variables.
        progress_callback: Optional progress callback.

    Returns:
        Tuple of (prepared DataFrame, metadata dictionary).
    """
    # Apply categorical encoding
    df_encoded, encoding_info = apply_categorical_encoding(df, target, encoding_method, progress_callback)

    # Apply basic feature engineering
    df_final = basic_feature_engineering(df_encoded, progress_callback)

    metadata = {
        'encoding_info': encoding_info,
        'original_shape': df.shape,
        'final_shape': df_final.shape,
        'encoding_method': encoding_method
    }

    return df_final, metadata
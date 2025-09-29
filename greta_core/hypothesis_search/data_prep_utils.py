"""
Data Preparation Utilities Module

Contains utilities for preparing data for genetic algorithms.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import List, Tuple, Dict, Any, Union
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from greta_core.preprocessing import prepare_features_for_modeling

# Type alias for DataFrame
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def _compute_mutual_info(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute mutual information scores for features.

    Args:
        data: Feature matrix.
        target: Target variable.

    Returns:
        Array of mutual information scores.
    """
    from greta_core.statistical_analysis import get_target_type
    target_type = get_target_type(target)
    if target_type == 'categorical':
        return mutual_info_classif(data, target)
    else:
        return mutual_info_regression(data, target)


def _select_top_features(scores: np.ndarray, fraction: float) -> np.ndarray:
    """
    Select top fraction of features based on scores.

    Args:
        scores: Array of scores.
        fraction: Fraction to select (e.g., 0.5 for top 50%).

    Returns:
        Indices of top features.
    """
    num_top = max(1, int(len(scores) * fraction))
    top_indices = np.argsort(scores)[-num_top:]
    return top_indices


def _prepare_data_for_ga(data: Union[np.ndarray, DataFrame], target: Union[np.ndarray, DataFrame, str], sample_frac: float = 1.0,
                        encoding_method: str = 'one_hot') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data and target for genetic algorithm with automatic categorical encoding.

    Args:
        data: Feature matrix.
        target: Target variable.
        sample_frac: Fraction of data to sample.
        encoding_method: Method for encoding categorical variables ('one_hot' or 'target_encoding').

    Returns:
        Tuple of (data_array, target_array, feature_names) as numpy arrays and list.
    """
    # Handle data
    if isinstance(data, np.ndarray):
        data_array = data
        feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        if sample_frac < 1.0:
            n_samples = int(len(data_array) * sample_frac)
            indices = np.random.choice(len(data_array), size=n_samples, replace=False)
            data_array = data_array[indices]
            if isinstance(target, np.ndarray):
                target_array = target[indices]
            else:
                target_array = target
    elif hasattr(data, 'compute'):  # Dask DataFrame
        # For large datasets, consider sampling
        if sample_frac < 1.0:
            data_sample = data.sample(frac=sample_frac)
            target_sample = target.sample(frac=sample_frac) if hasattr(target, 'sample') else target
        else:
            sample_size = min(10000, len(data))  # Sample up to 10k rows for GA
            data_sample = data.sample(frac=sample_size / len(data))
            if hasattr(target, 'sample'):
                target_sample = target.sample(frac=sample_size / len(data))
            else:
                target_sample = target

        # Apply preprocessing and encoding
        data_sample_computed = data_sample.compute()
        target_sample_computed = target_sample.compute() if hasattr(target_sample, 'compute') else target_sample

        # Prepare target for encoding
        if isinstance(target, str):
            target_for_encoding = data_sample_computed[target]
        else:
            target_for_encoding = target_sample_computed

        # Apply feature preparation
        data_prepared, _ = prepare_features_for_modeling(data_sample_computed, target_for_encoding, encoding_method)

        data_array = data_prepared.values
        feature_names = list(data_prepared.columns)
        target_array = target_for_encoding.values if hasattr(target_for_encoding, 'values') else target_for_encoding

    else:  # Pandas DataFrame
        if sample_frac < 1.0:
            data_sample = data.sample(frac=sample_frac)
            if isinstance(target, str):
                target_sample = data_sample[target]
            elif hasattr(target, 'sample'):
                target_sample = target.sample(frac=sample_frac)
            else:
                target_sample = target
        else:
            data_sample = data
            if isinstance(target, str):
                target_sample = data[target]
            else:
                target_sample = target

        # Apply preprocessing and encoding
        if isinstance(target, str):
            target_for_encoding = data_sample[target]
        else:
            target_for_encoding = target_sample

        # Apply feature preparation
        data_prepared, _ = prepare_features_for_modeling(data_sample, target_for_encoding, encoding_method)

        data_array = data_prepared.values
        feature_names = list(data_prepared.columns)
        target_array = target_for_encoding.values if hasattr(target_for_encoding, 'values') else target_for_encoding

    # Handle target for numpy case (already handled above for DataFrames)
    if isinstance(data, np.ndarray):
        if isinstance(target, np.ndarray):
            target_array = target
            if sample_frac < 1.0:
                target_array = target_array[indices]
        elif isinstance(target, str):
            raise ValueError("Cannot use string target with numpy array data")
        else:
            target_array = target

    return data_array, target_array, feature_names
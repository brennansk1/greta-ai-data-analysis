"""
Feature Engineering Module

Contains functions for creating new features and engineering existing ones.
Enhanced with advanced encoding methods, feature generation, and modular design.
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings

# Type alias for DataFrame that can be either Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]


def generate_polynomial_features(df: DataFrame, columns: List[str], degree: int = 2,
                                include_bias: bool = False) -> Tuple[DataFrame, Dict]:
    """Generate polynomial features for specified columns."""
    from sklearn.preprocessing import PolynomialFeatures

    is_dask = hasattr(df, 'compute')
    df_poly = df.copy() if is_dask else df.copy()
    poly_info = {}

    if is_dask:
        # For Dask, compute on partitions
        for col in columns:
            for d in range(2, degree + 1):
                df_poly[f'{col}_pow_{d}'] = df_poly[col] ** d
        poly_info = {'method': 'manual_polynomial', 'degree': degree, 'columns': columns}
    else:
        for col in columns:
            poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
            poly_features = poly.fit_transform(df_poly[[col]])
            feature_names = [f'{col}_pow_{i}' for i in range(1, degree + 1)]

            for i, name in enumerate(feature_names):
                df_poly[name] = poly_features[:, i + 1]  # Skip bias term

        poly_info = {'method': 'polynomial', 'degree': degree, 'columns': columns}

    return df_poly, poly_info


def generate_trigonometric_features(df: DataFrame, columns: List[str]) -> Tuple[DataFrame, Dict]:
    """Generate trigonometric features (sin, cos) for specified columns."""
    df_trig = df.copy() if hasattr(df, 'compute') else df.copy()
    trig_info = {'method': 'trigonometric', 'functions': ['sin', 'cos'], 'columns': columns}

    for col in columns:
        df_trig[f'{col}_sin'] = np.sin(df_trig[col])
        df_trig[f'{col}_cos'] = np.cos(df_trig[col])

    return df_trig, trig_info


def generate_logarithmic_features(df: DataFrame, columns: List[str], base: float = np.e) -> Tuple[DataFrame, Dict]:
    """Generate logarithmic features for specified columns."""
    df_log = df.copy() if hasattr(df, 'compute') else df.copy()
    log_info = {'method': 'logarithmic', 'base': base, 'columns': columns}

    for col in columns:
        # Add small constant to avoid log(0)
        min_val = df_log[col].min() if not hasattr(df_log, 'compute') else df_log[col].min().compute()
        shift = max(1e-10, abs(min_val) + 1e-10) if min_val <= 0 else 0
        if base == np.e:
            df_log[f'{col}_log'] = np.log(df_log[col] + shift)
        elif base == 10:
            df_log[f'{col}_log10'] = np.log10(df_log[col] + shift)
        else:
            df_log[f'{col}_log_{base}'] = np.log(df_log[col] + shift) / np.log(base)

    return df_log, log_info


def generate_binning_features(df: DataFrame, columns: List[str], n_bins: int = 5,
                             strategy: str = 'uniform') -> Tuple[DataFrame, Dict]:
    """Generate binning/discretization features for specified columns."""
    from sklearn.preprocessing import KBinsDiscretizer

    is_dask = hasattr(df, 'compute')
    df_binned = df.copy() if is_dask else df.copy()
    bin_info = {'method': 'binning', 'n_bins': n_bins, 'strategy': strategy, 'columns': columns}

    if is_dask:
        # For Dask, use quantile-based binning
        for col in columns:
            quantiles = df_binned[col].quantile([i/n_bins for i in range(1, n_bins)]).compute()
            df_binned[f'{col}_binned'] = df_binned[col].map_partitions(
                lambda x: pd.cut(x, bins=[-np.inf] + list(quantiles) + [np.inf],
                               labels=False, duplicates='drop'),
                meta=('int64')
            )
    else:
        for col in columns:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
            df_binned[f'{col}_binned'] = discretizer.fit_transform(df_binned[[col]]).flatten()

    return df_binned, bin_info


def generate_interaction_features(df: DataFrame, columns: List[str], max_features: int = 10) -> Tuple[DataFrame, Dict]:
    """Generate interaction features for specified columns."""
    df_interact = df.copy() if hasattr(df, 'compute') else df.copy()
    interaction_info = {'method': 'interactions', 'columns': columns, 'max_features': max_features}

    # Generate pairwise interactions
    interactions_added = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if interactions_added >= max_features:
                break
            col1, col2 = columns[i], columns[j]
            df_interact[f'{col1}_{col2}_interaction'] = df_interact[col1] * df_interact[col2]
            interactions_added += 1

    return df_interact, interaction_info


def generate_statistical_features(df: DataFrame, groupby_cols: List[str], agg_cols: List[str],
                                agg_funcs: List[str] = ['mean', 'std', 'min', 'max']) -> Tuple[DataFrame, Dict]:
    """Generate statistical aggregation features by grouping."""
    is_dask = hasattr(df, 'compute')
    df_stats = df.copy() if is_dask else df.copy()
    stats_info = {'method': 'statistical', 'groupby_cols': groupby_cols, 'agg_cols': agg_cols, 'agg_funcs': agg_funcs}

    if is_dask:
        # For Dask, use map_partitions for groupby operations
        for group_col in groupby_cols:
            for agg_col in agg_cols:
                for func in agg_funcs:
                    if func == 'mean':
                        grouped = df_stats.groupby(group_col)[agg_col].transform('mean')
                    elif func == 'std':
                        grouped = df_stats.groupby(group_col)[agg_col].transform('std')
                    elif func == 'min':
                        grouped = df_stats.groupby(group_col)[agg_col].transform('min')
                    elif func == 'max':
                        grouped = df_stats.groupby(group_col)[agg_col].transform('max')
                    else:
                        continue
                    df_stats[f'{agg_col}_{func}_by_{group_col}'] = grouped
    else:
        for group_col in groupby_cols:
            for agg_col in agg_cols:
                grouped_stats = df_stats.groupby(group_col)[agg_col].agg(agg_funcs)
                for func in agg_funcs:
                    df_stats[f'{agg_col}_{func}_by_{group_col}'] = df_stats[group_col].map(grouped_stats[func])

    return df_stats, stats_info


def basic_feature_engineering(df: DataFrame, numeric_transforms: List[str] = None,
                             max_interactions: int = 5, progress_callback=None) -> Union[DataFrame, Tuple[DataFrame, Dict]]:
    """
    Perform comprehensive feature engineering with multiple transformation types.

    Args:
        df: Input DataFrame (Pandas or Dask).
        numeric_transforms: List of transforms to apply ('polynomial', 'trigonometric', 'logarithmic', 'binning').
        max_interactions: Maximum number of interaction features to generate.
        progress_callback: Optional callback to report progress.

    Returns:
        Tuple of (engineered DataFrame, engineering metadata dictionary).
    """
    df_engineered = df.copy() if hasattr(df, 'compute') else df.copy()
    engineering_info = {}

    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return df_engineered, engineering_info

    # Default transforms if none specified
    if numeric_transforms is None:
        numeric_transforms = ['polynomial']

    # Apply polynomial features (squared terms)
    if 'polynomial' in numeric_transforms:
        df_engineered, poly_info = generate_polynomial_features(df_engineered, numeric_cols[:5], degree=2)
        engineering_info['polynomial'] = poly_info

    # Apply trigonometric features
    if 'trigonometric' in numeric_transforms:
        df_engineered, trig_info = generate_trigonometric_features(df_engineered, numeric_cols[:3])
        engineering_info['trigonometric'] = trig_info

    # Apply logarithmic features
    if 'logarithmic' in numeric_transforms:
        positive_cols = [col for col in numeric_cols if (df_engineered[col] > 0).all()]
        if positive_cols:
            df_engineered, log_info = generate_logarithmic_features(df_engineered, positive_cols[:3])
            engineering_info['logarithmic'] = log_info

    # Apply binning features
    if 'binning' in numeric_transforms:
        df_engineered, bin_info = generate_binning_features(df_engineered, numeric_cols[:3])
        engineering_info['binning'] = bin_info

    # Generate interaction features
    if max_interactions > 0 and len(numeric_cols) >= 2:
        df_engineered, interact_info = generate_interaction_features(df_engineered, numeric_cols[:min(5, len(numeric_cols))], max_interactions)
        engineering_info['interactions'] = interact_info

    if progress_callback:
        progress_callback()

    return df_engineered, engineering_info


def _get_categorical_columns(df: DataFrame, max_cardinality: int = 20) -> List[str]:
    """Get list of categorical columns with cardinality <= max_cardinality."""
    is_dask = hasattr(df, 'compute')
    categorical_cols = []

    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            if is_dask:
                unique_count = df[col].nunique().compute()
            else:
                unique_count = df[col].nunique()
            if unique_count <= max_cardinality:
                categorical_cols.append(col)

    return categorical_cols


def encode_one_hot(df: DataFrame, columns: List[str]) -> Tuple[DataFrame, Dict]:
    """Apply one-hot encoding to specified columns."""
    is_dask = hasattr(df, 'compute')
    df_encoded = df.copy() if is_dask else df.copy()
    encoding_info = {}

    for col in columns:
        if is_dask:
            dummies = dd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            df_encoded = dd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
        else:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

        encoding_info[col] = {
            'method': 'one_hot',
            'original_type': str(df[col].dtype),
            'categories': sorted(df[col].unique()) if not is_dask else sorted(df[col].unique().compute()),
            'new_columns': list(dummies.columns)
        }

    return df_encoded, encoding_info


def encode_label(df: DataFrame, columns: List[str]) -> Tuple[DataFrame, Dict]:
    """Apply label encoding to specified columns."""
    is_dask = hasattr(df, 'compute')
    df_encoded = df.copy() if is_dask else df.copy()
    encoding_info = {}

    for col in columns:
        if is_dask:
            # For Dask, use pandas operations on partitions
            unique_vals = df[col].unique().compute()
            mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
            df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
            df_encoded = df_encoded.drop(columns=[col])
        else:
            encoder = LabelEncoder()
            df_encoded[f'{col}_encoded'] = encoder.fit_transform(df_encoded[col])
            df_encoded = df_encoded.drop(columns=[col])

            encoding_info[col] = {
                'method': 'label',
                'original_type': str(df[col].dtype),
                'categories': list(encoder.classes_),
                'mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            }

    return df_encoded, encoding_info


def encode_ordinal(df: DataFrame, columns: List[str], orderings: Optional[Dict[str, List]] = None) -> Tuple[DataFrame, Dict]:
    """Apply ordinal encoding to specified columns."""
    is_dask = hasattr(df, 'compute')
    df_encoded = df.copy() if is_dask else df.copy()
    encoding_info = {}

    for col in columns:
        if is_dask:
            # For Dask, handle manually
            if orderings and col in orderings:
                ordering = orderings[col]
            else:
                ordering = sorted(df[col].unique().compute())
            mapping = {val: i for i, val in enumerate(ordering)}
            df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
            df_encoded = df_encoded.drop(columns=[col])
        else:
            if orderings and col in orderings:
                encoder = OrdinalEncoder(categories=[orderings[col]])
            else:
                encoder = OrdinalEncoder()
            df_encoded[f'{col}_encoded'] = encoder.fit_transform(df_encoded[[col]])
            df_encoded = df_encoded.drop(columns=[col])

            encoding_info[col] = {
                'method': 'ordinal',
                'original_type': str(df[col].dtype),
                'categories': list(encoder.categories_[0]),
                'mapping': dict(zip(encoder.categories_[0], range(len(encoder.categories_[0]))))
            }

    return df_encoded, encoding_info


def encode_frequency(df: DataFrame, columns: List[str]) -> Tuple[DataFrame, Dict]:
    """Apply frequency encoding to specified columns."""
    is_dask = hasattr(df, 'compute')
    df_encoded = df.copy() if is_dask else df.copy()
    encoding_info = {}

    for col in columns:
        if is_dask:
            freq = df[col].value_counts().compute()
            total = len(df)
            mapping = (freq / total).to_dict()
            df_encoded[f'{col}_freq_encoded'] = df_encoded[col].map(mapping)
            df_encoded = df_encoded.drop(columns=[col])
        else:
            freq = df[col].value_counts(normalize=True)
            df_encoded[f'{col}_freq_encoded'] = df_encoded[col].map(freq)
            df_encoded = df_encoded.drop(columns=[col])

        encoding_info[col] = {
            'method': 'frequency',
            'original_type': str(df[col].dtype),
            'frequencies': freq.to_dict() if not is_dask else (freq / len(df)).to_dict()
        }

    return df_encoded, encoding_info


def encode_target_mean(df: DataFrame, columns: List[str], target: Union[np.ndarray, pd.Series]) -> Tuple[DataFrame, Dict]:
    """Apply target mean encoding to specified columns."""
    is_dask = hasattr(df, 'compute')
    df_encoded = df.copy() if is_dask else df.copy()
    encoding_info = {}

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

    for col in columns:
        if is_dask:
            means = df_encoded.groupby(col)[target_name].mean().compute()
            df_encoded[f'{col}_target_encoded'] = df_encoded[col].map(means)
            df_encoded = df_encoded.drop(columns=[col])
        else:
            # Use cross-validation to avoid overfitting
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            encoded_values = np.zeros(len(df_encoded))

            for train_idx, val_idx in kf.split(df_encoded):
                train_data = df_encoded.iloc[train_idx]
                val_data = df_encoded.iloc[val_idx]

                means = train_data.groupby(col)[target_name].mean()
                encoded_values[val_idx] = val_data[col].map(means).fillna(df_encoded[target_name].mean())

            df_encoded[f'{col}_target_encoded'] = encoded_values
            df_encoded = df_encoded.drop(columns=[col])

        encoding_info[col] = {
            'method': 'target_mean',
            'original_type': str(df[col].dtype),
            'categories': sorted(df[col].unique()) if not is_dask else sorted(df[col].unique().compute())
        }

    # Remove temporary target column
    if '__target__' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['__target__'])

    return df_encoded, encoding_info


def apply_categorical_encoding(df: DataFrame, target: Optional[Union[np.ndarray, pd.Series, str]] = None,
                               encoding_method: str = 'auto', max_cardinality: int = 20,
                               orderings: Optional[Dict[str, List]] = None, progress_callback=None) -> Tuple[DataFrame, Dict]:
    """
    Apply categorical encoding to DataFrame columns with multiple encoding strategies.

    Args:
        df: Input DataFrame (Pandas or Dask).
        target: Target variable for supervised encoding methods.
        encoding_method: Encoding method ('auto', 'one_hot', 'label', 'ordinal', 'frequency', 'target_mean').
        max_cardinality: Maximum cardinality for categorical columns to encode.
        orderings: Custom orderings for ordinal encoding.
        progress_callback: Optional callback to report progress.

    Returns:
        Tuple of (encoded DataFrame, encoding metadata dictionary).
    """
    categorical_cols = _get_categorical_columns(df, max_cardinality)

    if not categorical_cols:
        return df.copy() if hasattr(df, 'compute') else df.copy(), {}

    encoding_info = {}

    if encoding_method == 'auto':
        # Auto-select based on cardinality and target availability
        if target is not None:
            encoding_method = 'target_mean'
        else:
            encoding_method = 'one_hot'

    # Apply selected encoding
    if encoding_method == 'one_hot':
        df_encoded, info = encode_one_hot(df, categorical_cols)
    elif encoding_method == 'label':
        df_encoded, info = encode_label(df, categorical_cols)
    elif encoding_method == 'ordinal':
        df_encoded, info = encode_ordinal(df, categorical_cols, orderings)
    elif encoding_method == 'frequency':
        df_encoded, info = encode_frequency(df, categorical_cols)
    elif encoding_method == 'target_mean' and target is not None:
        df_encoded, info = encode_target_mean(df, categorical_cols, target)
    else:
        warnings.warn(f"Unsupported encoding method: {encoding_method}. Using one_hot.")
        df_encoded, info = encode_one_hot(df, categorical_cols)

    encoding_info.update(info)

    if progress_callback:
        progress_callback()

    return df_encoded, encoding_info


def prepare_features_for_modeling(df: DataFrame, target: Optional[Union[np.ndarray, pd.Series, str]] = None,
                                  encoding_method: str = 'auto', numeric_transforms: List[str] = None,
                                  max_interactions: int = 5, max_cardinality: int = 20,
                                  orderings: Optional[Dict[str, List]] = None,
                                  progress_callback=None) -> Tuple[DataFrame, Dict]:
    """
    Prepare features for modeling by applying categorical encoding and comprehensive feature engineering.

    Args:
        df: Input DataFrame.
        target: Target variable for supervised encoding.
        encoding_method: Encoding method for categorical variables ('auto', 'one_hot', 'label', etc.).
        numeric_transforms: List of numeric transforms to apply.
        max_interactions: Maximum number of interaction features to generate.
        max_cardinality: Maximum cardinality for categorical encoding.
        orderings: Custom orderings for ordinal encoding.
        progress_callback: Optional progress callback.

    Returns:
        Tuple of (prepared DataFrame, comprehensive metadata dictionary).
    """
    import logging
    logger = logging.getLogger(__name__)

    # Input validation
    if df is None or df.empty:
        raise ValueError("Input DataFrame is None or empty")

    if encoding_method not in ['auto', 'one_hot', 'label', 'ordinal', 'frequency', 'target_mean']:
        raise ValueError(f"Unsupported encoding method: {encoding_method}")

    if numeric_transforms is not None:
        valid_transforms = ['polynomial', 'trigonometric', 'logarithmic', 'binning']
        invalid_transforms = [t for t in numeric_transforms if t not in valid_transforms]
        if invalid_transforms:
            raise ValueError(f"Unsupported numeric transforms: {invalid_transforms}")

    original_shape = df.shape
    df_encoded = df.copy() if hasattr(df, 'copy') else pd.DataFrame(df)
    encoding_info = {}
    engineering_info = {}

    try:
        # Apply categorical encoding with error handling
        logger.info(f"Applying categorical encoding with method: {encoding_method}")
        df_encoded, encoding_info = apply_categorical_encoding(
            df_encoded, target, encoding_method, max_cardinality, orderings, progress_callback
        )
        logger.info(f"Categorical encoding completed. Shape: {df_encoded.shape}")

    except Exception as e:
        logger.warning(f"Categorical encoding failed: {e}. Using original data.")
        encoding_info = {'error': str(e), 'fallback': 'no_encoding'}
        df_encoded = df.copy() if hasattr(df, 'copy') else pd.DataFrame(df)

    encoded_shape = df_encoded.shape

    try:
        # Apply comprehensive feature engineering with error handling
        logger.info(f"Applying feature engineering with transforms: {numeric_transforms}")
        df_final, engineering_info = basic_feature_engineering(
            df_encoded, numeric_transforms, max_interactions, progress_callback
        )
        logger.info(f"Feature engineering completed. Shape: {df_final.shape}")

    except Exception as e:
        logger.warning(f"Feature engineering failed: {e}. Using encoded data.")
        engineering_info = {'error': str(e), 'fallback': 'no_engineering'}
        df_final = df_encoded

    final_shape = df_final.shape

    # Ensure we have valid data
    if df_final.empty or df_final.shape[1] == 0:
        logger.warning("Feature engineering resulted in empty dataset. Using original data.")
        df_final = df.copy() if hasattr(df, 'copy') else pd.DataFrame(df)
        final_shape = df_final.shape

    metadata = {
        'encoding_info': encoding_info,
        'engineering_info': engineering_info,
        'original_shape': original_shape,
        'encoded_shape': encoded_shape,
        'final_shape': final_shape,
        'encoding_method': encoding_method,
        'numeric_transforms': numeric_transforms or ['polynomial'],
        'max_interactions': max_interactions,
        'max_cardinality': max_cardinality,
        'success': True
    }

    # Check for errors
    if 'error' in encoding_info or 'error' in engineering_info:
        metadata['success'] = False
        metadata['warnings'] = []
        if 'error' in encoding_info:
            metadata['warnings'].append(f"Encoding error: {encoding_info['error']}")
        if 'error' in engineering_info:
            metadata['warnings'].append(f"Engineering error: {engineering_info['error']}")

    return df_final, metadata


# Store the original function
_basic_feature_engineering_original = basic_feature_engineering


def _basic_feature_engineering_wrapper(*args, **kwargs):
    """Wrapper to maintain backward compatibility."""
    import inspect

    # Check if called with old signature (only df and progress_callback)
    sig = inspect.signature(_basic_feature_engineering_original)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # If only 'df' and optionally 'progress_callback' are provided, return just DataFrame
        if len([k for k in bound.arguments.keys() if k != 'progress_callback' and bound.arguments[k] != sig.parameters[k].default]) <= 1:
            # Legacy version for backward compatibility
            df_engineered, _ = _basic_feature_engineering_original(*args, **kwargs)
            return df_engineered
        else:
            return _basic_feature_engineering_original(*args, **kwargs)
    except TypeError:
        # If binding fails, assume new interface
        return _basic_feature_engineering_original(*args, **kwargs)


# Replace the function with the wrapper for backward compatibility
basic_feature_engineering = _basic_feature_engineering_wrapper
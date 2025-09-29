"""
Automated Feature Generation Module

Contains functions for automatically generating features based on data characteristics.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Optional
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings

# Type alias for DataFrame
DataFrame = Union[pd.DataFrame, np.ndarray]


class AutomatedFeatureGenerator:
    """
    Automated feature generation system that analyzes data and generates relevant features.
    """

    def __init__(self, max_features: int = 50, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.generated_features = []
        self.generation_log = []

    def analyze_data_characteristics(self, X: DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        Analyze data characteristics to determine appropriate feature generation strategies.

        Args:
            X: Feature matrix.
            y: Target variable (optional).

        Returns:
            Dictionary with data characteristics.
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        characteristics = {
            'n_samples': len(X_df),
            'n_features': X_df.shape[1],
            'data_types': X_df.dtypes.value_counts().to_dict(),
            'numeric_columns': X_df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': X_df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'missing_values': X_df.isnull().sum().to_dict(),
            'correlations': None,
            'target_correlations': None
        }

        # Calculate correlations for numeric features
        numeric_cols = characteristics['numeric_columns']
        if len(numeric_cols) > 1:
            corr_matrix = X_df[numeric_cols].corr()
            characteristics['correlations'] = corr_matrix

        # Calculate correlations with target if provided
        if y is not None:
            if isinstance(y, pd.Series):
                y_series = y
            else:
                y_series = pd.Series(y)

            if len(numeric_cols) > 0:
                target_corr = X_df[numeric_cols].corrwith(y_series)
                characteristics['target_correlations'] = target_corr.to_dict()

        return characteristics

    def generate_numeric_features(self, X: DataFrame, characteristics: Dict[str, Any]) -> Tuple[DataFrame, Dict]:
        """
        Generate features for numeric columns.

        Args:
            X: Feature matrix.
            characteristics: Data characteristics from analyze_data_characteristics.

        Returns:
            Tuple of (enhanced DataFrame, generation info).
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        numeric_cols = characteristics['numeric_columns']
        generation_info = {'numeric_features': []}

        features_added = 0

        for col in numeric_cols:
            if features_added >= self.max_features:
                break

            col_data = X_df[col]

            # Skip if too many missing values
            if col_data.isnull().sum() / len(col_data) > 0.5:
                continue

            # Basic statistical transformations
            if features_added < self.max_features:
                # Log transformation (for positive values)
                if (col_data > 0).all():
                    X_df[f'{col}_log'] = np.log(col_data + 1e-10)
                    generation_info['numeric_features'].append(f'{col}_log')
                    features_added += 1

            if features_added < self.max_features:
                # Square root transformation
                if (col_data >= 0).all():
                    X_df[f'{col}_sqrt'] = np.sqrt(col_data)
                    generation_info['numeric_features'].append(f'{col}_sqrt')
                    features_added += 1

            if features_added < self.max_features:
                # Polynomial features (degree 2)
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(col_data.values.reshape(-1, 1))
                for i in range(1, poly_features.shape[1]):
                    X_df[f'{col}_pow_{i+1}'] = poly_features[:, i]
                    generation_info['numeric_features'].append(f'{col}_pow_{i+1}')
                    features_added += 1
                    if features_added >= self.max_features:
                        break

            if features_added < self.max_features:
                # Binning
                try:
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                    X_df[f'{col}_binned'] = discretizer.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                    generation_info['numeric_features'].append(f'{col}_binned')
                    features_added += 1
                except:
                    pass

        return X_df, generation_info

    def generate_interaction_features(self, X: DataFrame, characteristics: Dict[str, Any],
                                    y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[DataFrame, Dict]:
        """
        Generate interaction features between important variables.

        Args:
            X: Feature matrix.
            characteristics: Data characteristics.
            y: Target variable (optional).

        Returns:
            Tuple of (enhanced DataFrame, generation info).
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        numeric_cols = characteristics['numeric_columns']
        generation_info = {'interaction_features': []}

        if len(numeric_cols) < 2:
            return X_df, generation_info

        # Select top correlated features if target is available
        if y is not None and characteristics['target_correlations']:
            target_corr = characteristics['target_correlations']
            # Sort by absolute correlation
            sorted_cols = sorted(numeric_cols, key=lambda x: abs(target_corr.get(x, 0)), reverse=True)
            candidate_cols = sorted_cols[:min(5, len(sorted_cols))]
        else:
            candidate_cols = numeric_cols[:min(5, len(numeric_cols))]

        features_added = 0
        for i in range(len(candidate_cols)):
            for j in range(i + 1, len(candidate_cols)):
                if features_added >= self.max_features:
                    break

                col1, col2 = candidate_cols[i], candidate_cols[j]
                X_df[f'{col1}_{col2}_interaction'] = X_df[col1] * X_df[col2]
                generation_info['interaction_features'].append(f'{col1}_{col2}_interaction')
                features_added += 1

        return X_df, generation_info

    def generate_ratio_features(self, X: DataFrame, characteristics: Dict[str, Any]) -> Tuple[DataFrame, Dict]:
        """
        Generate ratio features between numeric variables.

        Args:
            X: Feature matrix.
            characteristics: Data characteristics.

        Returns:
            Tuple of (enhanced DataFrame, generation info).
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        numeric_cols = characteristics['numeric_columns']
        generation_info = {'ratio_features': []}

        if len(numeric_cols) < 2:
            return X_df, generation_info

        features_added = 0
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                if i == j or features_added >= self.max_features:
                    continue

                col1, col2 = numeric_cols[i], numeric_cols[j]
                col1_data, col2_data = X_df[col1], X_df[col2]

                # Avoid division by zero
                if (col2_data != 0).any():
                    # Add small constant to avoid division by very small numbers
                    denominator = col2_data + np.sign(col2_data) * 1e-10
                    X_df[f'{col1}_div_{col2}'] = col1_data / denominator
                    generation_info['ratio_features'].append(f'{col1}_div_{col2}')
                    features_added += 1

        return X_df, generation_info

    def generate_features(self, X: DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None,
                         feature_types: List[str] = None) -> Tuple[DataFrame, Dict]:
        """
        Main method to generate features automatically.

        Args:
            X: Feature matrix.
            y: Target variable (optional).
            feature_types: Types of features to generate ('numeric', 'interactions', 'ratios').

        Returns:
            Tuple of (enhanced DataFrame, comprehensive generation info).
        """
        if feature_types is None:
            feature_types = ['numeric', 'interactions', 'ratios']

        # Analyze data
        characteristics = self.analyze_data_characteristics(X, y)

        X_enhanced = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)
        generation_info = {
            'characteristics': characteristics,
            'feature_types': feature_types,
            'total_features_generated': 0
        }

        # Generate different types of features
        if 'numeric' in feature_types:
            X_enhanced, numeric_info = self.generate_numeric_features(X_enhanced, characteristics)
            generation_info.update(numeric_info)
            generation_info['total_features_generated'] += len(numeric_info.get('numeric_features', []))

        if 'interactions' in feature_types:
            X_enhanced, interaction_info = self.generate_interaction_features(X_enhanced, characteristics, y)
            generation_info.update(interaction_info)
            generation_info['total_features_generated'] += len(interaction_info.get('interaction_features', []))

        if 'ratios' in feature_types:
            X_enhanced, ratio_info = self.generate_ratio_features(X_enhanced, characteristics)
            generation_info.update(ratio_info)
            generation_info['total_features_generated'] += len(ratio_info.get('ratio_features', []))

        return X_enhanced, generation_info


def generate_features_automated(X: DataFrame, y: Optional[Union[np.ndarray, pd.Series]] = None,
                               max_features: int = 50, feature_types: List[str] = None) -> Tuple[DataFrame, Dict]:
    """
    Convenience function for automated feature generation.

    Args:
        X: Feature matrix.
        y: Target variable (optional).
        max_features: Maximum number of features to generate.
        feature_types: Types of features to generate.

    Returns:
        Tuple of (enhanced DataFrame, generation info).
    """
    generator = AutomatedFeatureGenerator(max_features=max_features)
    return generator.generate_features(X, y, feature_types)
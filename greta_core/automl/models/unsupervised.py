"""
Unsupervised learning model implementations.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from .base import UnsupervisedModel


class KMeansModel(UnsupervisedModel):
    """K-Means clustering model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'n_clusters': 3,
            'init': 'k-means++',
            'n_init': 10,
            'max_iter': 300,
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        self.model = KMeans(**self.params)
        self.model.fit(X)

    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'KMeansModel':
        self.params.update(params)
        return self

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster centers")
        return self.model.cluster_centers_

    def get_inertia(self) -> float:
        """Get sum of squared distances of samples to their closest cluster center."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting inertia")
        return self.model.inertia_


class DBSCANModel(UnsupervisedModel):
    """DBSCAN clustering model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'eps': 0.5,
            'min_samples': 5,
            'metric': 'euclidean',
            'algorithm': 'auto'
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        self.model = DBSCAN(**self.params)
        self.model.fit(X)

    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        # DBSCAN doesn't have a predict method, use fit_predict
        return self.model.fit_predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'DBSCANModel':
        self.params.update(params)
        return self

    def get_core_sample_indices(self) -> np.ndarray:
        """Get indices of core samples."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting core sample indices")
        return self.model.core_sample_indices_

    def get_labels(self) -> np.ndarray:
        """Get cluster labels for each point."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting labels")
        return self.model.labels_


class GaussianMixtureModel(UnsupervisedModel):
    """Gaussian Mixture Model for clustering."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'n_components': 3,
            'covariance_type': 'full',
            'max_iter': 100,
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        self.model = GaussianMixture(**self.params)
        self.model.fit(X)

    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict posterior probability of each component given the data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'GaussianMixtureModel':
        self.params.update(params)
        return self

    def get_means(self) -> np.ndarray:
        """Get means of each mixture component."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting means")
        return self.model.means_

    def get_covariances(self) -> np.ndarray:
        """Get covariances of each mixture component."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting covariances")
        return self.model.covariances_

    def get_weights(self) -> np.ndarray:
        """Get weights of each mixture component."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting weights")
        return self.model.weights_


class PCAModel(UnsupervisedModel):
    """Principal Component Analysis for dimensionality reduction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'n_components': None,  # Keep all components
            'whiten': False,
            'random_state': 42
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        self.model = PCA(**self.params)
        self.model.fit(X)

    def _predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        # For PCA, predict returns the transformed data
        return self.transform(X)

    def _transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming data")
        return self.model.transform(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'PCAModel':
        self.params.update(params)
        return self

    def get_components(self) -> np.ndarray:
        """Get principal components."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting components")
        return self.model.components_

    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance for each component."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting explained variance")
        return self.model.explained_variance_

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting explained variance ratio")
        return self.model.explained_variance_ratio_

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data back to its original space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse transforming")
        return self.model.inverse_transform(X)
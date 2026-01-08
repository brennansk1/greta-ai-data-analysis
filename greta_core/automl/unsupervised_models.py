"""
Unsupervised Learning Models Module

This module contains implementations of unsupervised learning algorithms
including clustering algorithms (K-Means, DBSCAN) for automated segmentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class BaseUnsupervisedModel(ABC):
    """Abstract base class for unsupervised learning models."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.scaler = StandardScaler()

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def evaluate_clustering(self, data: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality using various metrics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        try:
            metrics = {}

            # Silhouette Score
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(data, labels)

            # Calinski-Harabasz Index
            if len(np.unique(labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)

            # Davies-Bouldin Index
            if len(np.unique(labels)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)

            # Number of clusters
            metrics['n_clusters'] = len(np.unique(labels))

            return metrics
        except Exception as e:
            logger.error(f"Failed to evaluate clustering: {e}")
            return {}


class KMeansModel(BaseUnsupervisedModel):
    """K-Means clustering model."""

    def __init__(self, n_clusters: int = 3, init: str = 'k-means++',
                 n_init: int = 10, max_iter: int = 300,
                 random_state: Optional[int] = None):
        super().__init__('kmeans')
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the K-Means model."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)

            # Initialize and fit the model
            self.model = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self.model.fit(scaled_data)
            self.is_fitted = True
            logger.info(f"K-Means model fitted with {self.n_clusters} clusters")
        except Exception as e:
            logger.error(f"Failed to fit K-Means model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            scaled_data = self.scaler.transform(data)
            return self.model.predict(scaled_data)
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise

    def get_cluster_centers(self) -> np.ndarray:
        """Get the cluster centers."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster centers")
        return self.scaler.inverse_transform(self.model.cluster_centers_)


class DBSCANModel(BaseUnsupervisedModel):
    """DBSCAN clustering model."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = 'euclidean', algorithm: str = 'auto'):
        super().__init__('dbscan')
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the DBSCAN model."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)

            # Initialize and fit the model
            self.model = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric=self.metric,
                algorithm=self.algorithm
            )
            self.model.fit(scaled_data)
            self.is_fitted = True

            # Count clusters (excluding noise labeled as -1)
            n_clusters = len(set(self.model.labels_)) - (1 if -1 in self.model.labels_ else 0)
            logger.info(f"DBSCAN model fitted with {n_clusters} clusters")
        except Exception as e:
            logger.error(f"Failed to fit DBSCAN model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            scaled_data = self.scaler.transform(data)
            return self.model.fit_predict(scaled_data)
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise


class GaussianMixtureModel(BaseUnsupervisedModel):
    """Gaussian Mixture Model for clustering."""

    def __init__(self, n_components: int = 3, covariance_type: str = 'full',
                 max_iter: int = 100, random_state: Optional[int] = None):
        super().__init__('gaussian_mixture')
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the Gaussian Mixture model."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)

            # Initialize and fit the model
            self.model = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self.model.fit(scaled_data)
            self.is_fitted = True
            logger.info(f"Gaussian Mixture model fitted with {self.n_components} components")
        except Exception as e:
            logger.error(f"Failed to fit Gaussian Mixture model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            scaled_data = self.scaler.transform(data)
            return self.model.predict(scaled_data)
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict posterior probabilities for each component."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            scaled_data = self.scaler.transform(data)
            return self.model.predict_proba(scaled_data)
        except Exception as e:
            logger.error(f"Failed to predict probabilities: {e}")
            raise


class PCAModel(BaseUnsupervisedModel):
    """Principal Component Analysis for dimensionality reduction."""

    def __init__(self, n_components: Optional[Union[int, float]] = None,
                 whiten: bool = False, random_state: Optional[int] = None):
        super().__init__('pca')
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the PCA model."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)

            # Initialize and fit the model
            self.model = PCA(
                n_components=self.n_components,
                whiten=self.whiten,
                random_state=self.random_state
            )
            self.model.fit(scaled_data)
            self.is_fitted = True

            explained_variance_ratio = self.model.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            logger.info(f"PCA fitted. Explained variance: {cumulative_variance[-1]:.3f}")
        except Exception as e:
            logger.error(f"Failed to fit PCA model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data to principal components."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        try:
            scaled_data = self.scaler.transform(data)
            return self.model.transform(scaled_data)
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            raise

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting explained variance")
        return self.model.explained_variance_ratio_

    def get_cumulative_explained_variance(self) -> np.ndarray:
        """Get the cumulative explained variance."""
        return np.cumsum(self.get_explained_variance_ratio())


# Unsupervised model registry
UNSUPERVISED_MODEL_REGISTRY = {
    'kmeans': KMeansModel,
    'dbscan': DBSCANModel,
    'gaussian_mixture': GaussianMixtureModel,
    'pca': PCAModel,
}


def get_unsupervised_model(model_name: str, **kwargs) -> BaseUnsupervisedModel:
    """Factory function to create an unsupervised model instance."""
    if model_name not in UNSUPERVISED_MODEL_REGISTRY:
        raise ValueError(f"Unsupervised model '{model_name}' not found in registry")
    return UNSUPERVISED_MODEL_REGISTRY[model_name](**kwargs)
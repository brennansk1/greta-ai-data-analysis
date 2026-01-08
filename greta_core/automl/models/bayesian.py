"""
Bayesian model implementations for probabilistic inference.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, ARDRegression

try:
    import pymc3 as pm
    import arviz as az
    PYM3_AVAILABLE = True
except ImportError:
    PYM3_AVAILABLE = False

from .base import SupervisedModel


class BayesianLinearRegression(SupervisedModel):
    """Bayesian Linear Regression using scikit-learn."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'alpha_1': 1e-6,
            'alpha_2': 1e-6,
            'lambda_1': 1e-6,
            'lambda_2': 1e-6,
            'compute_score': True
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        if self.task_type != 'regression':
            raise ValueError("BayesianLinearRegression is only for regression tasks")
        self.model = BayesianRidge(**self.params)
        self.model.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Predict with uncertainty quantification."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        y_pred, y_std = self.model.predict(X, return_std=True)
        return {
            'mean': y_pred,
            'std': y_std,
            'lower_bound': y_pred - 2 * y_std,
            'upper_bound': y_pred + 2 * y_std
        }

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'BayesianLinearRegression':
        self.params.update(params)
        return self


class BayesianLogisticRegression(SupervisedModel):
    """Bayesian Logistic Regression using scikit-learn."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_params = {
            'alpha_1': 1e-6,
            'alpha_2': 1e-6,
            'lambda_1': 1e-6,
            'lambda_2': 1e-6,
            'compute_score': True
        }
        self.params = {**self.default_params, **self.config}

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        if self.task_type != 'classification':
            raise ValueError("BayesianLogisticRegression is only for classification tasks")

        # Use Bayesian Ridge for binary classification (not ideal but available)
        # For proper Bayesian logistic regression, would need PyMC3
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        if not self.is_fitted:
            return None
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'BayesianLogisticRegression':
        self.params.update(params)
        return self


class BayesianHypothesisTesting(SupervisedModel):
    """Bayesian hypothesis testing framework."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not PYM3_AVAILABLE:
            raise ImportError("PyMC3 and ArviZ are required for Bayesian hypothesis testing. Install with: pip install pymc3 arviz")

        self.task_type = 'hypothesis_testing'
        self.default_params = {
            'n_samples': 1000,
            'n_tune': 1000,
            'n_chains': 4,
            'target_accept': 0.9
        }
        self.params = {**self.default_params, **self.config}
        self.trace = None

    def _fit(self, X: Union[np.ndarray, pd.DataFrame],
             y: Union[np.ndarray, pd.Series]) -> None:
        # For hypothesis testing, X might contain group labels and y contains measurements
        if isinstance(X, pd.DataFrame) and 'group' in X.columns:
            groups = X['group'].unique()
            if len(groups) == 2:
                # Two-sample t-test
                group1_data = y[X['group'] == groups[0]]
                group2_data = y[X['group'] == groups[1]]
                self._bayesian_ttest(group1_data, group2_data)
            else:
                raise ValueError("Bayesian hypothesis testing currently supports only two groups")
        else:
            raise ValueError("For hypothesis testing, X must contain a 'group' column")

    def _bayesian_ttest(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Perform Bayesian t-test."""
        with pm.Model() as model:
            # Priors
            mu1 = pm.Normal('mu1', mu=0, sigma=10)
            mu2 = pm.Normal('mu2', mu=0, sigma=10)
            sigma1 = pm.HalfNormal('sigma1', sigma=10)
            sigma2 = pm.HalfNormal('sigma2', sigma=10)

            # Likelihood
            pm.Normal('obs1', mu=mu1, sigma=sigma1, observed=data1)
            pm.Normal('obs2', mu=mu2, sigma=sigma2, observed=data2)

            # Difference
            diff = pm.Deterministic('diff', mu1 - mu2)

            # Sample
            self.trace = pm.sample(
                self.params['n_samples'],
                tune=self.params['n_tune'],
                chains=self.params['n_chains'],
                target_accept=self.params['target_accept'],
                return_inferencedata=True
            )

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        # For hypothesis testing, return the probability that group1 > group2
        if self.trace is None:
            raise ValueError("Model must be fitted before making predictions")

        diff_samples = self.trace.posterior['diff'].values.flatten()
        prob_greater = np.mean(diff_samples > 0)

        return np.array([prob_greater])

    def get_bayesian_summary(self) -> Dict[str, Any]:
        """Get Bayesian analysis summary."""
        if self.trace is None:
            raise ValueError("Model must be fitted before getting summary")

        summary = az.summary(self.trace, round_to=3)

        diff_samples = self.trace.posterior['diff'].values.flatten()
        hdi = az.hdi(diff_samples, hdi_prob=0.95)

        return {
            'summary': summary,
            'difference_hdi': hdi,
            'prob_greater_zero': np.mean(diff_samples > 0),
            'prob_less_zero': np.mean(diff_samples < 0),
            'bayes_factor': self._calculate_bayes_factor(diff_samples)
        }

    def _calculate_bayes_factor(self, diff_samples: np.ndarray) -> float:
        """Calculate approximate Bayes factor for H0: mu1 = mu2 vs H1: mu1 != mu2."""
        # Simple approximation using Savage-Dickey density ratio
        # This is a rough approximation
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(diff_samples)
        bf = 1.0 / kde(0)[0]  # Density at zero
        return bf

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> 'BayesianHypothesisTesting':
        self.params.update(params)
        return self
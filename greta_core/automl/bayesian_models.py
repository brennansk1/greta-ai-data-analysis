"""
Bayesian Inference Models Module

This module contains implementations of Bayesian statistical models
including Bayesian regression, Bayesian hypothesis testing, and
Bayesian inference for quantifying uncertainty distributions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta, norm, t, f, chi2
import pymc3 as pm
import arviz as az
import logging

logger = logging.getLogger(__name__)


class BaseBayesianModel(ABC):
    """Abstract base class for Bayesian models."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.trace = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the Bayesian model to the data."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty quantification."""
        pass

    def get_summary(self) -> pd.DataFrame:
        """Get model parameter summary."""
        if not self.is_fitted or self.trace is None:
            raise ValueError("Model must be fitted before getting summary")
        return az.summary(self.trace)

    def plot_trace(self) -> None:
        """Plot MCMC trace plots."""
        if not self.is_fitted or self.trace is None:
            raise ValueError("Model must be fitted before plotting")
        az.plot_trace(self.trace)


class BayesianLinearRegression(BaseBayesianModel):
    """Bayesian Linear Regression model."""

    def __init__(self, n_draws: int = 1000, n_tune: int = 1000,
                 target_accept: float = 0.9, random_seed: Optional[int] = None):
        super().__init__('bayesian_linear_regression')
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.target_accept = target_accept
        self.random_seed = random_seed

    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> None:
        """Fit the Bayesian linear regression model."""
        try:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            # Prepare data
            X = data.drop(columns=[target_column]).values
            y = data[target_column].values
            n_features = X.shape[1]

            # Define the model
            with pm.Model() as self.model:
                # Priors
                intercept = pm.Normal('intercept', mu=0, sigma=10)
                beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
                sigma = pm.HalfNormal('sigma', sigma=1)

                # Expected value
                mu = intercept + pm.math.dot(X, beta)

                # Likelihood
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

                # Sample from posterior
                self.trace = pm.sample(
                    draws=self.n_draws,
                    tune=self.n_tune,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    return_inferencedata=True
                )

            self.is_fitted = True
            logger.info("Bayesian Linear Regression model fitted")
        except Exception as e:
            logger.error(f"Failed to fit Bayesian Linear Regression: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            X_new = data.values

            with self.model:
                # Generate posterior predictive samples
                ppc = pm.sample_posterior_predictive(self.trace, var_names=['y_obs'])

            # Get mean predictions and uncertainty intervals
            y_pred_mean = ppc['y_obs'].mean(axis=0)
            y_pred_std = ppc['y_obs'].std(axis=0)

            return y_pred_mean, y_pred_std
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise


class BayesianLogisticRegression(BaseBayesianModel):
    """Bayesian Logistic Regression model."""

    def __init__(self, n_draws: int = 1000, n_tune: int = 1000,
                 target_accept: float = 0.9, random_seed: Optional[int] = None):
        super().__init__('bayesian_logistic_regression')
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.target_accept = target_accept
        self.random_seed = random_seed

    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> None:
        """Fit the Bayesian logistic regression model."""
        try:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            # Prepare data
            X = data.drop(columns=[target_column]).values
            y = data[target_column].values
            n_features = X.shape[1]

            # Define the model
            with pm.Model() as self.model:
                # Priors
                intercept = pm.Normal('intercept', mu=0, sigma=10)
                beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)

                # Expected value
                logit_p = intercept + pm.math.dot(X, beta)

                # Likelihood
                y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)

                # Sample from posterior
                self.trace = pm.sample(
                    draws=self.n_draws,
                    tune=self.n_tune,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    return_inferencedata=True
                )

            self.is_fitted = True
            logger.info("Bayesian Logistic Regression model fitted")
        except Exception as e:
            logger.error(f"Failed to fit Bayesian Logistic Regression: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        try:
            X_new = data.values

            with self.model:
                # Generate posterior predictive samples
                ppc = pm.sample_posterior_predictive(self.trace, var_names=['y_obs'])

            # Get mean predictions and uncertainty intervals
            y_pred_prob = ppc['y_obs'].mean(axis=0)
            y_pred_std = ppc['y_obs'].std(axis=0)

            return y_pred_prob, y_pred_std
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise


class BayesianHypothesisTesting:
    """Bayesian hypothesis testing framework."""

    def __init__(self, prior_type: str = 'normal', effect_size_prior: Optional[Dict] = None):
        self.prior_type = prior_type
        self.effect_size_prior = effect_size_prior or {'mu': 0, 'sigma': 1}
        self.results = {}

    def test_mean_difference(self, group1: np.ndarray, group2: np.ndarray,
                           alpha: float = 0.05) -> Dict[str, Any]:
        """Bayesian test for difference in means."""
        try:
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

            # Pooled variance
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))

            # Effect size
            effect_size = (mean1 - mean2) / np.sqrt(pooled_var)

            # Bayesian analysis
            with pm.Model() as model:
                # Prior on effect size
                if self.prior_type == 'normal':
                    delta = pm.Normal('delta', mu=self.effect_size_prior['mu'],
                                    sigma=self.effect_size_prior['sigma'])
                elif self.prior_type == 'cauchy':
                    delta = pm.Cauchy('delta', alpha=0, beta=1)
                else:
                    delta = pm.Normal('delta', mu=0, sigma=1)

                # Likelihood
                obs1 = pm.Normal('obs1', mu=delta/2, sigma=1, observed=(group1 - mean1)/np.sqrt(var1))
                obs2 = pm.Normal('obs2', mu=-delta/2, sigma=1, observed=(group2 - mean2)/np.sqrt(var2))

                # Sample
                trace = pm.sample(2000, tune=1000, return_inferencedata=True)

            # Calculate Bayes factor and posterior probabilities
            posterior_delta = trace.posterior['delta'].values.flatten()
            bf_10 = self._calculate_bayes_factor(posterior_delta, 0)

            # Credible interval
            hdi = az.hdi(trace, var_names=['delta'])

            result = {
                'test_type': 'mean_difference',
                'effect_size': effect_size,
                'bayes_factor_10': bf_10,
                'posterior_mean': np.mean(posterior_delta),
                'posterior_std': np.std(posterior_delta),
                'credible_interval': hdi['delta'].values,
                'prob_delta_gt_0': np.mean(posterior_delta > 0),
                'prob_delta_lt_0': np.mean(posterior_delta < 0)
            }

            self.results['mean_difference'] = result
            return result
        except Exception as e:
            logger.error(f"Failed to perform Bayesian hypothesis test: {e}")
            raise

    def test_proportion_difference(self, success1: int, n1: int, success2: int, n2: int) -> Dict[str, Any]:
        """Bayesian test for difference in proportions."""
        try:
            # Beta priors
            alpha_prior, beta_prior = 1, 1

            with pm.Model() as model:
                # Priors
                p1 = pm.Beta('p1', alpha=alpha_prior, beta=beta_prior)
                p2 = pm.Beta('p2', alpha=alpha_prior, beta=beta_prior)

                # Likelihood
                obs1 = pm.Binomial('obs1', n=n1, p=p1, observed=success1)
                obs2 = pm.Binomial('obs2', n=n2, p=p2, observed=success2)

                # Difference
                delta = pm.Deterministic('delta', p1 - p2)

                # Sample
                trace = pm.sample(2000, tune=1000, return_inferencedata=True)

            posterior_delta = trace.posterior['delta'].values.flatten()
            bf_10 = self._calculate_bayes_factor(posterior_delta, 0)

            hdi = az.hdi(trace, var_names=['delta'])

            result = {
                'test_type': 'proportion_difference',
                'bayes_factor_10': bf_10,
                'posterior_mean': np.mean(posterior_delta),
                'posterior_std': np.std(posterior_delta),
                'credible_interval': hdi['delta'].values,
                'prob_delta_gt_0': np.mean(posterior_delta > 0),
                'prob_delta_lt_0': np.mean(posterior_delta < 0)
            }

            self.results['proportion_difference'] = result
            return result
        except Exception as e:
            logger.error(f"Failed to perform proportion difference test: {e}")
            raise

    def _calculate_bayes_factor(self, posterior_samples: np.ndarray, null_value: float = 0) -> float:
        """Calculate Bayes factor from posterior samples."""
        try:
            # Simple approximation using Savage-Dickey density ratio
            from scipy.stats import gaussian_kde

            # Fit kernel density estimate
            kde = gaussian_kde(posterior_samples)

            # Evaluate density at null value
            posterior_density_null = kde(null_value)[0]

            # Prior density at null value (assuming normal prior)
            prior_density_null = stats.norm.pdf(null_value,
                                              self.effect_size_prior['mu'],
                                              self.effect_size_prior['sigma'])

            # Bayes factor (BF_10)
            bf_10 = prior_density_null / posterior_density_null

            return bf_10
        except Exception as e:
            logger.warning(f"Failed to calculate Bayes factor: {e}")
            return np.nan

    def get_uncertainty_distribution(self, parameter_name: str) -> Dict[str, Any]:
        """Get uncertainty distribution for a parameter."""
        if parameter_name not in self.results:
            raise ValueError(f"Parameter '{parameter_name}' not found in results")

        result = self.results[parameter_name]
        return {
            'mean': result.get('posterior_mean'),
            'std': result.get('posterior_std'),
            'credible_interval': result.get('credible_interval'),
            'prob_positive': result.get('prob_delta_gt_0'),
            'prob_negative': result.get('prob_delta_lt_0')
        }


# Bayesian model registry
BAYESIAN_MODEL_REGISTRY = {
    'bayesian_linear_regression': BayesianLinearRegression,
    'bayesian_logistic_regression': BayesianLogisticRegression,
    'bayesian_hypothesis_testing': BayesianHypothesisTesting,
}


def get_bayesian_model(model_name: str, **kwargs) -> Union[BaseBayesianModel, BayesianHypothesisTesting]:
    """Factory function to create a Bayesian model instance."""
    if model_name not in BAYESIAN_MODEL_REGISTRY:
        raise ValueError(f"Bayesian model '{model_name}' not found in registry")
    return BAYESIAN_MODEL_REGISTRY[model_name](**kwargs)
"""
Hypothesis Evaluation Module

Contains functions for evaluating hypothesis significance, effect size, coverage, and parsimony.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score


def get_target_type(y, threshold=3):
    unique = np.unique(y)
    if len(unique) <= threshold:
        return 'categorical'
    else:
        return 'continuous'


def detect_feature_types_from_matrix(X: np.ndarray, feature_names: list[str] = None) -> dict[str, str]:
    """
    Detect feature types from a numeric matrix (after encoding).

    Args:
        X: Feature matrix (numeric after encoding).
        feature_names: Optional list of feature names.

    Returns:
        Dictionary mapping feature indices/names to types.
    """
    feature_types = {}

    for i in range(X.shape[1]):
        col_data = X[:, i]
        unique_vals = np.unique(col_data)

        # Check if it's binary (likely one-hot encoded categorical)
        if len(unique_vals) == 2 and set(unique_vals) == {0.0, 1.0}:
            feature_types[i if feature_names is None else feature_names[i]] = 'binary_encoded'
        # Check if it's low cardinality (likely target encoded categorical)
        elif len(unique_vals) <= 10 and len(unique_vals) / len(col_data) < 0.1:
            feature_types[i if feature_names is None else feature_names[i]] = 'categorical_encoded'
        else:
            feature_types[i if feature_names is None else feature_names[i]] = 'numeric'

    return feature_types


def calculate_significance(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate statistical significance for a hypothesis.

    For categorical targets: uses t-test/ANOVA for single feature, logistic accuracy for multiple.
    For continuous targets: uses correlation/regression F-test.

    Args:
        X: Feature matrix.
        y: Target variable.

    Returns:
        Significance score (1 - p_value or accuracy, higher is better).
    """
    # Handle edge cases with insufficient data
    if len(y) < 2:
        return 0.5  # Neutral significance for insufficient data

    target_type = get_target_type(y)

    if target_type == 'categorical':
        unique = np.unique(y)
        if len(unique) == 2:
            if X.shape[1] == 1:
                groups = [X[y == val].flatten() for val in unique]
                if all(len(g) >= 1 for g in groups):
                    try:
                        _, p_value = stats.ttest_ind(*groups)
                        return 1 - p_value
                    except:
                        return 0.5
                else:
                    return 0.5
            else:
                if len(y) >= 2:
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X, y)
                    return model.score(X, y)
                else:
                    return 0.5
        else:
            if X.shape[1] == 1:
                groups = [X[y == val].flatten() for val in unique]
                if all(len(g) >= 1 for g in groups) and len(groups) >= 2:
                    try:
                        _, p_value = stats.f_oneway(*groups)
                        return 1 - p_value
                    except:
                        return 0.5
                else:
                    return 0.5
            else:
                if len(y) >= len(unique):
                    model = LogisticRegression(max_iter=1000, multi_class='ovr')
                    model.fit(X, y)
                    return model.score(X, y)
                else:
                    return 0.5
    else:
        if X.shape[1] == 1:
            if len(y) >= 2:
                try:
                    corr, p_value = stats.pearsonr(X.flatten(), y)
                    return 1 - p_value
                except:
                    return 0.5
            else:
                return 0.5
        else:
            if len(y) > X.shape[1] + 1:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_res > 0 and ss_tot > 0:
                    f_stat = ((ss_tot - ss_res) / X.shape[1]) / (ss_res / (len(y) - X.shape[1] - 1))
                    p_value = 1 - stats.f.cdf(f_stat, X.shape[1], len(y) - X.shape[1] - 1)
                    return 1 - p_value
                else:
                    return 0.5
            else:
                return 0.5


def calculate_effect_size(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate effect size for a hypothesis.

    For categorical: Cohen's d/eta squared for single, accuracy for multiple.
    For continuous: correlation/sqrt(R2).

    Args:
        X: Feature matrix.
        y: Target variable.

    Returns:
        Effect size.
    """
    target_type = get_target_type(y)

    if target_type == 'categorical':
        unique = np.unique(y)
        if len(unique) == 2:
            if X.shape[1] == 1:
                groups = [X[y == val].flatten() for val in unique]
                mean1, mean2 = np.mean(groups[0]), np.mean(groups[1])
                std1, std2 = np.std(groups[0], ddof=1), np.std(groups[1], ddof=1)
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                return d
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                return model.score(X, y)
        else:
            if X.shape[1] == 1:
                groups = [X[y == val].flatten() for val in unique]
                all_data = np.concatenate(groups)
                ss_between = sum(len(g) * (np.mean(g) - np.mean(all_data))**2 for g in groups)
                ss_total = sum((x - np.mean(all_data))**2 for x in all_data)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                return eta_squared
            else:
                model = LogisticRegression(max_iter=1000, multi_class='ovr')
                model.fit(X, y)
                return model.score(X, y)
    else:
        if X.shape[1] == 1:
            corr, _ = stats.pearsonr(X.flatten(), y)
            return abs(corr)
        else:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            return np.sqrt(r2)


def calculate_coverage(X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate coverage (proportion of variance explained or accuracy).

    Args:
        X: Feature matrix.
        y: Target variable.

    Returns:
        Coverage score (R-squared or accuracy).
    """
    target_type = get_target_type(y)

    if target_type == 'categorical':
        if len(np.unique(y)) == 2:
            model = LogisticRegression(max_iter=1000)
        else:
            model = LogisticRegression(max_iter=1000, multi_class='ovr')
        model.fit(X, y)
        return model.score(X, y)
    else:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return r2_score(y, y_pred)


def calculate_parsimony(num_selected: int, total_features: int) -> float:
    """
    Calculate parsimony penalty.

    Args:
        num_selected: Number of selected features.
        total_features: Total number of features.

    Returns:
        Parsimony penalty (higher when more features selected).
    """
    if total_features == 0:
        return 0.0
    return num_selected / total_features


def perform_statistical_test(X: np.ndarray, y: np.ndarray, test_type: str = 'auto') -> dict[str, float]:
    """
    Perform statistical test on hypothesis.

    Args:
        X: Feature matrix.
        y: Target variable.
        test_type: Type of test ('t_test', 'anova', 'mann_whitney', 'kruskal_wallis', 'permutation', 'auto').

    Returns:
        Dictionary with test results.
    """
    from .parametric_tests import perform_t_test, perform_anova
    from .nonparametric_tests import perform_mann_whitney, perform_kruskal_wallis, perform_permutation_test

    results = {}

    if test_type == 'auto':
        target_type = get_target_type(y)
        if target_type == 'categorical':
            unique = np.unique(y)
            if len(unique) == 2:
                test_type = 't_test'
            else:
                test_type = 'anova'
        else:
            test_type = 'regression'

    if test_type == 't_test' and len(np.unique(y)) == 2:
        # Split data by target
        unique_vals = np.unique(y)
        group1 = X[y == unique_vals[0]]
        group2 = X[y == unique_vals[1]]
        if X.shape[1] == 1:
            t_stat, p_value = perform_t_test(group1.flatten(), group2.flatten())
            results = {'test': 't_test', 't_stat': t_stat, 'p_value': p_value}
        else:
            # Multiple features, use logistic regression for binary classification
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            results = {'test': 'logistic', 'accuracy': model.score(X, y)}
    elif test_type == 'mann_whitney' and len(np.unique(y)) == 2:
        # Split data by target
        unique_vals = np.unique(y)
        group1 = X[y == unique_vals[0]]
        group2 = X[y == unique_vals[1]]
        if X.shape[1] == 1:
            u_stat, p_value = perform_mann_whitney(group1.flatten(), group2.flatten())
            results = {'test': 'mann_whitney', 'u_stat': u_stat, 'p_value': p_value}
        else:
            # Multiple features, use logistic regression for binary classification
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            results = {'test': 'logistic', 'accuracy': model.score(X, y)}
    elif test_type == 'permutation' and len(np.unique(y)) == 2:
        # Split data by target
        unique_vals = np.unique(y)
        group1 = X[y == unique_vals[0]]
        group2 = X[y == unique_vals[1]]
        if X.shape[1] == 1:
            u_stat, p_value = perform_permutation_test(group1.flatten(), group2.flatten())
            results = {'test': 'permutation', 'u_stat': u_stat, 'p_value': p_value}
        else:
            # Multiple features, use logistic regression for binary classification
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            results = {'test': 'logistic', 'accuracy': model.score(X, y)}
    elif test_type == 'anova':
        # For ANOVA, need to group by target
        groups = [X[y == val] for val in np.unique(y)]
        if all(g.shape[1] == 1 for g in groups):
            f_stat, p_value = perform_anova(*[g.flatten() for g in groups])
            results = {'test': 'anova', 'f_stat': f_stat, 'p_value': p_value}
        else:
            # Multiple features, use logistic
            model = LogisticRegression(max_iter=1000, multi_class='ovr')
            model.fit(X, y)
            results = {'test': 'logistic', 'accuracy': model.score(X, y)}
    elif test_type == 'kruskal_wallis':
        # For Kruskal-Wallis, need to group by target
        groups = [X[y == val] for val in np.unique(y)]
        if all(g.shape[1] == 1 for g in groups):
            h_stat, p_value = perform_kruskal_wallis(*[g.flatten() for g in groups])
            results = {'test': 'kruskal_wallis', 'h_stat': h_stat, 'p_value': p_value}
        else:
            # Multiple features, use logistic
            model = LogisticRegression(max_iter=1000, multi_class='ovr')
            model.fit(X, y)
            results = {'test': 'logistic', 'accuracy': model.score(X, y)}
    elif test_type == 'regression':
        from ..regression_analysis import perform_multiple_linear_regression
        reg_results = perform_multiple_linear_regression(X, y)
        results = {'test': 'regression', 'r_squared': reg_results['adj_r_squared'], 'p_value': reg_results['f_p_value']}

    return results
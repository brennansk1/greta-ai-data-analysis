"""
Nonparametric Tests Module

Contains functions for nonparametric statistical tests (Mann-Whitney, Kruskal-Wallis, permutation test).
"""

import numpy as np
from scipy import stats


def perform_mann_whitney(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float]:
    """
    Perform Mann-Whitney U test between two groups (non-parametric).

    Args:
        group1: First group data.
        group2: Second group data.

    Returns:
        Tuple of (U-statistic, p-value).
    """
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("Groups must have at least one element")
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return u_stat, p_value


def perform_kruskal_wallis(*groups: np.ndarray) -> tuple[float, float]:
    """
    Perform Kruskal-Wallis H test on multiple groups (non-parametric).

    Args:
        *groups: Variable number of group arrays.

    Returns:
        Tuple of (H-statistic, p-value).
    """
    try:
        h_stat, p_value = stats.kruskal(*groups)
        if np.isnan(h_stat):
            h_stat = 0.0
            p_value = 1.0
        return h_stat, p_value
    except:
        return 0.0, 1.0


def perform_permutation_test(group1: np.ndarray, group2: np.ndarray, n_permutations: int = 1000) -> tuple[float, float]:
    """
    Perform permutation test for two groups to estimate p-value robustness.

    Uses Mann-Whitney U as the test statistic.

    Args:
        group1: First group data.
        group2: Second group data.
        n_permutations: Number of permutations.

    Returns:
        Tuple of (observed_U, permutation_p_value).
    """
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("Groups must have at least one element")

    combined = np.concatenate([group1, group2])
    n1, n2 = len(group1), len(group2)

    # Observed statistic
    observed_u, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Permutation
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        perm_u, _ = stats.mannwhitneyu(perm_group1, perm_group2, alternative='two-sided')
        if perm_u >= observed_u:
            count += 1

    p_value = count / n_permutations
    return observed_u, p_value
"""
Parametric Tests Module

Contains functions for parametric statistical tests (t-test, ANOVA).
"""

import numpy as np
from scipy import stats


def perform_t_test(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float]:
    """
    Perform independent t-test between two groups.

    Args:
        group1: First group data.
        group2: Second group data.

    Returns:
        Tuple of (t-statistic, p-value).
    """
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("Groups must have at least one element")
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value


def perform_anova(*groups: np.ndarray) -> tuple[float, float]:
    """
    Perform one-way ANOVA on multiple groups.

    Args:
        *groups: Variable number of group arrays.

    Returns:
        Tuple of (F-statistic, p-value).
    """
    try:
        f_stat, p_value = stats.f_oneway(*groups)
        if np.isnan(f_stat):
            f_stat = 0.0
            p_value = 1.0
        return f_stat, p_value
    except:
        return 0.0, 1.0
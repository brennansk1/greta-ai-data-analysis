"""
Narratives Module

Contains functions for generating human-readable narratives from analysis results.
"""

from .causal_narratives import *
from .regression_narratives import *
from .time_series_narratives import *
from .summary_narratives import *

__all__ = [
    # From causal_narratives
    'generate_causal_narrative', 'generate_causal_insight',
    # From regression_narratives
    'generate_regression_narrative',
    # From time_series_narratives
    'generate_time_series_narrative',
    # From summary_narratives
    'generate_hypothesis_narrative', 'generate_summary_narrative',
    'generate_insight_narrative', 'create_report'
]
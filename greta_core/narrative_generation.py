"""
Narrative Generation Module

Translates statistical results into plain-English insights and explanations.
Creates human-readable summaries of findings, including confidence levels
and practical implications.
"""

# Import all functions from submodules to maintain backward compatibility
from .regression_narratives import *
from .time_series_narratives import *
from .summary_narratives import *
from .causal_narratives import *
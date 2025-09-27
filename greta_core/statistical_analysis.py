"""
Statistical Analysis Module

Conducts rigorous statistical testing on generated hypotheses. Supports basic tests
such as t-tests and ANOVA for Phase 1, with extensible design for future additions.
"""

# Import all functions from submodules to maintain backward compatibility
from .significance_tests import *
from .regression_analysis import *
from .time_series_analysis import *
from .causal_analysis import *
"""
Greta Core Engine - Automated Data Analysis Library

This package provides modular components for data ingestion, preprocessing,
hypothesis search via genetic algorithms, statistical analysis, and narrative generation.
"""

__version__ = "0.1.0"

from . import ingestion
from . import preprocessing
from . import statistical_analysis
from . import narrative_generation

try:
    from . import hypothesis_search
    _has_deap = True
except ImportError:
    _has_deap = False
    import warnings
    warnings.warn("DEAP not installed. Hypothesis search functionality will not be available.")

__all__ = [
    "ingestion",
    "preprocessing",
    "statistical_analysis",
    "narrative_generation",
]

if _has_deap:
    __all__.append("hypothesis_search")
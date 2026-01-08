"""
Plugin Architecture for GRETA

This module provides a comprehensive plugin system that allows extending GRETA's
capabilities through custom data connectors, statistical tests, feature engineering
methods, and analysis modules.
"""

from .base import Plugin, PluginManager, PluginError
from .interfaces import (
    DataConnectorPlugin,
    StatisticalTestPlugin,
    FeatureEngineeringPlugin,
    AnalysisPlugin,
    VisualizationPlugin
)
from .discovery import PluginDiscovery
from .config import PluginConfig

__all__ = [
    'Plugin',
    'PluginManager',
    'PluginError',
    'DataConnectorPlugin',
    'StatisticalTestPlugin',
    'FeatureEngineeringPlugin',
    'AnalysisPlugin',
    'VisualizationPlugin',
    'PluginDiscovery',
    'PluginConfig'
]
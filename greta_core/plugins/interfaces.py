"""
Plugin interfaces for different GRETA plugin types.

This module defines the specific interfaces that plugins must implement
for different functionality areas.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd

from .base import Plugin


class DataConnectorPlugin(Plugin):
    """
    Interface for data connector plugins.

    Data connector plugins provide access to external data sources
    like databases, APIs, cloud storage, etc.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection to the data source.
        """
        pass

    @abstractmethod
    def load_data(self, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from the source.

        Args:
            query: Query string or configuration for data retrieval
            **kwargs: Additional parameters

        Returns:
            Pandas DataFrame containing the loaded data
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get schema information for the data source.

        Returns:
            Dictionary describing the data structure
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class StatisticalTestPlugin(Plugin):
    """
    Interface for statistical test plugins.

    Statistical test plugins provide custom hypothesis testing methods
    beyond the built-in tests.
    """

    @abstractmethod
    def perform_test(self, data: Union[pd.DataFrame, pd.Series, List],
                    groups: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform the statistical test.

        Args:
            data: Input data for the test
            groups: Group labels for comparison tests
            **kwargs: Test-specific parameters

        Returns:
            Dictionary containing test results (p-value, statistic, etc.)
        """
        pass

    @abstractmethod
    def get_test_info(self) -> Dict[str, Any]:
        """
        Get information about the statistical test.

        Returns:
            Dictionary with test metadata (name, description, assumptions, etc.)
        """
        pass

    @abstractmethod
    def validate_assumptions(self, data: Union[pd.DataFrame, pd.Series, List]) -> List[str]:
        """
        Validate statistical assumptions for the test.

        Args:
            data: Input data to validate

        Returns:
            List of assumption violations or empty list if all assumptions met
        """
        pass


class FeatureEngineeringPlugin(Plugin):
    """
    Interface for feature engineering plugins.

    Feature engineering plugins provide custom methods for creating
    and transforming features.
    """

    @abstractmethod
    def transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform input data by creating new features.

        Args:
            data: Input DataFrame
            target: Optional target variable for supervised feature engineering

        Returns:
            DataFrame with additional engineered features
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names of features created by this plugin.

        Returns:
            List of feature names
        """
        pass

    @abstractmethod
    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about the feature transformation.

        Returns:
            Dictionary with transformation metadata
        """
        pass


class AnalysisPlugin(Plugin):
    """
    Interface for analysis plugins.

    Analysis plugins provide custom analysis methods and algorithms
    beyond the built-in GRETA capabilities.
    """

    @abstractmethod
    def analyze(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the custom analysis.

        Args:
            data: Input data for analysis
            config: Analysis configuration parameters

        Returns:
            Dictionary containing analysis results
        """
        pass

    @abstractmethod
    def get_analysis_type(self) -> str:
        """
        Get the type of analysis provided by this plugin.

        Returns:
            String identifier for the analysis type
        """
        pass

    @abstractmethod
    def validate_data_requirements(self, data: pd.DataFrame) -> List[str]:
        """
        Validate that input data meets analysis requirements.

        Args:
            data: Input data to validate

        Returns:
            List of validation errors or empty list if valid
        """
        pass


class VisualizationPlugin(Plugin):
    """
    Interface for visualization plugins.

    Visualization plugins provide custom plotting and charting capabilities
    for analysis results.
    """

    @abstractmethod
    def create_visualization(self, data: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """
        Create a visualization from analysis results.

        Args:
            data: Analysis results data
            config: Visualization configuration

        Returns:
            Visualization object (plotly figure, matplotlib figure, etc.)
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported output formats.

        Returns:
            List of format strings (e.g., ['html', 'png', 'svg'])
        """
        pass

    @abstractmethod
    def export_visualization(self, visualization: Any, format: str, path: str) -> None:
        """
        Export visualization to a file.

        Args:
            visualization: The visualization object to export
            format: Export format
            path: Output file path
        """
        pass
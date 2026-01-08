"""
Base Connector Classes and Exceptions

Provides the foundation for all data connectors with common interfaces and error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Iterator
import pandas as pd
import dask.dataframe as dd
import logging

logger = logging.getLogger(__name__)

# Type alias for DataFrame that can be Pandas or Dask
DataFrame = Union[pd.DataFrame, dd.DataFrame]


class ConnectionError(Exception):
    """Base exception for connection failures."""
    pass


class AuthenticationError(ConnectionError):
    """Raised when authentication fails."""
    pass


class QueryError(Exception):
    """Raised when data query fails."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors.

    Provides common interface and lifecycle management for data connections.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the connector.

        Args:
            config: Configuration dictionary specific to the connector type.
        """
        self.config = config
        self.connection = None
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to data source.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError: If connection fails.
            AuthenticationError: If authentication fails.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection to data source."""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate that connection is active and functional.

        Returns:
            True if connection is valid, False otherwise.
        """
        pass

    @abstractmethod
    def load_data(self, query: Optional[str] = None, **kwargs) -> DataFrame:
        """
        Load data from source and return as DataFrame.

        Args:
            query: Optional query string (SQL, API endpoint, etc.)
            **kwargs: Additional connector-specific parameters.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            QueryError: If query execution fails.
            ValidationError: If data validation fails.
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get schema information for the data source.

        Returns:
            Dictionary with schema information.
        """
        pass

    def load_data_streaming(self, query: Optional[str] = None, chunk_size: int = 1000, **kwargs) -> Iterator[DataFrame]:
        """
        Load data in chunks for memory efficiency.

        Args:
            query: Optional query string.
            chunk_size: Number of rows per chunk.
            **kwargs: Additional parameters.

        Yields:
            DataFrame chunks.

        Note:
            Default implementation falls back to load_data().
            Subclasses should override for true streaming support.
        """
        logger.warning(f"Streaming not implemented for {self.__class__.__name__}, falling back to full load")
        yield self.load_data(query, **kwargs)

    def is_connected(self) -> bool:
        """Check if connector is currently connected."""
        return self._connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def _ensure_connected(self):
        """Ensure connection is active, reconnect if necessary."""
        if not self.is_connected():
            self.connect()
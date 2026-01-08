"""
Connector Registry

Manages registration and retrieval of connector classes.
"""

from typing import Dict, Type, Any
import logging

from .base import BaseConnector

logger = logging.getLogger(__name__)

# Global registry of connector classes
_CONNECTOR_REGISTRY: Dict[str, Type[BaseConnector]] = {}


def register_connector(connector_type: str, connector_class: Type[BaseConnector]):
    """
    Register a connector class for a given type.

    Args:
        connector_type: String identifier for the connector type.
        connector_class: Connector class to register.
    """
    if not issubclass(connector_class, BaseConnector):
        raise TypeError(f"Connector class {connector_class} must inherit from BaseConnector")

    _CONNECTOR_REGISTRY[connector_type.lower()] = connector_class
    logger.debug(f"Registered connector: {connector_type} -> {connector_class.__name__}")


def get_connector_class(connector_type: str) -> Type[BaseConnector]:
    """
    Get connector class for a given type.

    Args:
        connector_type: String identifier for the connector type.

    Returns:
        Connector class.

    Raises:
        ValueError: If connector type is not registered.
    """
    connector_class = _CONNECTOR_REGISTRY.get(connector_type.lower())
    if connector_class is None:
        available = list(_CONNECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown connector type: {connector_type}. Available: {available}")

    return connector_class


def list_connector_types() -> list:
    """Get list of all registered connector types."""
    return list(_CONNECTOR_REGISTRY.keys())


def create_connector(connector_type: str, config: Dict[str, Any]) -> BaseConnector:
    """
    Create a connector instance.

    Args:
        connector_type: Type of connector to create.
        config: Configuration dictionary.

    Returns:
        Connector instance.
    """
    connector_class = get_connector_class(connector_type)
    return connector_class(config)
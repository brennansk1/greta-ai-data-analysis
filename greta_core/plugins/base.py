"""
Base classes and core plugin infrastructure for GRETA's plugin system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import logging
import importlib
import inspect

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""
    pass


class PluginValidationError(PluginError):
    """Raised when a plugin fails validation."""
    pass


class Plugin(ABC):
    """
    Base class for all GRETA plugins.

    All plugins must inherit from this class and implement the required methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.

        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = getattr(self, 'VERSION', '1.0.0')
        self.description = getattr(self, 'DESCRIPTION', '')
        self.author = getattr(self, 'AUTHOR', '')
        self.dependencies = getattr(self, 'DEPENDENCIES', [])

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup plugin resources.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information.

        Returns:
            Dictionary containing plugin metadata
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'dependencies': self.dependencies,
            'config': self.config
        }

    def validate_config(self) -> List[str]:
        """
        Validate plugin configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        return []


class PluginManager:
    """
    Manages plugin loading, registration, and lifecycle.
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_types: Dict[str, Type[Plugin]] = {}
        self.logger = logging.getLogger(__name__)

    def register_plugin_type(self, plugin_type: str, base_class: Type[Plugin]) -> None:
        """
        Register a plugin type with its base class.

        Args:
            plugin_type: String identifier for the plugin type
            base_class: Base class that plugins of this type must inherit from
        """
        self.plugin_types[plugin_type] = base_class
        self.logger.info(f"Registered plugin type: {plugin_type}")

    def load_plugin(self, plugin_class: Type[Plugin], config: Optional[Dict[str, Any]] = None) -> Plugin:
        """
        Load and initialize a plugin.

        Args:
            plugin_class: The plugin class to instantiate
            config: Configuration for the plugin

        Returns:
            Initialized plugin instance

        Raises:
            PluginLoadError: If plugin fails to load or initialize
        """
        try:
            # Instantiate plugin
            plugin = plugin_class(config)

            # Validate plugin type
            plugin_type = self._get_plugin_type(plugin_class)
            if plugin_type and not isinstance(plugin, self.plugin_types.get(plugin_type, Plugin)):
                raise PluginValidationError(f"Plugin {plugin_class.__name__} is not a valid {plugin_type} plugin")

            # Initialize plugin
            if not plugin.initialize():
                raise PluginLoadError(f"Plugin {plugin_class.__name__} failed to initialize")

            # Register plugin
            self.plugins[plugin.name] = plugin
            self.logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")

            return plugin

        except Exception as e:
            error_msg = f"Failed to load plugin {plugin_class.__name__}: {str(e)}"
            self.logger.error(error_msg)
            raise PluginLoadError(error_msg) from e

    def unload_plugin(self, plugin_name: str) -> None:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload
        """
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            try:
                plugin.cleanup()
                del self.plugins[plugin_name]
                self.logger.info(f"Unloaded plugin: {plugin_name}")
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_name}: {str(e)}")

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin by name.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: str) -> List[Plugin]:
        """
        Get all loaded plugins of a specific type.

        Args:
            plugin_type: Type of plugins to retrieve

        Returns:
            List of plugins of the specified type
        """
        base_class = self.plugin_types.get(plugin_type)
        if not base_class:
            return []

        return [plugin for plugin in self.plugins.values() if isinstance(plugin, base_class)]

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all loaded plugins with their information.

        Returns:
            Dictionary mapping plugin names to their info
        """
        return {name: plugin.get_info() for name, plugin in self.plugins.items()}

    def validate_all_plugins(self) -> Dict[str, List[str]]:
        """
        Validate all loaded plugins.

        Returns:
            Dictionary mapping plugin names to lists of validation errors
        """
        validation_results = {}
        for name, plugin in self.plugins.items():
            errors = plugin.validate_config()
            if errors:
                validation_results[name] = errors
        return validation_results

    def _get_plugin_type(self, plugin_class: Type[Plugin]) -> Optional[str]:
        """
        Determine the plugin type from its class hierarchy.

        Args:
            plugin_class: The plugin class

        Returns:
            Plugin type string or None if not recognized
        """
        for plugin_type, base_class in self.plugin_types.items():
            if issubclass(plugin_class, base_class):
                return plugin_type
        return None
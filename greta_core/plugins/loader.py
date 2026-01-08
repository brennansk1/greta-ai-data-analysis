"""
Plugin discovery and loading mechanism for GRETA.

This module provides functionality to discover, load, and manage plugins
dynamically at runtime.
"""

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Set
import logging

from .base import Plugin, PluginError, PluginLoadError
from .config import PluginConfig

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Handles discovery and loading of plugins.
    """

    def __init__(self, plugin_config: PluginConfig):
        self.plugin_config = plugin_config
        self.loaded_plugins: Dict[str, Plugin] = {}
        self.plugin_classes: Dict[str, Type[Plugin]] = {}
        self.plugin_paths: Set[Path] = set()

    def add_plugin_path(self, path: Path) -> None:
        """
        Add a directory to search for plugins.

        Args:
            path: Directory path to search for plugins
        """
        if path.is_dir():
            self.plugin_paths.add(path)
            logger.info(f"Added plugin search path: {path}")
        else:
            logger.warning(f"Plugin path is not a directory: {path}")

    def discover_plugins(self) -> Dict[str, Path]:
        """
        Discover available plugins in configured paths.

        Returns:
            Dictionary mapping plugin names to their file paths
        """
        discovered_plugins = {}

        for plugin_path in self.plugin_paths:
            if not plugin_path.exists():
                continue

            # Look for Python files in the plugin directory
            for py_file in plugin_path.glob("*.py"):
                if py_file.name.startswith('_'):
                    continue

                plugin_name = py_file.stem
                discovered_plugins[plugin_name] = py_file

            # Look for plugin subdirectories
            for subdir in plugin_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('_'):
                    init_file = subdir / "__init__.py"
                    if init_file.exists():
                        discovered_plugins[subdir.name] = init_file

        logger.info(f"Discovered {len(discovered_plugins)} potential plugins")
        return discovered_plugins

    def load_plugin_class(self, plugin_name: str, plugin_path: Path) -> Type[Plugin]:
        """
        Load a plugin class from a file path.

        Args:
            plugin_name: Name of the plugin
            plugin_path: Path to the plugin file

        Returns:
            The plugin class

        Raises:
            PluginLoadError: If the plugin cannot be loaded
        """
        try:
            # Convert path to module name
            module_name = self._path_to_module_name(plugin_path)

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Could not create module spec for {plugin_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find the plugin class
            plugin_class = self._find_plugin_class(module, plugin_name)
            if plugin_class is None:
                raise PluginLoadError(f"No plugin class found in {plugin_path}")

            logger.info(f"Loaded plugin class {plugin_class.__name__} from {plugin_path}")
            return plugin_class

        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin {plugin_name} from {plugin_path}: {e}")

    def _path_to_module_name(self, path: Path) -> str:
        """Convert a file path to a module name."""
        # Create a unique module name based on the path
        return f"greta_plugin_{path.stem}_{hash(str(path))}"

    def _find_plugin_class(self, module, plugin_name: str) -> Optional[Type[Plugin]]:
        """
        Find the plugin class in a loaded module.

        Args:
            module: The loaded module
            plugin_name: Expected plugin name

        Returns:
            The plugin class if found, None otherwise
        """
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, Plugin) and
                obj != Plugin):
                return obj

        return None

    def load_plugin(self, plugin_name: str) -> Plugin:
        """
        Load and instantiate a plugin.

        Args:
            plugin_name: Name of the plugin to load

        Returns:
            The loaded plugin instance

        Raises:
            PluginLoadError: If the plugin cannot be loaded
        """
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name]

        # Get plugin configuration
        config = self.plugin_config.get_plugin_config(plugin_name)
        if not config:
            raise PluginLoadError(f"No configuration found for plugin {plugin_name}")

        # Get plugin class
        if plugin_name not in self.plugin_classes:
            # Discover and load the class
            discovered = self.discover_plugins()
            if plugin_name not in discovered:
                raise PluginLoadError(f"Plugin {plugin_name} not found in search paths")

            plugin_path = discovered[plugin_name]
            self.plugin_classes[plugin_name] = self.load_plugin_class(plugin_name, plugin_path)

        plugin_class = self.plugin_classes[plugin_name]

        try:
            # Instantiate the plugin
            plugin_instance = plugin_class(config)
            self.loaded_plugins[plugin_name] = plugin_instance

            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return plugin_instance

        except Exception as e:
            raise PluginLoadError(f"Failed to instantiate plugin {plugin_name}: {e}")

    def load_plugins(self, plugin_names: List[str]) -> Dict[str, Plugin]:
        """
        Load multiple plugins, resolving dependencies.

        Args:
            plugin_names: List of plugin names to load

        Returns:
            Dictionary of loaded plugins

        Raises:
            PluginLoadError: If any plugin cannot be loaded
        """
        # Resolve dependencies
        try:
            ordered_plugins = self.plugin_config.resolve_dependencies(plugin_names)
        except Exception as e:
            raise PluginLoadError(f"Failed to resolve plugin dependencies: {e}")

        loaded = {}

        for plugin_name in ordered_plugins:
            try:
                plugin = self.load_plugin(plugin_name)
                loaded[plugin_name] = plugin
            except Exception as e:
                raise PluginLoadError(f"Failed to load plugin {plugin_name}: {e}")

        return loaded

    def unload_plugin(self, plugin_name: str) -> None:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload
        """
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            try:
                plugin.cleanup()
            except Exception as e:
                logger.warning(f"Error during plugin cleanup for {plugin_name}: {e}")

            del self.loaded_plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")

    def unload_all_plugins(self) -> None:
        """Unload all loaded plugins."""
        for plugin_name in list(self.loaded_plugins.keys()):
            self.unload_plugin(plugin_name)

    def get_loaded_plugins(self) -> Dict[str, Plugin]:
        """
        Get all currently loaded plugins.

        Returns:
            Dictionary of loaded plugins
        """
        return self.loaded_plugins.copy()

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """
        Check if a plugin is currently loaded.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if the plugin is loaded, False otherwise
        """
        return plugin_name in self.loaded_plugins

    def reload_plugin(self, plugin_name: str) -> Plugin:
        """
        Reload a plugin.

        Args:
            plugin_name: Name of the plugin to reload

        Returns:
            The reloaded plugin instance

        Raises:
            PluginLoadError: If the plugin cannot be reloaded
        """
        self.unload_plugin(plugin_name)
        return self.load_plugin(plugin_name)
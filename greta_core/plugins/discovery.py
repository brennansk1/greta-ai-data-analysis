"""
Plugin discovery and loading mechanism for GRETA.

This module handles finding, loading, and managing plugins dynamically.
"""

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Set, Union, Callable
import logging
from abc import ABC, abstractmethod
import pkgutil
import zipimport

from .config import PluginConfigManager, PluginMetadata

logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Exception raised when a plugin fails to load."""
    pass


class PluginInterface(ABC):
    """Base interface that all plugins must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.

        Args:
            config: Plugin configuration dictionary
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description
        )


class PluginLoader:
    """Handles loading of individual plugins."""

    def __init__(self, config_manager: PluginConfigManager):
        """
        Initialize the plugin loader.

        Args:
            config_manager: Plugin configuration manager
        """
        self.config_manager = config_manager
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_modules: Dict[str, Any] = {}

    def load_plugin(self, plugin_path: Path, plugin_name: str) -> Optional[PluginInterface]:
        """
        Load a plugin from a file path.

        Args:
            plugin_path: Path to the plugin file
            plugin_name: Name of the plugin

        Returns:
            Loaded plugin instance, or None if loading failed

        Raises:
            PluginLoadError: If plugin loading fails
        """
        try:
            # Check if plugin is enabled
            if not self.config_manager.is_plugin_enabled(plugin_name):
                logger.debug(f"Plugin {plugin_name} is disabled, skipping")
                return None

            # Load the module
            module = self._load_module_from_path(plugin_path, plugin_name)

            if not module:
                return None

            # Find the plugin class
            plugin_class = self._find_plugin_class(module)

            if not plugin_class:
                logger.warning(f"No plugin class found in {plugin_path}")
                return None

            # Get plugin configuration
            config = self.config_manager.get_plugin_config(plugin_name) or {}

            # Instantiate the plugin
            plugin_instance = plugin_class()

            # Initialize the plugin
            plugin_instance.initialize(config)

            # Store the loaded plugin
            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_modules[plugin_name] = module

            logger.info(f"Successfully loaded plugin: {plugin_name} v{plugin_instance.version}")
            return plugin_instance

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name} from {plugin_path}: {e}")
            raise PluginLoadError(f"Plugin loading failed: {e}")

    def unload_plugin(self, plugin_name: str) -> None:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of the plugin
        """
        if plugin_name in self.loaded_plugins:
            try:
                plugin = self.loaded_plugins[plugin_name]
                plugin.shutdown()
                logger.info(f"Unloaded plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_name}: {e}")
            finally:
                del self.loaded_plugins[plugin_name]
                if plugin_name in self.plugin_modules:
                    del self.plugin_modules[plugin_name]

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """
        Get a loaded plugin instance.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin instance, or None if not loaded
        """
        return self.loaded_plugins.get(plugin_name)

    def get_loaded_plugins(self) -> Dict[str, PluginInterface]:
        """
        Get all loaded plugins.

        Returns:
            Dictionary of plugin name -> plugin instance
        """
        return self.loaded_plugins.copy()

    def _load_module_from_path(self, plugin_path: Path, plugin_name: str) -> Optional[Any]:
        """
        Load a Python module from a file path.

        Args:
            plugin_path: Path to the plugin file
            plugin_name: Name of the plugin

        Returns:
            Loaded module, or None if loading failed
        """
        try:
            # Create a unique module name
            module_name = f"greta_plugin_{plugin_name}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not create module spec for {plugin_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            logger.error(f"Failed to load module from {plugin_path}: {e}")
            return None

    def _find_plugin_class(self, module: Any) -> Optional[Type[PluginInterface]]:
        """
        Find the plugin class in a module.

        Args:
            module: Loaded module

        Returns:
            Plugin class, or None if not found
        """
        # Look for classes that inherit from PluginInterface
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, PluginInterface) and
                obj != PluginInterface):
                return obj

        return None


class PluginDiscovery:
    """Handles discovery of plugins in search paths."""

    def __init__(self, config_manager: PluginConfigManager):
        """
        Initialize the plugin discovery.

        Args:
            config_manager: Plugin configuration manager
        """
        self.config_manager = config_manager
        self.discovered_plugins: Dict[str, Path] = {}

    def discover_plugins(self) -> Dict[str, Path]:
        """
        Discover all plugins in the configured search paths.

        Returns:
            Dictionary of plugin name -> plugin path
        """
        self.discovered_plugins.clear()

        search_paths = self.config_manager.get_plugin_search_paths()

        for search_path in search_paths:
            if not search_path.exists():
                logger.debug(f"Search path does not exist: {search_path}")
                continue

            self._discover_in_path(search_path)

        logger.info(f"Discovered {len(self.discovered_plugins)} plugins")
        return self.discovered_plugins.copy()

    def _discover_in_path(self, search_path: Path) -> None:
        """
        Discover plugins in a specific path.

        Args:
            search_path: Path to search for plugins
        """
        # Look for Python files that might be plugins
        for item in search_path.rglob("*.py"):
            if item.is_file() and not item.name.startswith('_'):
                plugin_name = self._extract_plugin_name(item)
                if plugin_name and plugin_name not in self.discovered_plugins:
                    self.discovered_plugins[plugin_name] = item
                    logger.debug(f"Discovered plugin: {plugin_name} at {item}")

        # Look for plugin directories with __init__.py
        for item in search_path.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                init_file = item / "__init__.py"
                if init_file.exists():
                    plugin_name = item.name
                    if plugin_name not in self.discovered_plugins:
                        self.discovered_plugins[plugin_name] = init_file
                        logger.debug(f"Discovered plugin package: {plugin_name} at {init_file}")

    def _extract_plugin_name(self, plugin_path: Path) -> Optional[str]:
        """
        Extract plugin name from file path.

        Args:
            plugin_path: Path to plugin file

        Returns:
            Plugin name, or None if cannot be determined
        """
        # Use the filename without extension as plugin name
        name = plugin_path.stem

        # Skip common non-plugin files
        if name in ['setup', 'conftest', 'test']:
            return None

        return name

    def get_discovered_plugins(self) -> Dict[str, Path]:
        """
        Get all discovered plugins.

        Returns:
            Dictionary of plugin name -> plugin path
        """
        return self.discovered_plugins.copy()

    def is_plugin_discovered(self, plugin_name: str) -> bool:
        """
        Check if a plugin has been discovered.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if plugin was discovered
        """
        return plugin_name in self.discovered_plugins

    def get_plugin_path(self, plugin_name: str) -> Optional[Path]:
        """
        Get the path for a discovered plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin path, or None if not discovered
        """
        return self.discovered_plugins.get(plugin_name)


class PluginManager:
    """Main plugin management system."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the plugin manager.

        Args:
            config_file: Path to plugin configuration file
        """
        self.config_manager = PluginConfigManager(config_file)
        self.discovery = PluginDiscovery(self.config_manager)
        self.loader = PluginLoader(self.config_manager)
        self.initialized = False

    def initialize(self) -> None:
        """Initialize the plugin system."""
        if self.initialized:
            return

        logger.info("Initializing plugin system")

        # Discover plugins
        discovered = self.discovery.discover_plugins()

        # Load enabled plugins in dependency order
        enabled_plugins = [name for name in discovered.keys()
                          if self.config_manager.is_plugin_enabled(name)]

        if enabled_plugins:
            # Resolve dependencies and get loading order
            try:
                loading_order = self.config_manager.resolve_dependencies(enabled_plugins)
            except Exception as e:
                logger.warning(f"Dependency resolution failed, loading in discovery order: {e}")
                loading_order = enabled_plugins

            # Load plugins
            for plugin_name in loading_order:
                plugin_path = discovered.get(plugin_name)
                if plugin_path:
                    try:
                        self.loader.load_plugin(plugin_path, plugin_name)
                    except Exception as e:
                        logger.error(f"Failed to load plugin {plugin_name}: {e}")

        self.initialized = True
        logger.info(f"Plugin system initialized with {len(self.loader.loaded_plugins)} plugins")

    def shutdown(self) -> None:
        """Shutdown the plugin system."""
        if not self.initialized:
            return

        logger.info("Shutting down plugin system")

        # Unload all plugins
        for plugin_name in list(self.loader.loaded_plugins.keys()):
            self.loader.unload_plugin(plugin_name)

        self.initialized = False
        logger.info("Plugin system shut down")

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """
        Get a loaded plugin instance.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin instance, or None if not loaded
        """
        return self.loader.get_plugin(plugin_name)

    def get_loaded_plugins(self) -> Dict[str, PluginInterface]:
        """
        Get all loaded plugins.

        Returns:
            Dictionary of plugin name -> plugin instance
        """
        return self.loader.get_loaded_plugins()

    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if reload was successful
        """
        try:
            # Unload the plugin
            self.loader.unload_plugin(plugin_name)

            # Re-discover plugins
            self.discovery.discover_plugins()

            # Load the plugin again
            plugin_path = self.discovery.get_plugin_path(plugin_name)
            if plugin_path:
                self.loader.load_plugin(plugin_path, plugin_name)
                return True

        except Exception as e:
            logger.error(f"Failed to reload plugin {plugin_name}: {e}")

        return False

    def reload_all_plugins(self) -> None:
        """Reload all plugins."""
        logger.info("Reloading all plugins")

        # Shutdown and reinitialize
        self.shutdown()
        self.initialize()

    def get_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all plugins.

        Returns:
            Dictionary of plugin name -> plugin info
        """
        info = {}

        # Discovered plugins
        for name, path in self.discovery.get_discovered_plugins().items():
            plugin_info = {
                'path': str(path),
                'discovered': True,
                'loaded': name in self.loader.loaded_plugins,
                'enabled': self.config_manager.is_plugin_enabled(name)
            }

            if name in self.loader.loaded_plugins:
                plugin = self.loader.loaded_plugins[name]
                plugin_info.update({
                    'version': plugin.version,
                    'description': plugin.description,
                    'metadata': plugin.get_metadata().__dict__
                })

            info[name] = plugin_info

        return info

    def install_plugin(self, plugin_path: Union[str, Path], plugin_name: Optional[str] = None) -> bool:
        """
        Install a plugin by copying it to the first search path.

        Args:
            plugin_path: Path to the plugin file to install
            plugin_name: Name for the plugin (auto-detected if None)

        Returns:
            True if installation was successful
        """
        try:
            source_path = Path(plugin_path)
            if not source_path.exists():
                logger.error(f"Plugin source does not exist: {source_path}")
                return False

            # Get the first search path
            search_paths = self.config_manager.get_plugin_search_paths()
            if not search_paths:
                logger.error("No plugin search paths configured")
                return False

            install_dir = search_paths[0]
            install_dir.mkdir(parents=True, exist_ok=True)

            # Determine plugin name
            if not plugin_name:
                plugin_name = source_path.stem

            # Copy the plugin
            dest_path = install_dir / f"{plugin_name}.py"
            import shutil
            shutil.copy2(source_path, dest_path)

            logger.info(f"Installed plugin {plugin_name} to {dest_path}")

            # Re-discover plugins
            self.discovery.discover_plugins()

            return True

        except Exception as e:
            logger.error(f"Failed to install plugin: {e}")
            return False

    def uninstall_plugin(self, plugin_name: str) -> bool:
        """
        Uninstall a plugin by removing its file.

        Args:
            plugin_name: Name of the plugin to uninstall

        Returns:
            True if uninstallation was successful
        """
        try:
            # Unload the plugin first
            self.loader.unload_plugin(plugin_name)

            # Find and remove the plugin file
            plugin_path = self.discovery.get_plugin_path(plugin_name)
            if plugin_path and plugin_path.exists():
                plugin_path.unlink()
                logger.info(f"Uninstalled plugin {plugin_name}")

                # Re-discover plugins
                self.discovery.discover_plugins()

                return True

        except Exception as e:
            logger.error(f"Failed to uninstall plugin {plugin_name}: {e}")

        return False
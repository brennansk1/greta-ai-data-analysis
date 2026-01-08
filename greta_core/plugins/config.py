"""
Plugin configuration management for GRETA.

This module handles plugin configuration loading, validation, dependency resolution,
and runtime configuration management.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin states."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"  # Enable if dependencies are satisfied


@dataclass
class PluginDependency:
    """Represents a plugin dependency."""
    name: str
    version: Optional[str] = None
    required: bool = True
    description: Optional[str] = None

    def __hash__(self):
        return hash((self.name, self.version or ""))

    def __eq__(self, other):
        if not isinstance(other, PluginDependency):
            return False
        return self.name == other.name and self.version == other.version


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    dependencies: List[PluginDependency] = field(default_factory=list)


@dataclass
class PluginConfig:
    """Configuration for a single plugin."""
    name: str
    enabled: PluginState = PluginState.AUTO
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Loading priority (higher = loaded later)


class PluginConfigurationError(Exception):
    """Exception raised when plugin configuration is invalid."""
    pass


class PluginDependencyError(Exception):
    """Exception raised when plugin dependencies cannot be resolved."""
    pass


class PluginConfigManager:
    """Manages plugin configuration and dependencies."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the plugin configuration manager.

        Args:
            config_file: Path to plugin configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self.plugins: Dict[str, PluginConfig] = {}
        self.search_paths: List[Path] = []
        self.auto_discover: bool = True
        self.dependency_graph: nx.DiGraph = nx.DiGraph()

        # Load default configuration
        self._load_default_config()

        # Load user configuration if provided
        if self.config_file and self.config_file.exists():
            self.load_config(self.config_file)

    def _load_default_config(self) -> None:
        """Load default plugin configuration."""
        # Default search paths
        self.search_paths = [
            Path.cwd() / "plugins",
            Path.home() / ".greta" / "plugins",
            Path(__file__).parent.parent.parent / "plugins"  # greta_core/plugins
        ]

        # Default settings
        self.auto_discover = True

    def load_config(self, config_file: Union[str, Path]) -> None:
        """
        Load plugin configuration from file.

        Args:
            config_file: Path to configuration file

        Raises:
            PluginConfigurationError: If configuration is invalid
        """
        config_path = Path(config_file)

        if not config_path.exists():
            logger.warning(f"Plugin config file not found: {config_path}")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise PluginConfigurationError(
                        f"Unsupported config file format: {config_path.suffix}"
                    )

            self._parse_config(config_data)

        except Exception as e:
            raise PluginConfigurationError(f"Failed to load config from {config_path}: {e}")

    def _parse_config(self, config_data: Dict[str, Any]) -> None:
        """
        Parse configuration data.

        Args:
            config_data: Configuration dictionary
        """
        # Parse global settings
        global_config = config_data.get('global', {})

        # Search paths
        if 'search_paths' in global_config:
            self.search_paths = [
                Path(path).expanduser() for path in global_config['search_paths']
            ]

        # Auto discovery
        if 'auto_discover' in global_config:
            self.auto_discover = global_config['auto_discover']

        # Parse plugin configurations
        plugins_config = config_data.get('plugins', {})

        for plugin_name, plugin_data in plugins_config.items():
            if isinstance(plugin_data, dict):
                # Full plugin configuration
                enabled = plugin_data.get('enabled', 'auto')
                config = plugin_data.get('config', {})
                priority = plugin_data.get('priority', 0)
            else:
                # Simple boolean/string configuration
                if isinstance(plugin_data, bool):
                    enabled = PluginState.ENABLED if plugin_data else PluginState.DISABLED
                    config = {}
                    priority = 0
                elif isinstance(plugin_data, str):
                    enabled = PluginState(plugin_data.lower())
                    config = {}
                    priority = 0
                else:
                    logger.warning(f"Invalid plugin config for {plugin_name}: {plugin_data}")
                    continue

            # Convert enabled to enum
            if isinstance(enabled, str):
                try:
                    enabled = PluginState(enabled.lower())
                except ValueError:
                    logger.warning(f"Invalid plugin state for {plugin_name}: {enabled}")
                    enabled = PluginState.AUTO

            plugin_config = PluginConfig(
                name=plugin_name,
                enabled=enabled,
                config=config,
                priority=priority
            )

            self.plugins[plugin_name] = plugin_config

    def save_config(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.

        Args:
            config_file: Path to save configuration, uses self.config_file if None
        """
        save_path = Path(config_file) if config_file else self.config_file

        if not save_path:
            raise PluginConfigurationError("No config file specified")

        # Prepare config data
        config_data = {
            'global': {
                'search_paths': [str(path) for path in self.search_paths],
                'auto_discover': self.auto_discover
            },
            'plugins': {}
        }

        # Add plugin configurations
        for plugin_name, plugin_config in self.plugins.items():
            config_data['plugins'][plugin_name] = {
                'enabled': plugin_config.enabled.value,
                'config': plugin_config.config,
                'priority': plugin_config.priority
            }

        # Save to file
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise PluginConfigurationError(
                        f"Unsupported config file format: {save_path.suffix}"
                    )

            logger.info(f"Saved plugin config to {save_path}")

        except Exception as e:
            raise PluginConfigurationError(f"Failed to save config to {save_path}: {e}")

    def get_plugin_config(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin configuration dictionary, or None if not found
        """
        plugin = self.plugins.get(plugin_name)
        return plugin.config if plugin else None

    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> None:
        """
        Set configuration for a specific plugin.

        Args:
            plugin_name: Name of the plugin
            config: Configuration dictionary
        """
        if plugin_name not in self.plugins:
            self.plugins[plugin_name] = PluginConfig(name=plugin_name)

        self.plugins[plugin_name].config = config

    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """
        Check if a plugin is enabled.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if plugin is enabled
        """
        plugin = self.plugins.get(plugin_name)

        if plugin is None:
            # Plugin not configured, check if auto mode
            return self.auto_discover

        if plugin.enabled == PluginState.ENABLED:
            return True
        elif plugin.enabled == PluginState.DISABLED:
            return False
        elif plugin.enabled == PluginState.AUTO:
            # Check if dependencies are satisfied
            return self._are_dependencies_satisfied(plugin_name)

        return False

    def enable_plugin(self, plugin_name: str) -> None:
        """
        Enable a plugin.

        Args:
            plugin_name: Name of the plugin
        """
        if plugin_name not in self.plugins:
            self.plugins[plugin_name] = PluginConfig(name=plugin_name)

        self.plugins[plugin_name].enabled = PluginState.ENABLED

    def disable_plugin(self, plugin_name: str) -> None:
        """
        Disable a plugin.

        Args:
            plugin_name: Name of the plugin
        """
        if plugin_name not in self.plugins:
            self.plugins[plugin_name] = PluginConfig(name=plugin_name)

        self.plugins[plugin_name].enabled = PluginState.DISABLED

    def get_plugin_search_paths(self) -> List[Path]:
        """
        Get plugin search paths.

        Returns:
            List of search paths
        """
        return self.search_paths.copy()

    def add_search_path(self, path: Union[str, Path]) -> None:
        """
        Add a plugin search path.

        Args:
            path: Path to add
        """
        path_obj = Path(path).expanduser()
        if path_obj not in self.search_paths:
            self.search_paths.append(path_obj)

    def remove_search_path(self, path: Union[str, Path]) -> None:
        """
        Remove a plugin search path.

        Args:
            path: Path to remove
        """
        path_obj = Path(path).expanduser()
        if path_obj in self.search_paths:
            self.search_paths.remove(path_obj)

    def get_plugin_dependencies(self, plugin_name: str) -> List[PluginDependency]:
        """
        Get dependencies for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            List of plugin dependencies
        """
        # This would be populated from plugin metadata
        # For now, return empty list - dependencies are handled by plugins themselves
        return []

    def resolve_dependencies(self, plugin_names: List[str]) -> List[str]:
        """
        Resolve plugin dependencies and return loading order.

        Args:
            plugin_names: List of plugin names to resolve

        Returns:
            List of plugin names in dependency order

        Raises:
            PluginDependencyError: If dependencies cannot be resolved
        """
        # Build dependency graph
        self.dependency_graph.clear()

        # Add all plugins as nodes
        for plugin_name in plugin_names:
            self.dependency_graph.add_node(plugin_name)

        # Add dependency edges
        for plugin_name in plugin_names:
            deps = self.get_plugin_dependencies(plugin_name)
            for dep in deps:
                if dep.required and dep.name in plugin_names:
                    self.dependency_graph.add_edge(dep.name, plugin_name)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            cycles = list(nx.simple_cycles(self.dependency_graph))
            raise PluginDependencyError(f"Dependency cycles detected: {cycles}")

        # Get topological sort (dependency order)
        try:
            ordered_plugins = list(nx.topological_sort(self.dependency_graph))
            return ordered_plugins
        except nx.NetworkXError as e:
            raise PluginDependencyError(f"Failed to resolve dependencies: {e}")

    def get_missing_dependencies(self, plugin_names: List[str]) -> List[str]:
        """
        Get list of missing dependencies for plugins.

        Args:
            plugin_names: List of plugin names

        Returns:
            List of missing dependency names
        """
        missing = []

        for plugin_name in plugin_names:
            deps = self.get_plugin_dependencies(plugin_name)
            for dep in deps:
                if dep.required and dep.name not in plugin_names:
                    missing.append(dep.name)

        return list(set(missing))  # Remove duplicates

    def _are_dependencies_satisfied(self, plugin_name: str) -> bool:
        """
        Check if dependencies for a plugin are satisfied.

        Args:
            plugin_name: Name of the plugin

        Returns:
            True if dependencies are satisfied
        """
        deps = self.get_plugin_dependencies(plugin_name)
        missing = self.get_missing_dependencies([plugin_name])
        return len(missing) == 0

    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.

        Returns:
            List of validation errors
        """
        errors = []

        # Check search paths exist
        for path in self.search_paths:
            if not path.exists():
                errors.append(f"Search path does not exist: {path}")

        # Check plugin configurations
        for plugin_name, plugin_config in self.plugins.items():
            # Validate plugin name
            if not plugin_name or not isinstance(plugin_name, str):
                errors.append(f"Invalid plugin name: {plugin_name}")

            # Validate priority
            if not isinstance(plugin_config.priority, int):
                errors.append(f"Invalid priority for plugin {plugin_name}: {plugin_config.priority}")

        return errors

    def get_enabled_plugins(self) -> List[str]:
        """
        Get list of enabled plugin names.

        Returns:
            List of enabled plugin names
        """
        return [name for name in self.plugins.keys() if self.is_plugin_enabled(name)]

    def get_disabled_plugins(self) -> List[str]:
        """
        Get list of disabled plugin names.

        Returns:
            List of disabled plugin names
        """
        return [name for name in self.plugins.keys() if not self.is_plugin_enabled(name)]

    def get_all_plugins(self) -> List[str]:
        """
        Get list of all configured plugin names.

        Returns:
            List of all plugin names
        """
        return list(self.plugins.keys())

    def clear_config(self) -> None:
        """Clear all plugin configurations."""
        self.plugins.clear()
        self._load_default_config()

    def merge_config(self, other_config: 'PluginConfigManager') -> None:
        """
        Merge another configuration into this one.

        Args:
            other_config: Configuration to merge
        """
        # Merge search paths
        for path in other_config.search_paths:
            if path not in self.search_paths:
                self.search_paths.append(path)

        # Merge plugin configs (other_config takes precedence)
        for plugin_name, plugin_config in other_config.plugins.items():
            self.plugins[plugin_name] = plugin_config

        # Merge global settings
        self.auto_discover = other_config.auto_discover
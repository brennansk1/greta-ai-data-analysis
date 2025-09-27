"""
Configuration handling for Greta CLI.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Data source configuration."""
    source: str = Field(..., description="Path to data file")
    type: str = Field("csv", description="Data file type (csv, excel)")
    target_column: Optional[str] = Field(None, description="Name of target column")
    sheet_name: Optional[str] = Field(None, description="Excel sheet name")


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    missing_strategy: str = Field("mean", description="Strategy for missing values (mean, median, mode, drop)")
    outlier_method: str = Field("iqr", description="Outlier detection method (iqr, zscore)")
    outlier_threshold: float = Field(1.5, description="Outlier detection threshold")
    normalize_types: bool = Field(True, description="Whether to normalize data types")
    feature_engineering: bool = Field(True, description="Whether to perform basic feature engineering")


class HypothesisSearchConfig(BaseModel):
    """Hypothesis search configuration."""
    pop_size: int = Field(100, description="Population size for genetic algorithm")
    num_generations: int = Field(50, description="Number of generations")
    cx_prob: float = Field(0.7, description="Crossover probability")
    mut_prob: float = Field(0.2, description="Mutation probability")


class OutputConfig(BaseModel):
    """Output configuration."""
    format: str = Field("json", description="Output format (json, yaml)")
    file: Optional[str] = Field(None, description="Output file path")


class GretaConfig(BaseModel):
    """Main configuration model."""
    data: DataConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    hypothesis_search: HypothesisSearchConfig = HypothesisSearchConfig()
    output: OutputConfig = OutputConfig()


def load_config(config_path: str) -> GretaConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        GretaConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return GretaConfig(**data)
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")


def save_config(config: GretaConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: GretaConfig instance.
        config_path: Path to save config file.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def validate_config(config: GretaConfig) -> list:
    """
    Validate configuration and return any warnings.

    Args:
        config: GretaConfig instance.

    Returns:
        List of validation warnings.
    """
    warnings = []

    # Check data source
    path = Path(config.data.source)
    if not path.exists():
        warnings.append(f"Data source file does not exist: {config.data.source}")
    if config.data.type not in ['csv', 'excel']:
        warnings.append(f"Unsupported data type: {config.data.type}")

    # Check preprocessing parameters
    if config.preprocessing.missing_strategy not in ['mean', 'median', 'mode', 'drop']:
        warnings.append(f"Invalid missing strategy: {config.preprocessing.missing_strategy}")

    if config.preprocessing.outlier_method not in ['iqr', 'zscore']:
        warnings.append(f"Invalid outlier method: {config.preprocessing.outlier_method}")

    # Check hypothesis search parameters
    if config.hypothesis_search.pop_size < 10:
        warnings.append("Population size too small, may affect results")

    if config.hypothesis_search.num_generations < 10:
        warnings.append("Number of generations too small, may affect results")

    # Check output format
    if config.output.format not in ['json', 'yaml']:
        warnings.append(f"Unsupported output format: {config.output.format}")

    return warnings
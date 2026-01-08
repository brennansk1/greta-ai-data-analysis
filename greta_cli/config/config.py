"""
Configuration handling for Greta CLI.
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class DatabaseConnection(BaseModel):
    """Database connection configuration."""
    host: str = Field(..., description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    database: str = Field(..., description="Database name")
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    ssl_mode: Optional[str] = Field("require", description="SSL mode for connection")
    connection_timeout: Optional[int] = Field(30, description="Connection timeout in seconds")


class CloudConnection(BaseModel):
    """Cloud storage connection configuration."""
    bucket: str = Field(..., description="Storage bucket name")
    region: Optional[str] = Field("us-east-1", description="Cloud region")
    access_key: Optional[str] = Field(None, description="Access key")
    secret_key: Optional[str] = Field(None, description="Secret key")
    endpoint_url: Optional[str] = Field(None, description="Custom endpoint URL")


class APIConnection(BaseModel):
    """API connection configuration."""
    base_url: str = Field(..., description="API base URL")
    endpoint: str = Field("", description="API endpoint path")
    method: str = Field("GET", description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    auth_type: Optional[str] = Field(None, description="Authentication type (bearer, basic, api_key)")
    auth_token: Optional[str] = Field(None, description="Authentication token")
    timeout: Optional[int] = Field(30, description="Request timeout")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Pagination configuration")


class DataConfig(BaseModel):
    """Data source configuration."""
    type: str = Field("csv", description="Data source type (csv, excel, postgres, s3, api, etc.)")

    # File-based sources
    source: Optional[str] = Field(None, description="Path to data file (for csv/excel)")
    sheet_name: Optional[str] = Field(None, description="Excel sheet name")

    # Database connections
    connection: Optional[DatabaseConnection] = None

    # Cloud connections
    cloud_config: Optional[CloudConnection] = None

    # API connections
    api_config: Optional[APIConnection] = None

    # Common options
    query: Optional[str] = Field(None, description="SQL query, API endpoint, or object key")
    target_column: Optional[str] = Field(None, description="Name of target column")
    chunk_size: Optional[int] = Field(None, description="Chunk size for streaming")


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    missing_strategy: str = Field("mean", description="Strategy for missing values (mean, median, mode, drop)")
    outlier_method: str = Field("iqr", description="Outlier detection method (iqr, zscore)")
    outlier_threshold: float = Field(1.5, description="Outlier detection threshold")
    normalize_types: bool = Field(True, description="Whether to normalize data types")
    feature_engineering: bool = Field(True, description="Whether to perform enhanced feature engineering")
    encoding_method: str = Field("auto", description="Categorical encoding method (auto, one_hot, label, ordinal, frequency, target_mean)")
    max_cardinality: int = Field(20, description="Maximum cardinality for categorical encoding")
    numeric_transforms: Optional[list] = Field(["polynomial", "trigonometric", "logarithmic"], description="Numeric feature transformations to apply")
    max_interactions: int = Field(5, description="Maximum number of interaction features to generate")
    feature_selection: bool = Field(False, description="Whether to perform feature selection")
    selection_method: str = Field("mutual_info", description="Feature selection method (correlation, mutual_info, univariate, rfe, importance)")
    max_features: int = Field(50, description="Maximum number of features to select")


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


class AutoMLConfig(BaseModel):
    """AutoML configuration."""
    enabled: bool = Field(False, description="Enable AutoML features")
    task_type: str = Field("auto", description="ML task type (auto, classification, regression, forecasting, clustering)")
    models: Optional[list] = Field(None, description="List of models to try (None for auto-selection)")
    max_models: int = Field(5, description="Maximum number of models to evaluate")
    max_computational_cost: str = Field("high", description="Maximum computational cost (low, medium, high, very_high)")

    # Hyperparameter tuning
    tuning: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "method": "random",
        "max_evals": 50,
        "n_top_models": 3
    }, description="Hyperparameter tuning configuration")

    # Ensemble learning
    ensemble: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "n_models": 3
    }, description="Ensemble learning configuration")

    # Time series specific
    time_series: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "forecast_horizon": 30
    }, description="Time series forecasting configuration")

    # Clustering specific
    clustering: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "n_clusters": "auto"
    }, description="Clustering configuration")

    # Bayesian inference
    bayesian: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "uncertainty_quantification": True
    }, description="Bayesian inference configuration")


class ScalabilityConfig(BaseModel):
    """Scalability and big data configuration."""
    enabled: bool = Field(True, description="Enable scalability features")
    ingestion_backend: str = Field("auto", description="Data ingestion backend")
    distributed_backend: str = Field("auto", description="Distributed computation backend")
    async_processing: bool = Field(False, description="Enable async job processing")

    # Backend-specific configs
    dask_config: Dict[str, Any] = Field(default_factory=dict)
    spark_config: Dict[str, Any] = Field(default_factory=dict)
    celery_config: Dict[str, Any] = Field(default_factory=dict)


class GretaConfig(BaseModel):
    """Main configuration model."""
    data: DataConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    hypothesis_search: HypothesisSearchConfig = HypothesisSearchConfig()
    automl: AutoMLConfig = AutoMLConfig()
    output: OutputConfig = OutputConfig()
    scalability: ScalabilityConfig = ScalabilityConfig()


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
    logger.debug(f"Attempting to load config from {config_path}")
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Config data loaded from YAML: keys={list(data.keys()) if data else 'None'}")
        config = GretaConfig(**data)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to load config: {e}")


def save_config(config: GretaConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: GretaConfig instance.
        config_path: Path to save config file.
    """
    logger.debug(f"Attempting to save config to {config_path}")
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_dict = config.model_dump()
        logger.debug(f"Saving config with keys: {list(config_dict.keys())}")
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved successfully to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}", exc_info=True)
        raise


def validate_config(config: GretaConfig) -> list:
    """
    Validate configuration and return any warnings.

    Args:
        config: GretaConfig instance.

    Returns:
        List of validation warnings.
    """
    warnings = []

    # Check data source based on type
    data_type = config.data.type

    if data_type in ['csv', 'excel']:
        # File-based validation
        if not config.data.source:
            warnings.append("Source path is required for file-based data sources")
        else:
            path = Path(config.data.source)
            if not path.exists():
                warnings.append(f"Data source file does not exist: {config.data.source}")

        if data_type not in ['csv', 'excel']:
            warnings.append(f"Unsupported file type: {data_type}")

    elif data_type in ['postgres', 'postgresql', 'mysql', 'sqlserver', 'mongodb', 'cassandra']:
        # Database validation
        if not config.data.connection:
            warnings.append(f"Database connection configuration required for {data_type}")
        elif not config.data.connection.database:
            warnings.append("Database name is required")

    elif data_type in ['s3', 'gcs', 'azure']:
        # Cloud storage validation
        if not config.data.cloud_config:
            warnings.append(f"Cloud configuration required for {data_type}")
        elif not config.data.cloud_config.bucket:
            warnings.append("Bucket name is required for cloud storage")

    elif data_type in ['api', 'rest']:
        # API validation
        if not config.data.api_config:
            warnings.append(f"API configuration required for {data_type}")
        elif not config.data.api_config.base_url:
            warnings.append("Base URL is required for API connections")

    else:
        warnings.append(f"Unknown data source type: {data_type}")

    # Check preprocessing parameters
    if config.preprocessing.missing_strategy not in ['mean', 'median', 'mode', 'drop']:
        warnings.append(f"Invalid missing strategy: {config.preprocessing.missing_strategy}")

    if config.preprocessing.outlier_method not in ['iqr', 'zscore']:
        warnings.append(f"Invalid outlier method: {config.preprocessing.outlier_method}")

    if config.preprocessing.encoding_method not in ['auto', 'one_hot', 'label', 'ordinal', 'frequency', 'target_mean']:
        warnings.append(f"Invalid encoding method: {config.preprocessing.encoding_method}")

    if config.preprocessing.selection_method not in ['correlation', 'mutual_info', 'univariate', 'rfe', 'importance']:
        warnings.append(f"Invalid selection method: {config.preprocessing.selection_method}")

    # Check hypothesis search parameters
    if config.hypothesis_search.pop_size < 10:
        warnings.append("Population size too small, may affect results")

    if config.hypothesis_search.num_generations < 10:
        warnings.append("Number of generations too small, may affect results")

    # Check output format
    if config.output.format not in ['json', 'yaml']:
        warnings.append(f"Unsupported output format: {config.output.format}")

    # Check scalability parameters
    if config.scalability.ingestion_backend not in ['pandas', 'dask', 'spark', 'auto']:
        warnings.append(f"Invalid ingestion backend: {config.scalability.ingestion_backend}")

    if config.scalability.distributed_backend not in ['multiprocessing', 'dask', 'spark', 'auto']:
        warnings.append(f"Invalid distributed backend: {config.scalability.distributed_backend}")

    return warnings
"""
Data Ingestion Module

Handles loading data from various sources including CSV files, Excel spreadsheets,
databases, cloud storage, and APIs. Provides unified data structures (Pandas DataFrames)
for downstream processing, with support for schema detection and basic validation.
"""

import pandas as pd
import dask.dataframe as dd
from typing import Union, Optional, Dict, Any
import os
import sys
from ..scalability_errors import SparkUnavailableError

# Import connector system
try:
    from .connectors import create_connector
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False

try:
    import pyspark.sql
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    pyspark = None

# Type alias for DataFrame that can be Pandas, Dask, or Spark
DataFrame = Union[pd.DataFrame, dd.DataFrame]
if SPARK_AVAILABLE:
    DataFrame = Union[DataFrame, pyspark.sql.DataFrame]

# Configurable thresholds for switching to Dask
ROW_THRESHOLD = 1_000_000  # 1M rows
SIZE_THRESHOLD_MB = 500  # 500MB


def should_use_dask(file_path: str, estimated_rows: Optional[int] = None) -> bool:
    """
    Determine if Dask should be used based on file size and estimated rows.

    Args:
        file_path: Path to the file.
        estimated_rows: Estimated number of rows (optional).

    Returns:
        True if Dask should be used, False for Pandas.
    """
    if estimated_rows and estimated_rows > ROW_THRESHOLD:
        return True

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > SIZE_THRESHOLD_MB:
            return True
    except OSError:
        pass  # If can't get file size, default to Pandas

    return False


def should_use_spark(file_path: str, estimated_rows: Optional[int] = None) -> bool:
    """
    Determine if Spark should be used based on data size.

    Args:
        file_path: Path to the file.
        estimated_rows: Estimated number of rows (optional).

    Returns:
        True if Spark should be used, False otherwise.
    """
    if not SPARK_AVAILABLE:
        return False

    # Use Spark for very large datasets (>10M rows or >2GB)
    SPARK_ROW_THRESHOLD = 10_000_000
    SPARK_SIZE_THRESHOLD_MB = 2000

    if estimated_rows and estimated_rows > SPARK_ROW_THRESHOLD:
        return True

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > SPARK_SIZE_THRESHOLD_MB:
            return True
    except OSError:
        pass

    return False


def load_spark_dataframe(file_path: str, **kwargs) -> 'pyspark.sql.DataFrame':
    """
    Load data using Spark for distributed processing.

    Args:
        file_path: Path to the file.
        **kwargs: Additional arguments passed to Spark read methods.

    Returns:
        Spark DataFrame.

    Raises:
        ImportError: If PySpark is not available.
        ValueError: If file cannot be loaded.
    """
    if not SPARK_AVAILABLE:
        raise SparkUnavailableError("PySpark is not available. Install pyspark to use Spark backend.")

    from pyspark.sql import SparkSession

    # Get or create Spark session
    spark = SparkSession.builder.getOrCreate()

    try:
        if file_path.endswith('.csv'):
            df = spark.read.csv(file_path, header=True, inferSchema=True, **kwargs)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = spark.read.excel(file_path, header=True, inferSchema=True, **kwargs)
        else:
            raise ValueError(f"Unsupported file format for Spark: {file_path}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load with Spark: {e}")


def convert_to_dask_from_spark(spark_df: 'pyspark.sql.DataFrame') -> dd.DataFrame:
    """
    Convert Spark DataFrame to Dask for compatibility.

    Args:
        spark_df: Spark DataFrame.

    Returns:
        Dask DataFrame.
    """
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark is not available.")

    # Convert to Pandas first, then to Dask
    pandas_df = spark_df.toPandas()
    return dd.from_pandas(pandas_df, npartitions=4)


def convert_to_pandas_from_spark(spark_df: 'pyspark.sql.DataFrame') -> pd.DataFrame:
    """
    Convert Spark DataFrame to Pandas.

    Args:
        spark_df: Spark DataFrame.

    Returns:
        Pandas DataFrame.
    """
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark is not available.")

    return spark_df.toPandas()


def load_csv(file_path: str, backend: str = "auto", **kwargs) -> DataFrame:
    """
    Load data from a CSV file, choosing backend based on configuration.

    Args:
        file_path: Path to the CSV file.
        backend: Backend to use ("pandas", "dask", "spark", "auto").
        **kwargs: Additional arguments passed to read methods.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine backend
    if backend == "auto":
        if should_use_spark(file_path):
            backend = "spark"
        elif should_use_dask(file_path):
            backend = "dask"
        else:
            backend = "pandas"
    elif backend not in ["pandas", "dask", "spark"]:
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        if backend == "spark":
            df = load_spark_dataframe(file_path, **kwargs)
        elif backend == "dask":
            df = dd.read_csv(file_path, **kwargs)
        else:  # pandas
            df = pd.read_csv(file_path, **kwargs)
    except UnicodeDecodeError:
        # Try with latin-1 encoding
        try:
            if backend == "spark":
                df = load_spark_dataframe(file_path, encoding='latin-1', **kwargs)
            elif backend == "dask":
                df = dd.read_csv(file_path, encoding='latin-1', **kwargs)
            else:
                df = pd.read_csv(file_path, encoding='latin-1', **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    # Additional validation (skip for Spark as it's harder to check)
    if backend != "spark":
        if hasattr(df, 'empty') and df.empty:
            raise ValueError("Failed to load CSV: File is empty")
        # Note: to_string() check removed for simplicity

    return df


def load_excel(file_path: str, sheet_name: Optional[Union[str, int]] = 0, backend: str = "auto", **kwargs) -> DataFrame:
    """
    Load data from an Excel file, choosing backend based on configuration.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Name or index of the sheet to load (default: 0).
        backend: Backend to use ("pandas", "dask", "spark", "auto").
        **kwargs: Additional arguments passed to read methods.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as Excel.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine backend
    if backend == "auto":
        if should_use_spark(file_path):
            backend = "spark"
        elif should_use_dask(file_path):
            backend = "dask"
        else:
            backend = "pandas"
    elif backend not in ["pandas", "dask", "spark"]:
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        if backend == "spark":
            # Spark doesn't have built-in Excel support, fall back to pandas then convert
            temp_df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            df = spark.createDataFrame(temp_df)
        elif backend == "dask":
            df = dd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        else:  # pandas
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load Excel: {e}")


def detect_schema(df: DataFrame) -> dict:
    """
    Detect basic schema information from a DataFrame.

    Args:
        df: Input DataFrame (Pandas, Dask, or Spark).

    Returns:
        Dictionary with schema information including column types and basic stats.
    """
    is_dask = hasattr(df, 'compute')
    is_spark = SPARK_AVAILABLE and hasattr(df, 'rdd')  # Spark DataFrame has rdd attribute

    if is_spark:
        shape = (df.count(), len(df.columns))
        dtypes = {field.name: str(field.dataType) for field in df.schema.fields}
        is_dask = False
    elif is_dask:
        shape = (df.shape[0].compute(), df.shape[1])
        dtypes = df.dtypes.to_dict()
    else:
        shape = df.shape
        dtypes = df.dtypes.to_dict()

    schema = {
        'columns': {},
        'shape': shape,
        'dtypes': dtypes,
        'is_dask': is_dask,
        'is_spark': is_spark
    }

    for col in df.columns:
        if is_spark:
            null_count = df.filter(df[col].isNull()).count()
            # For Spark, unique count is expensive, sample values too
            sample_values = df.select(col).limit(3).collect()
            sample_values = [row[0] for row in sample_values if row[0] is not None]
            unique_count = None  # Too expensive for large datasets
            dtype = str(df.schema[col].dataType)
        elif is_dask:
            null_count = df[col].isnull().sum().compute()
            unique_count = df[col].nunique().compute()
            sample_series = df[col].dropna().head(3)
            sample_values = sample_series.compute().tolist() if hasattr(sample_series, 'compute') else sample_series.tolist()
            dtype = str(df[col].dtype)
        else:
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().head(3).tolist()
            dtype = str(df[col].dtype)

        schema['columns'][col] = {
            'dtype': dtype,
            'null_count': null_count,
            'unique_count': unique_count,
            'sample_values': sample_values
        }

    return schema


def validate_data(df: DataFrame) -> list:
    """
    Perform basic validation on the DataFrame.

    Args:
        df: Input DataFrame (Pandas, Dask, or Spark).

    Returns:
        List of validation warnings/errors.
    """
    warnings = []
    is_dask = hasattr(df, 'compute')
    is_spark = SPARK_AVAILABLE and hasattr(df, 'rdd')

    if is_spark:
        is_empty = df.count() == 0
        num_cols = len(df.columns)
    elif is_dask:
        is_empty = len(df) == 0
        num_cols = df.shape[1]
    else:
        is_empty = df.empty
        num_cols = df.shape[1]

    if is_empty:
        warnings.append("DataFrame is empty")

    if num_cols == 0:
        warnings.append("No columns found")

    for col in df.columns:
        if is_spark:
            null_count = df.filter(df[col].isNull()).count()
            total_count = df.count()
            all_null = null_count == total_count
        elif is_dask:
            null_count = df[col].isnull().sum().compute()
            all_null = null_count == len(df)
        else:
            null_count = df[col].isnull().sum()
            all_null = null_count == len(df)

        if all_null:
            warnings.append(f"Column '{col}' is entirely null")
        elif null_count > 0:
            warnings.append(f"Column '{col}' contains {null_count} null values")

    return warnings


# Connector-based data loading functions

def load_from_connector(connector_type: str, config: Dict[str, Any], **kwargs) -> DataFrame:
    """
    Load data using specified connector type.

    Args:
        connector_type: Type of connector to use (e.g., 'postgres', 's3', 'api')
        config: Configuration dictionary for the connector
        **kwargs: Additional parameters passed to connector

    Returns:
        DataFrame containing the loaded data

    Raises:
        ImportError: If connectors are not available
        ValueError: If connector type is unknown
    """
    if not CONNECTORS_AVAILABLE:
        raise ImportError("Data connectors are not available. Install required dependencies.")

    connector = create_connector(connector_type, config)
    with connector:
        return connector.load_data(**kwargs)


def load_data_unified(source_config: Dict[str, Any], **kwargs) -> DataFrame:
    """
    Unified data loading function that dispatches to appropriate loader.

    Args:
        source_config: Configuration dictionary with 'type' and type-specific config
        **kwargs: Additional parameters

    Returns:
        DataFrame containing the loaded data
    """
    source_type = source_config.get('type', 'csv')

    if source_type in ['csv', 'excel']:
        # Use existing file loaders
        source = source_config.get('source')
        if not source:
            raise ValueError("Source path is required for file-based loading")

        if source_type == 'csv':
            return load_csv(source, **kwargs)
        else:  # excel
            sheet_name = source_config.get('sheet_name')
            return load_excel(source, sheet_name=sheet_name, **kwargs)

    elif CONNECTORS_AVAILABLE:
        # Use connector system
        connector_config = source_config.copy()
        connector_config.pop('type', None)  # Remove type from config
        return load_from_connector(source_type, connector_config, **kwargs)

    else:
        raise ValueError(f"Unsupported data source type: {source_type}. Connectors not available.")


def get_available_connectors() -> list:
    """
    Get list of available connector types.

    Returns:
        List of available connector type strings
    """
    if not CONNECTORS_AVAILABLE:
        return []

    try:
        from .connectors.registry import list_connector_types
        return list_connector_types()
    except ImportError:
        return []


# Import connector classes for convenience
from .connectors.base import BaseConnector, ConnectionError, AuthenticationError, QueryError, ValidationError
from .connectors.registry import get_connector_class, register_connector, create_connector, list_connector_types

__all__ = [
    # Original ingestion functions
    'load_csv',
    'load_excel',
    'detect_schema',
    'validate_data',
    'should_use_dask',
    'should_use_spark',
    'load_spark_dataframe',
    'convert_to_dask_from_spark',
    'convert_to_pandas_from_spark',
    'DataFrame',
    'ROW_THRESHOLD',
    'SIZE_THRESHOLD_MB',
    # New connector functions
    'load_from_connector',
    'load_data_unified',
    'get_available_connectors',
    # Connector classes
    'BaseConnector',
    'ConnectionError',
    'AuthenticationError',
    'QueryError',
    'ValidationError',
    'get_connector_class',
    'register_connector',
    'create_connector',
    'list_connector_types'
]
"""
Spark Connector Module

Provides a connector for Apache Spark integration, handling session management
and data loading for distributed processing.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkSession = None
    logger.warning("PySpark not available. Spark features will be disabled.")


class SparkConnector:
    """
    Connector for Apache Spark operations.

    Manages Spark session lifecycle and provides data loading capabilities
    for distributed processing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Spark connector.

        Args:
            config: Configuration dictionary with Spark settings.
        """
        self.config = config
        self.spark: Optional[SparkSession] = None

    def initialize_spark_session(self) -> Optional[SparkSession]:
        """
        Create and configure Spark session.

        Returns:
            Configured SparkSession instance, or None if Spark unavailable.
        """
        if not SPARK_AVAILABLE:
            logger.error("PySpark is not installed. Cannot initialize Spark session.")
            return None

        try:
            builder = SparkSession.builder.appName("GretaAnalysis")

            # Apply configuration from config
            spark_config = self.config.get('spark_config', {})
            for key, value in spark_config.items():
                builder = builder.config(key, value)

            # Default configurations if not specified
            defaults = {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.executor.memory": "2g",
                "spark.driver.memory": "2g",
            }

            for key, value in defaults.items():
                if key not in spark_config:
                    builder = builder.config(key, value)

            self.spark = builder.getOrCreate()
            logger.info("Spark session initialized successfully.")
            return self.spark

        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {e}")
            return None

    def load_data(self, file_path: str, **kwargs) -> Optional['pyspark.sql.DataFrame']:
        """
        Load data using Spark.

        Args:
            file_path: Path to the data file.
            **kwargs: Additional arguments for Spark read methods.

        Returns:
            Spark DataFrame, or None if loading fails.
        """
        if not self.spark:
            logger.error("Spark session not initialized.")
            return None

        try:
            if file_path.endswith('.csv'):
                df = self.spark.read.csv(file_path, header=True, inferSchema=True, **kwargs)
            elif file_path.endswith(('.xlsx', '.xls')):
                # Spark doesn't have native Excel support, use pandas then convert
                import pandas as pd
                temp_df = pd.read_excel(file_path, **kwargs)
                df = self.spark.createDataFrame(temp_df)
            else:
                logger.error(f"Unsupported file format for Spark: {file_path}")
                return None

            logger.info(f"Data loaded successfully with Spark from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data with Spark: {e}")
            return None

    def convert_to_pandas(self, df: 'pyspark.sql.DataFrame') -> Optional['pd.DataFrame']:
        """
        Convert Spark DataFrame to Pandas for compatibility.

        Args:
            df: Spark DataFrame to convert.

        Returns:
            Pandas DataFrame, or None if conversion fails.
        """
        if not SPARK_AVAILABLE:
            logger.error("PySpark not available for conversion.")
            return None

        try:
            pandas_df = df.toPandas()
            logger.info("Spark DataFrame converted to Pandas successfully.")
            return pandas_df
        except Exception as e:
            logger.error(f"Failed to convert Spark DataFrame to Pandas: {e}")
            return None

    def close_session(self):
        """Close the Spark session."""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session closed.")
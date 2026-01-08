"""
PostgreSQL Database Connector

Provides connectivity to PostgreSQL databases with support for SQL queries and schema detection.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from .base import BaseConnector, ConnectionError, AuthenticationError, QueryError

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    logger.warning("psycopg2 not available. PostgreSQL connector will be disabled.")


class PostgreSQLConnector(BaseConnector):
    """
    Connector for PostgreSQL databases.

    Supports connection pooling, SQL queries, and automatic schema detection.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._connection_string = self._build_connection_string()

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from config."""
        config = self.config

        # Required parameters
        host = config.get('host', 'localhost')
        database = config.get('database')
        if not database:
            raise ValueError("Database name is required for PostgreSQL connection")

        # Optional parameters
        port = config.get('port', 5432)
        user = config.get('username')
        password = config.get('password')
        ssl_mode = config.get('ssl_mode', 'require')
        connect_timeout = config.get('connection_timeout', 30)

        # Build connection string
        conn_parts = [
            f"host={host}",
            f"port={port}",
            f"dbname={database}",
            f"sslmode={ssl_mode}",
            f"connect_timeout={connect_timeout}"
        ]

        if user:
            conn_parts.append(f"user={user}")
        if password:
            conn_parts.append(f"password={password}")

        return " ".join(conn_parts)

    def connect(self) -> bool:
        """Establish connection to PostgreSQL database."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL connections")

        try:
            logger.debug("Attempting to connect to PostgreSQL database")
            self.connection = psycopg2.connect(self._connection_string)
            self.connection.autocommit = True  # Enable autocommit for queries
            self._connected = True
            logger.info("Successfully connected to PostgreSQL database")
            return True

        except psycopg2.OperationalError as e:
            error_msg = f"Failed to connect to PostgreSQL: {e}"
            logger.error(error_msg)
            if "authentication failed" in str(e).lower():
                raise AuthenticationError(error_msg) from e
            else:
                raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to PostgreSQL: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def disconnect(self):
        """Close PostgreSQL connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.debug("PostgreSQL connection closed")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")
            finally:
                self.connection = None
                self._connected = False

    def validate_connection(self) -> bool:
        """Validate PostgreSQL connection is active."""
        if not self.connection:
            return False

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.warning(f"PostgreSQL connection validation failed: {e}")
            self._connected = False
            return False

    def load_data(self, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from PostgreSQL using SQL query."""
        self._ensure_connected()

        if not query:
            raise QueryError("Query is required for PostgreSQL data loading")

        try:
            logger.debug(f"Executing PostgreSQL query: {query[:100]}...")

            # Use pandas read_sql for efficient data loading
            df = pd.read_sql_query(query, self.connection, **kwargs)

            logger.info(f"Loaded {len(df)} rows from PostgreSQL")
            return df

        except Exception as e:
            error_msg = f"Failed to execute PostgreSQL query: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e

    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for PostgreSQL database."""
        self._ensure_connected()

        try:
            schema_info = {
                'tables': [],
                'columns': {}
            }

            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Get all tables
                cursor.execute("""
                    SELECT schemaname, tablename
                    FROM pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY schemaname, tablename
                """)

                tables = cursor.fetchall()
                schema_info['tables'] = [f"{row['schemaname']}.{row['tablename']}" for row in tables]

                # Get column information for each table
                for table_info in tables:
                    schema_name = table_info['schemaname']
                    table_name = table_info['tablename']
                    full_table_name = f"{schema_name}.{table_name}"

                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                    """, (schema_name, table_name))

                    columns = cursor.fetchall()
                    schema_info['columns'][full_table_name] = [
                        {
                            'name': col['column_name'],
                            'type': col['data_type'],
                            'nullable': col['is_nullable'] == 'YES',
                            'default': col['column_default']
                        }
                        for col in columns
                    ]

            return schema_info

        except Exception as e:
            error_msg = f"Failed to get PostgreSQL schema: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e
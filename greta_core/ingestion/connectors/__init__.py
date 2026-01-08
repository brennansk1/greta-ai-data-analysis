"""
Data Connectors Module

Provides connectors for various data sources including databases, cloud storage, and APIs.
"""

from .base import BaseConnector, ConnectionError, AuthenticationError, QueryError, ValidationError
from .registry import get_connector_class, register_connector, create_connector

# Import connector implementations to register them
try:
    from .postgres import PostgreSQLConnector
    register_connector('postgres', PostgreSQLConnector)
    register_connector('postgresql', PostgreSQLConnector)
except ImportError:
    pass  # psycopg2 not available

try:
    from .s3 import S3Connector
    register_connector('s3', S3Connector)
except ImportError:
    pass  # boto3 not available

try:
    from .mongodb import MongoDBConnector
    register_connector('mongodb', MongoDBConnector)
    register_connector('mongo', MongoDBConnector)
except ImportError:
    pass  # pymongo not available

try:
    from .api import APIConnector
    register_connector('api', APIConnector)
    register_connector('rest', APIConnector)
except ImportError:
    pass  # requests not available

__all__ = [
    'BaseConnector',
    'ConnectionError',
    'AuthenticationError',
    'QueryError',
    'ValidationError',
    'get_connector_class',
    'register_connector',
    'create_connector'
]
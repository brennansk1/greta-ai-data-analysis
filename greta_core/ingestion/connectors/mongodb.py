"""
MongoDB Database Connector

Provides connectivity to MongoDB databases with support for queries and document flattening.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

from .base import BaseConnector, ConnectionError, AuthenticationError, QueryError

logger = logging.getLogger(__name__)

try:
    import pymongo
    from pymongo.errors import ConnectionFailure, OperationFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    pymongo = None
    ConnectionFailure = Exception
    OperationFailure = Exception
    logger.warning("pymongo not available. MongoDB connector will be disabled.")


class MongoDBConnector(BaseConnector):
    """
    Connector for MongoDB databases.

    Supports connection to MongoDB collections with query support and automatic document flattening.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 27017)
        self.database = config.get('database')
        if not self.database:
            raise ValueError("Database name is required for MongoDB connection")

        self.collection = config.get('collection')
        self.username = config.get('username')
        self.password = config.get('password')
        self.ssl = config.get('ssl', False)
        self.auth_source = config.get('auth_source', 'admin')

        self._client = None
        self._db = None

    def _build_connection_string(self) -> str:
        """Build MongoDB connection string."""
        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"

        ssl_part = "?ssl=true" if self.ssl else ""

        return f"mongodb://{auth_part}{self.host}:{self.port}/{self.auth_source}{ssl_part}"

    def connect(self) -> bool:
        """Establish connection to MongoDB."""
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB connections")

        try:
            logger.debug("Attempting to connect to MongoDB")
            self._client = pymongo.MongoClient(self._build_connection_string())

            # Test connection
            self._client.admin.command('ping')
            self._db = self._client[self.database]

            self._connected = True
            logger.info("Successfully connected to MongoDB")
            return True

        except ConnectionFailure as e:
            error_msg = f"Failed to connect to MongoDB: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except OperationFailure as e:
            if "Authentication failed" in str(e):
                error_msg = "MongoDB authentication failed"
                logger.error(error_msg)
                raise AuthenticationError(error_msg) from e
            else:
                error_msg = f"MongoDB operation failed: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to MongoDB: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def disconnect(self):
        """Close MongoDB connection."""
        if self._client:
            try:
                self._client.close()
                logger.debug("MongoDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {e}")
            finally:
                self._client = None
                self._db = None
                self._connected = False

    def validate_connection(self) -> bool:
        """Validate MongoDB connection is active."""
        if not self._client or not self._db:
            return False

        try:
            # Try to list collections
            list(self._db.list_collections(limit=1))
            return True
        except Exception as e:
            logger.warning(f"MongoDB connection validation failed: {e}")
            self._connected = False
            return False

    def load_data(self, query: Optional[Dict[str, Any]] = None, **kwargs) -> pd.DataFrame:
        """Load data from MongoDB collection."""
        self._ensure_connected()

        if not self.collection:
            raise QueryError("Collection name is required for MongoDB data loading")

        try:
            logger.debug(f"Loading data from MongoDB collection: {self.collection}")

            # Default query is empty (get all documents)
            mongo_query = query or {}

            # Get collection
            coll = self._db[self.collection]

            # Execute query
            cursor = coll.find(mongo_query, **kwargs)

            # Convert to list of dicts, then to DataFrame
            documents = list(cursor)

            if not documents:
                logger.warning("No documents found matching the query")
                return pd.DataFrame()

            # Flatten nested documents and convert to DataFrame
            df = pd.json_normalize(documents)

            logger.info(f"Loaded {len(df)} documents from MongoDB collection {self.collection}")
            return df

        except Exception as e:
            error_msg = f"Failed to load data from MongoDB: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e

    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for MongoDB database."""
        self._ensure_connected()

        try:
            schema_info = {
                'database': self.database,
                'collections': []
            }

            # Get collection names
            collections = self._db.list_collection_names()
            schema_info['collections'] = collections

            # Get sample document from each collection for schema inference
            for collection_name in collections:
                try:
                    coll = self._db[collection_name]
                    sample_doc = coll.find_one()

                    if sample_doc:
                        # Get field names and types from sample
                        fields = {}
                        for key, value in sample_doc.items():
                            if key != '_id':  # Skip MongoDB ObjectId
                                fields[key] = type(value).__name__

                        schema_info[collection_name] = {
                            'document_count': coll.count_documents({}),
                            'sample_fields': fields
                        }
                except Exception as e:
                    logger.warning(f"Could not get schema for collection {collection_name}: {e}")

            return schema_info

        except Exception as e:
            error_msg = f"Failed to get MongoDB schema: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e
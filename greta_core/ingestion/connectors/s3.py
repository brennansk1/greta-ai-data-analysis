"""
AWS S3 Storage Connector

Provides connectivity to Amazon S3 for loading data from cloud storage.
Supports CSV, JSON, and Parquet files.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging
from io import BytesIO

from .base import BaseConnector, ConnectionError, AuthenticationError, QueryError

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception
    logger.warning("boto3 not available. S3 connector will be disabled.")


class S3Connector(BaseConnector):
    """
    Connector for Amazon S3 storage.

    Supports loading data from S3 buckets with various file formats.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bucket = config.get('bucket')
        if not self.bucket:
            raise ValueError("Bucket name is required for S3 connection")

        self.region = config.get('region', 'us-east-1')
        self.endpoint_url = config.get('endpoint_url')
        self._s3_client = None

    def connect(self) -> bool:
        """Establish connection to S3."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 connections")

        try:
            logger.debug("Attempting to connect to S3")

            # Build client configuration
            client_kwargs = {
                'region_name': self.region
            }

            if self.endpoint_url:
                client_kwargs['endpoint_url'] = self.endpoint_url

            # Get credentials from config or environment
            aws_access_key = self.config.get('access_key')
            aws_secret_key = self.config.get('secret_key')

            if aws_access_key and aws_secret_key:
                client_kwargs['aws_access_key_id'] = aws_access_key
                client_kwargs['aws_secret_access_key'] = aws_secret_key

            self._s3_client = boto3.client('s3', **client_kwargs)
            self._connected = True
            logger.info("Successfully connected to S3")
            return True

        except NoCredentialsError as e:
            error_msg = "AWS credentials not found"
            logger.error(error_msg)
            raise AuthenticationError(error_msg) from e
        except ClientError as e:
            error_msg = f"S3 connection failed: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to S3: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def disconnect(self):
        """Close S3 connection."""
        if self._s3_client:
            # boto3 clients don't need explicit closing
            self._s3_client = None
            self._connected = False
            logger.debug("S3 connection closed")

    def validate_connection(self) -> bool:
        """Validate S3 connection is active."""
        if not self._s3_client:
            return False

        try:
            # Try to list objects in bucket (with max 1 result)
            self._s3_client.list_objects_v2(Bucket=self.bucket, MaxKeys=1)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.warning(f"S3 bucket does not exist: {self.bucket}")
            else:
                logger.warning(f"S3 connection validation failed: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.warning(f"S3 connection validation failed: {e}")
            self._connected = False
            return False

    def load_data(self, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from S3 object.

        Args:
            query: S3 object key (path to file in bucket)
            **kwargs: Additional pandas read parameters
        """
        self._ensure_connected()

        if not query:
            raise QueryError("S3 object key is required for data loading")

        try:
            logger.debug(f"Loading data from S3: s3://{self.bucket}/{query}")

            # Get object from S3
            response = self._s3_client.get_object(Bucket=self.bucket, Key=query)
            body = response['Body']

            # Determine file format from key or content type
            file_format = self._detect_format(query, response.get('ContentType', ''))

            # Read data based on format
            if file_format == 'csv':
                df = pd.read_csv(BytesIO(body.read()), **kwargs)
            elif file_format == 'json':
                df = pd.read_json(BytesIO(body.read()), **kwargs)
            elif file_format == 'parquet':
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(BytesIO(body.read()))
                    df = table.to_pandas(**kwargs)
                except ImportError:
                    raise ImportError("pyarrow is required for Parquet file support")
            else:
                raise QueryError(f"Unsupported file format: {file_format}")

            logger.info(f"Loaded {len(df)} rows from S3 object {query}")
            return df

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                error_msg = f"S3 object does not exist: s3://{self.bucket}/{query}"
            else:
                error_msg = f"Failed to load from S3: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load data from S3: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e

    def _detect_format(self, key: str, content_type: str) -> str:
        """Detect file format from key or content type."""
        key_lower = key.lower()

        if key_lower.endswith('.csv') or 'csv' in content_type:
            return 'csv'
        elif key_lower.endswith(('.json', '.jsonl')) or 'json' in content_type:
            return 'json'
        elif key_lower.endswith('.parquet') or 'parquet' in content_type:
            return 'parquet'
        else:
            # Default to CSV
            return 'csv'

    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for S3 bucket."""
        self._ensure_connected()

        try:
            schema_info = {
                'bucket': self.bucket,
                'objects': []
            }

            # List objects in bucket
            paginator = self._s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket)

            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        schema_info['objects'].append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'etag': obj['ETag']
                        })

            return schema_info

        except ClientError as e:
            error_msg = f"Failed to get S3 schema: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e
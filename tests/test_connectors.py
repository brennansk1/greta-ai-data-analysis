"""
Tests for data connectors.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from greta_core.ingestion import (
    BaseConnector,
    ConnectionError,
    AuthenticationError,
    QueryError,
    ValidationError,
    get_connector_class,
    create_connector,
    list_connector_types
)


class TestBaseConnector:
    """Test base connector functionality."""

    def test_base_connector_abstract_methods(self):
        """Test that BaseConnector has abstract methods."""
        # This should raise TypeError since we can't instantiate abstract class
        with pytest.raises(TypeError):
            BaseConnector({})

    def test_connector_registry(self):
        """Test connector registration and retrieval."""
        # Test listing available connectors
        available = list_connector_types()
        assert isinstance(available, list)

        # Test getting a non-existent connector
        with pytest.raises(ValueError, match="Unknown connector type"):
            get_connector_class("nonexistent")


class TestPostgreSQLConnector:
    """Test PostgreSQL connector."""

    @patch('greta_core.ingestion.connectors.postgres.psycopg2')
    def test_postgres_connector_creation(self, mock_psycopg2):
        """Test PostgreSQL connector can be created."""
        mock_psycopg2.connect.return_value = Mock()
        mock_psycopg2.connect.return_value.autocommit = True

        config = {
            'host': 'localhost',
            'database': 'testdb',
            'username': 'user',
            'password': 'pass'
        }

        connector = create_connector('postgres', config)
        assert connector is not None
        assert hasattr(connector, 'connect')

    @patch('greta_core.ingestion.connectors.postgres.PSYCOPG2_AVAILABLE', False)
    def test_postgres_unavailable(self):
        """Test behavior when psycopg2 is not available."""
        config = {'host': 'localhost', 'database': 'testdb'}

        with pytest.raises(ImportError, match="psycopg2 is required"):
            connector = create_connector('postgres', config)
            connector.connect()


class TestS3Connector:
    """Test S3 connector."""

    @patch('greta_core.ingestion.connectors.s3.boto3')
    def test_s3_connector_creation(self, mock_boto3):
        """Test S3 connector can be created."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        config = {
            'bucket': 'test-bucket',
            'region': 'us-east-1'
        }

        connector = create_connector('s3', config)
        assert connector is not None
        assert hasattr(connector, 'connect')

    @patch('greta_core.ingestion.connectors.s3.BOTO3_AVAILABLE', False)
    def test_s3_unavailable(self):
        """Test behavior when boto3 is not available."""
        config = {'bucket': 'test-bucket'}

        with pytest.raises(ImportError, match="boto3 is required"):
            connector = create_connector('s3', config)
            connector.connect()


class TestAPIConnector:
    """Test API connector."""

    @patch('greta_core.ingestion.connectors.api.requests')
    def test_api_connector_creation(self, mock_requests):
        """Test API connector can be created."""
        mock_session = Mock()
        mock_requests.Session.return_value = mock_session

        config = {
            'base_url': 'https://api.example.com',
            'endpoint': '/data'
        }

        connector = create_connector('api', config)
        assert connector is not None
        assert hasattr(connector, 'connect')

    @patch('greta_core.ingestion.connectors.api.REQUESTS_AVAILABLE', False)
    def test_api_unavailable(self):
        """Test behavior when requests is not available."""
        config = {'base_url': 'https://api.example.com'}

        with pytest.raises(ImportError, match="requests is required"):
            connector = create_connector('api', config)
            connector.connect()


class TestMongoDBConnector:
    """Test MongoDB connector."""

    @patch('greta_core.ingestion.connectors.mongodb.pymongo')
    def test_mongodb_connector_creation(self, mock_pymongo):
        """Test MongoDB connector can be created."""
        mock_client = Mock()
        mock_pymongo.MongoClient.return_value = mock_client
        mock_client.admin.command.return_value = None  # ping succeeds

        config = {
            'host': 'localhost',
            'database': 'testdb',
            'collection': 'testcoll'
        }

        connector = create_connector('mongodb', config)
        assert connector is not None
        assert hasattr(connector, 'connect')

    @patch('greta_core.ingestion.connectors.mongodb.PYMONGO_AVAILABLE', False)
    def test_mongodb_unavailable(self):
        """Test behavior when pymongo is not available."""
        config = {'host': 'localhost', 'database': 'testdb'}

        with pytest.raises(ImportError, match="pymongo is required"):
            connector = create_connector('mongodb', config)
            connector.connect()


class TestIngestionIntegration:
    """Test integration with ingestion module."""

    @patch('greta_core.ingestion.CONNECTORS_AVAILABLE', True)
    @patch('greta_core.ingestion.create_connector')
    def test_load_from_connector(self, mock_create_connector, sample_data):
        """Test loading data from connector."""
        mock_connector = Mock()
        mock_connector.load_data.return_value = sample_data
        mock_connector.__enter__ = Mock(return_value=mock_connector)
        mock_connector.__exit__ = Mock(return_value=None)
        mock_create_connector.return_value = mock_connector

        from greta_core.ingestion import load_from_connector

        result = load_from_connector('test', {'key': 'value'})
        pd.testing.assert_frame_equal(result, sample_data)

    def test_load_data_unified_file(self, tmp_path, sample_csv_data):
        """Test unified loading for file sources."""
        from greta_core.ingestion import load_data_unified

        # Create test CSV file
        csv_file = tmp_path / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)

        config = {
            'type': 'csv',
            'source': str(csv_file)
        }

        result = load_data_unified(config)
        pd.testing.assert_frame_equal(result, sample_csv_data)

    @patch('greta_core.ingestion.CONNECTORS_AVAILABLE', False)
    def test_load_data_unified_connector_unavailable(self):
        """Test unified loading when connectors are not available."""
        from greta_core.ingestion import load_data_unified

        config = {
            'type': 'postgres',
            'connection': {'host': 'localhost', 'database': 'test'}
        }

        with pytest.raises(ValueError, match="Connectors not available"):
            load_data_unified(config)
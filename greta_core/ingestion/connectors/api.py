"""
Generic API Connector

Provides connectivity to REST and GraphQL APIs for data ingestion.
Supports authentication, pagination, and various response formats.
"""

from typing import Dict, Any, Optional, Iterator
import pandas as pd
import logging
import json
from urllib.parse import urljoin, urlparse

from .base import BaseConnector, ConnectionError, AuthenticationError, QueryError

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None
    logger.warning("requests not available. API connector will be disabled.")


class APIConnector(BaseConnector):
    """
    Generic connector for REST and GraphQL APIs.

    Supports various authentication methods and automatic pagination.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.base_url = config.get('base_url')
        if not self.base_url:
            raise ValueError("base_url is required for API connection")

        self.endpoint = config.get('endpoint', '')
        self.method = config.get('method', 'GET').upper()
        self.headers = config.get('headers', {})
        self.params = config.get('params', {})
        self.auth_type = config.get('auth_type')
        self.auth_token = config.get('auth_token')
        self.timeout = config.get('timeout', 30)

        # Pagination settings
        self.pagination = config.get('pagination', {})

        # Session for connection reuse
        self._session = None

    def connect(self) -> bool:
        """Establish API connection (create session)."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required for API connections")

        try:
            logger.debug("Creating API session")
            self._session = requests.Session()

            # Set default headers
            self._session.headers.update({
                'User-Agent': 'Greta-Data-Connector/1.0',
                'Accept': 'application/json'
            })

            # Add custom headers
            self._session.headers.update(self.headers)

            # Configure authentication
            self._configure_auth()

            self._connected = True
            logger.info("API session created successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to create API session: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def _configure_auth(self):
        """Configure authentication for the session."""
        if not self.auth_type or not self.auth_token:
            return

        if self.auth_type.lower() == 'bearer':
            self._session.headers['Authorization'] = f"Bearer {self.auth_token}"
        elif self.auth_type.lower() == 'basic':
            # Basic auth token should be base64 encoded user:pass
            self._session.headers['Authorization'] = f"Basic {self.auth_token}"
        elif self.auth_type.lower() == 'api_key':
            # Add API key to headers (key name could be configurable)
            self._session.headers['X-API-Key'] = self.auth_token
        else:
            logger.warning(f"Unknown auth type: {self.auth_type}")

    def disconnect(self):
        """Close API session."""
        if self._session:
            self._session.close()
            self._session = None
            self._connected = False
            logger.debug("API session closed")

    def validate_connection(self) -> bool:
        """Validate API connection is active."""
        if not self._session:
            return False

        try:
            # Try a simple HEAD request to base URL
            response = self._session.head(self.base_url, timeout=10)
            return response.status_code < 400
        except Exception as e:
            logger.warning(f"API connection validation failed: {e}")
            self._connected = False
            return False

    def load_data(self, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from API endpoint.

        Args:
            query: Optional query string or GraphQL query
            **kwargs: Additional parameters
        """
        self._ensure_connected()

        # Build full URL
        url = urljoin(self.base_url, query or self.endpoint)

        try:
            logger.debug(f"Making {self.method} request to {url}")

            # Prepare request parameters
            request_kwargs = {
                'timeout': self.timeout
            }

            # Add query parameters
            params = self.params.copy()
            if kwargs.get('params'):
                params.update(kwargs['params'])

            if self.method in ['GET', 'DELETE']:
                request_kwargs['params'] = params
            elif self.method in ['POST', 'PUT', 'PATCH']:
                # Check if this is a GraphQL request
                if self._is_graphql_request(query):
                    request_kwargs['json'] = {'query': query}
                else:
                    request_kwargs['json'] = params

            # Make request
            response = self._session.request(self.method, url, **request_kwargs)
            response.raise_for_status()

            # Parse response
            return self._parse_response(response, **kwargs)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            if e.response.status_code == 401:
                raise AuthenticationError(error_msg) from e
            else:
                raise QueryError(error_msg) from e
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def _is_graphql_request(self, query: Optional[str]) -> bool:
        """Check if the request is a GraphQL query."""
        if not query:
            return False
        # Simple heuristic: GraphQL queries often start with query/mutation
        query_lower = query.strip().lower()
        return query_lower.startswith(('query', 'mutation', '{'))

    def _parse_response(self, response: 'requests.Response', **kwargs) -> pd.DataFrame:
        """Parse API response into DataFrame."""
        content_type = response.headers.get('content-type', '').lower()

        try:
            if 'json' in content_type:
                data = response.json()

                # Handle different JSON response structures
                if isinstance(data, list):
                    # Direct array of objects
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try common patterns for nested data
                    if 'data' in data:
                        nested_data = data['data']
                        if isinstance(nested_data, list):
                            df = pd.DataFrame(nested_data)
                        elif isinstance(nested_data, dict):
                            # Single object, convert to single-row DataFrame
                            df = pd.DataFrame([nested_data])
                        else:
                            raise QueryError(f"Unexpected data structure in 'data' field: {type(nested_data)}")
                    elif 'results' in data:
                        df = pd.DataFrame(data['results'])
                    else:
                        # Assume the dict values are the data
                        # This is a fallback that might not work for all APIs
                        df = pd.DataFrame([data])
                else:
                    raise QueryError(f"Unexpected JSON response type: {type(data)}")

            elif 'csv' in content_type:
                import io
                df = pd.read_csv(io.StringIO(response.text), **kwargs)
            else:
                # Try to parse as JSON anyway
                try:
                    data = response.json()
                    df = pd.DataFrame(data)
                except json.JSONDecodeError:
                    raise QueryError(f"Unsupported response content type: {content_type}")

            logger.info(f"Loaded {len(df)} rows from API")
            return df

        except Exception as e:
            error_msg = f"Failed to parse API response: {e}"
            logger.error(error_msg)
            raise QueryError(error_msg) from e

    def load_data_streaming(self, query: Optional[str] = None, chunk_size: int = 1000, **kwargs) -> Iterator[pd.DataFrame]:
        """Load data with pagination support."""
        pagination_config = self.pagination

        if not pagination_config:
            # No pagination, fall back to single request
            yield self.load_data(query, **kwargs)
            return

        # Handle different pagination types
        pagination_type = pagination_config.get('type', 'offset')

        if pagination_type == 'offset':
            yield from self._paginate_offset(query, pagination_config, chunk_size, **kwargs)
        elif pagination_type == 'cursor':
            yield from self._paginate_cursor(query, pagination_config, **kwargs)
        elif pagination_type == 'page':
            yield from self._paginate_page(query, pagination_config, **kwargs)
        else:
            logger.warning(f"Unknown pagination type: {pagination_type}, falling back to single request")
            yield self.load_data(query, **kwargs)

    def _paginate_offset(self, query: Optional[str], config: Dict[str, Any], chunk_size: int, **kwargs) -> Iterator[pd.DataFrame]:
        """Handle offset-based pagination."""
        offset_param = config.get('offset_param', 'offset')
        limit_param = config.get('limit_param', 'limit')
        limit_value = config.get('limit_value', chunk_size)
        total_param = config.get('total_param', 'total')

        offset = 0
        while True:
            # Add pagination parameters
            paginated_params = kwargs.get('params', {}).copy()
            paginated_params[offset_param] = offset
            paginated_params[limit_param] = limit_value

            paginated_kwargs = kwargs.copy()
            paginated_kwargs['params'] = paginated_params

            df = self.load_data(query, **paginated_kwargs)

            if len(df) == 0:
                break

            yield df
            offset += limit_value

            # Check if we've reached the total
            if total_param in paginated_params and offset >= paginated_params[total_param]:
                break

    def _paginate_cursor(self, query: Optional[str], config: Dict[str, Any], **kwargs) -> Iterator[pd.DataFrame]:
        """Handle cursor-based pagination."""
        cursor_param = config.get('cursor_param', 'cursor')
        cursor = None

        while True:
            paginated_params = kwargs.get('params', {}).copy()
            if cursor:
                paginated_params[cursor_param] = cursor

            paginated_kwargs = kwargs.copy()
            paginated_kwargs['params'] = paginated_params

            df = self.load_data(query, **paginated_kwargs)

            if len(df) == 0:
                break

            yield df

            # Get next cursor (this would need to be extracted from response headers or body)
            # This is a simplified implementation
            cursor = df.attrs.get('next_cursor') if hasattr(df, 'attrs') else None
            if not cursor:
                break

    def _paginate_page(self, query: Optional[str], config: Dict[str, Any], **kwargs) -> Iterator[pd.DataFrame]:
        """Handle page-based pagination."""
        page_param = config.get('page_param', 'page')
        page = 1

        while True:
            paginated_params = kwargs.get('params', {}).copy()
            paginated_params[page_param] = page

            paginated_kwargs = kwargs.copy()
            paginated_kwargs['params'] = paginated_params

            df = self.load_data(query, **paginated_kwargs)

            if len(df) == 0:
                break

            yield df
            page += 1

    def get_schema(self) -> Dict[str, Any]:
        """Get API schema information."""
        # This is a basic implementation - could be extended for OpenAPI spec parsing
        return {
            'base_url': self.base_url,
            'endpoint': self.endpoint,
            'method': self.method,
            'auth_type': self.auth_type,
            'pagination': self.pagination
        }
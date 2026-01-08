"""
Scalability Error Classes

Custom exceptions for scalability and big data related errors.
"""

class ScalabilityError(Exception):
    """Base class for scalability-related errors."""
    pass

class SparkUnavailableError(ScalabilityError):
    """Raised when Spark is required but unavailable."""
    pass

class DistributedComputationError(ScalabilityError):
    """Raised when distributed computation fails."""
    pass
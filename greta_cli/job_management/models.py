"""
Job Status and Result Models
"""

from pydantic import BaseModel
from typing import Any, Optional

class JobStatus(BaseModel):
    """Model for job status information."""
    job_id: str
    status: str
    current: Optional[Any]

class JobResult(BaseModel):
    """Model for job result information."""
    job_id: str
    status: str
    result: Optional[Any]
    error: Optional[str]
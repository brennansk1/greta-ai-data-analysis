"""
Job Management Module

Handles asynchronous job submission and management using Celery and Redis.
"""

from typing import Dict, Any
from celery import Celery
import redis
from greta_cli.config import GretaConfig

celery_app = Celery('greta')

class JobManager:
    """
    Manages asynchronous analysis jobs using Celery.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", broker_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.broker_url = broker_url
        self.celery = celery_app
        self.celery.conf.broker_url = broker_url
        self.celery.conf.result_backend = redis_url
        self.redis_client = redis.from_url(redis_url)

    def submit_analysis_job(self, config: GretaConfig, data_path: str) -> str:
        """Submit analysis job asynchronously."""
        from .tasks import run_analysis_task
        config_dict = config.model_dump()
        task = run_analysis_task.delay(config_dict, data_path)
        return task.id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of running job."""
        from celery.result import AsyncResult
        result = AsyncResult(job_id, app=self.celery)
        return {
            "job_id": job_id,
            "status": result.status,
            "current": result.info if result.info else None
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel running job."""
        from celery.result import AsyncResult
        result = AsyncResult(job_id, app=self.celery)
        result.revoke(terminate=True)
        return True

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve completed job results."""
        from celery.result import AsyncResult
        result = AsyncResult(job_id, app=self.celery)
        if result.ready():
            return {
                "job_id": job_id,
                "status": "SUCCESS",
                "result": result.result
            }
        else:
            return self.get_job_status(job_id)
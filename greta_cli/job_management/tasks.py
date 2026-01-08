"""
Celery Tasks for Asynchronous Job Processing
"""

from .job_manager import celery_app

@celery_app.task(bind=True)
def run_analysis_task(self, config_dict: dict, data_path: str) -> dict:
    """Celery task for running analysis pipeline."""
    from greta_cli.config import GretaConfig
    from greta_cli.core_integration import integration

    config = GretaConfig(**config_dict)
    try:
        result = integration.run_analysis_pipeline(config, data_path)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
# Scalability & Big Data Infrastructure Feature Specification

## Overview

This document outlines the technical specification for implementing Feature 1: Scalability & Big Data Infrastructure. The feature introduces full Dask/Spark integration, distributed computation for genetic algorithm evaluation, and asynchronous job management to handle long-running analyses without blocking the CLI.

## Current Architecture Analysis

### Data Flow
The current pipeline follows a sequential process:
1. Data Ingestion (CSV/Excel → Pandas/Dask DataFrame)
2. Preprocessing (cleaning, feature engineering, selection)
3. Hypothesis Search (GA optimization)
4. Statistical Analysis
5. Narrative Generation

### Key Integration Points
- `greta_core/ingestion.py`: Already has basic Dask support with size-based switching
- `greta_core/hypothesis_search/optimizers.py`: GA optimizer has Dask distributed support
- `greta_cli/core_integration/integration.py`: Main pipeline orchestration
- `greta_cli/config/config.py`: Configuration management

## Technical Specification

### 1. Full Dask/Spark Integration for Data Ingestion

#### Modifications to `greta_core/ingestion.py`

**New Functions:**
```python
def load_spark_dataframe(file_path: str, **kwargs) -> pyspark.sql.DataFrame:
    """Load data using Spark for distributed processing."""

def should_use_spark(file_path: str, estimated_rows: Optional[int] = None) -> bool:
    """Determine if Spark should be used based on data size."""

def convert_to_dask_from_spark(spark_df: pyspark.sql.DataFrame) -> dd.DataFrame:
    """Convert Spark DataFrame to Dask for compatibility."""
```

**Enhanced DataFrame Type Union:**
```python
DataFrame = Union[pd.DataFrame, dd.DataFrame, pyspark.sql.DataFrame]
```

**Configuration Options:**
- `ingestion.backend`: "pandas" | "dask" | "spark" | "auto"
- `ingestion.spark_config`: Dict of Spark configuration parameters
- `ingestion.dask_config`: Dict of Dask configuration parameters

#### New Module: `greta_core/ingestion/spark_connector.py`
```python
class SparkConnector:
    def __init__(self, config: Dict[str, Any]):
        self.spark = None
        self.config = config

    def initialize_spark_session(self) -> pyspark.sql.SparkSession:
        """Create and configure Spark session."""

    def load_data(self, file_path: str, **kwargs) -> pyspark.sql.DataFrame:
        """Load data using Spark."""

    def convert_to_pandas(self, df: pyspark.sql.DataFrame) -> pd.DataFrame:
        """Convert Spark DF to Pandas for compatibility."""
```

### 2. Distributed Core Engine for GA Evaluation

#### Modifications to `greta_core/hypothesis_search/optimizers.py`

**Enhanced GeneticAlgorithmOptimizer:**
- Add `use_spark` parameter for Spark-based distributed evaluation
- Integrate with Spark MLlib for distributed GA operations
- Maintain backward compatibility with existing Dask/multiprocessing

**New Distributed Evaluation Methods:**
```python
def evaluate_with_spark(self, population: List[List[int]]) -> List[Tuple[float, ...]]:
    """Evaluate population using Spark distributed computing."""

def evaluate_with_dask_distributed(self, population: List[List[int]]) -> List[Tuple[float, ...]]:
    """Enhanced Dask distributed evaluation with cluster support."""
```

**Configuration Options:**
- `hypothesis_search.distributed_backend`: "multiprocessing" | "dask" | "spark" | "auto"
- `hypothesis_search.cluster_address`: Address for Dask/Spark cluster
- `hypothesis_search.max_workers`: Maximum parallel workers

### 3. Asynchronous Job Management System

#### New Module: `greta_cli/job_management/`

**`job_manager.py`:**
```python
from celery import Celery
import redis

class JobManager:
    def __init__(self, redis_url: str, broker_url: str):
        self.celery = Celery('greta', broker=broker_url, backend=redis_url)
        self.redis_client = redis.from_url(redis_url)

    def submit_analysis_job(self, config: GretaConfig, data_path: str) -> str:
        """Submit analysis job asynchronously."""

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of running job."""

    def cancel_job(self, job_id: str) -> bool:
        """Cancel running job."""

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve completed job results."""
```

**`tasks.py`:**
```python
from .job_manager import celery_app

@celery_app.task(bind=True)
def run_analysis_task(self, config_dict: Dict, data_path: str) -> Dict:
    """Celery task for running analysis pipeline."""
    # Implementation delegates to existing integration.run_analysis_pipeline
```

#### CLI Enhancements

**New Commands:**
- `greta submit <config.yml>`: Submit async analysis job
- `greta status <job_id>`: Check job status
- `greta results <job_id>`: Retrieve job results
- `greta cancel <job_id>`: Cancel running job
- `greta list`: List active/completed jobs

**Modified `run` Command:**
- Add `--async` flag to run analysis asynchronously
- Return job ID for async execution

### 4. Configuration Schema Updates

#### New Config Classes in `greta_cli/config/config.py`

```python
class ScalabilityConfig(BaseModel):
    """Scalability and big data configuration."""
    enabled: bool = Field(True, description="Enable scalability features")
    ingestion_backend: str = Field("auto", description="Data ingestion backend")
    distributed_backend: str = Field("auto", description="Distributed computation backend")
    async_processing: bool = Field(False, description="Enable async job processing")

    # Backend-specific configs
    dask_config: Dict[str, Any] = Field(default_factory=dict)
    spark_config: Dict[str, Any] = Field(default_factory=dict)
    celery_config: Dict[str, Any] = Field(default_factory=dict)

class GretaConfig(BaseModel):
    # ... existing fields ...
    scalability: ScalabilityConfig = ScalabilityConfig()
```

### 5. Dependencies to Add

Update `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "pyspark>=3.3.0",
    "celery>=5.3.0",
    "redis>=4.5.0",
    "flower>=2.0.0",  # Optional: Celery monitoring
]
```

### 6. New Modules Structure

```
greta_cli/
├── job_management/
│   ├── __init__.py
│   ├── job_manager.py
│   ├── tasks.py
│   └── models.py  # Job status/result models

greta_core/
├── ingestion/
│   ├── spark_connector.py
│   └── distributed_utils.py

├── hypothesis_search/
│   └── distributed_optimizers.py  # Spark-based GA
```

### 7. Integration Points

#### Core Integration Layer Updates (`greta_cli/core_integration/integration.py`)

**New Function:**
```python
def run_analysis_async(config: GretaConfig, data_path: str) -> str:
    """Run analysis asynchronously via job manager."""
    job_manager = JobManager(config.scalability.celery_config)
    return job_manager.submit_analysis_job(config, data_path)
```

**Modified `run_analysis_pipeline`:**
- Add checks for scalability features availability
- Graceful degradation when big data tools not available
- Support for different DataFrame types throughout pipeline

#### Backward Compatibility

**Graceful Degradation:**
- If Spark not available → Fall back to Dask
- If Dask not available → Fall back to Pandas + multiprocessing
- If Celery/Redis not available → Run synchronously
- Configuration validation with warnings for missing dependencies

**Feature Flags:**
- All new features disabled by default
- Clear error messages when dependencies missing
- Auto-detection of available backends

### 8. Error Handling and Monitoring

#### New Exception Classes
```python
class ScalabilityError(Exception):
    """Base class for scalability-related errors."""

class SparkUnavailableError(ScalabilityError):
    """Raised when Spark is required but unavailable."""

class DistributedComputationError(ScalabilityError):
    """Raised when distributed computation fails."""
```

#### Logging Enhancements
- Job progress tracking with unique IDs
- Performance metrics for distributed operations
- Resource usage monitoring (memory, CPU, network)

### 9. Testing Strategy

#### Unit Tests
- Mock Spark/Dask/Celery dependencies for CI
- Test graceful degradation scenarios
- Validate configuration parsing

#### Integration Tests
- End-to-end async job workflow
- Distributed GA evaluation accuracy
- Spark DataFrame processing pipeline

#### Performance Benchmarks
- Compare processing times across backends
- Memory usage analysis for large datasets
- Scalability testing with increasing data sizes

### 10. Migration Path

#### Phase 1: Core Integration
- Implement Dask/Spark ingestion
- Add distributed GA evaluation
- Maintain full backward compatibility

#### Phase 2: Async Processing
- Add Celery/Redis job management
- Update CLI with async commands
- Add job status/result retrieval

#### Phase 3: Production Readiness
- Comprehensive testing
- Documentation updates
- Performance optimization

## Implementation Timeline

1. **Week 1-2**: Dask/Spark ingestion integration
2. **Week 3-4**: Distributed GA evaluation
3. **Week 5-6**: Async job management system
4. **Week 7-8**: Configuration, testing, and documentation

## Risk Assessment

### Technical Risks
- **Dependency Conflicts**: Spark/PySpark version compatibility
- **Performance Overhead**: Distributed systems may have higher latency
- **Resource Management**: Proper cleanup of distributed resources

### Mitigation Strategies
- Comprehensive testing with multiple backend combinations
- Graceful degradation with clear error messages
- Resource monitoring and automatic cleanup
- Extensive documentation for deployment configurations

## Success Criteria

- Maintain <5% performance regression for small datasets
- Support datasets >100GB with Spark backend
- Async jobs complete without blocking CLI
- Full backward compatibility with existing configurations
- Clear error messages and graceful degradation
import logging
import time
import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import json

from celery import Celery, Task
from celery.result import AsyncResult
from redis import Redis

from .priority_manager import Priority, PriorityManager, get_priority_manager
from ..resources.resource_pool import ResourcePoolManager, PoolType
from ..utils.exceptions import (
    JobSubmissionError,
    JobExecutionError,
    JobNotFoundError,
    JobTimeoutError
)

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class JobConfig:
    """Configuration for a training job"""
    job_id: str
    user_id: str
    job_type: str  # e.g., "fine_tuning", "inference"

    # Resource requirements
    num_gpus: int = 1
    pool_type: str = "development"  # "development" or "production"
    is_preemptible: bool = False

    # Training parameters
    model_name: str = "bert-base-uncased"
    dataset_path: str = ""
    output_dir: str = "./output"

    # Job metadata
    priority: str = "MEDIUM"
    estimated_duration: int = 3600
    max_retries: int = 3
    timeout: int = 86400  # 24 hours

    # Additional config
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class JobResult:
    """Result of job execution"""
    job_id: str
    status: JobStatus
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: Optional[float] = None
    output: Optional[Dict] = None
    error: Optional[str] = None
    retry_count: int = 0


class MLPlatformTask(Task):
    """
    Custom Celery task class for ML platform
    Handles resource allocation and cleanup
    """

    def __init__(self):
        super().__init__()
        self.resource_manager = None
        self.allocation = None

    def before_start(self, task_id, args, kwargs):
        """Called before task execution"""
        logger.info(f"Task {task_id} starting")

    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion"""
        logger.info(f"Task {task_id} completed successfully")
        if self.allocation:
            self._release_resources()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        if self.allocation:
            self._release_resources()

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry"""
        logger.warning(f"Task {task_id} retrying: {exc}")
        if self.allocation:
            self._release_resources()

    def _release_resources(self):
        """Release allocated resources"""
        if self.allocation and self.resource_manager:
            try:
                pool = self.resource_manager.get_pool(self.allocation.pool_name)
                if pool:
                    pool.release(self.allocation.job_id)
                logger.info(f"Released resources for job {self.allocation.job_id}")
            except Exception as e:
                logger.error(f"Failed to release resources: {e}")


# Initialize Celery app
def create_celery_app(
    broker_url: str = None,
    backend_url: str = None
) -> Celery:
    """
    Create and configure Celery application

    Args:
        broker_url: Redis broker URL
        backend_url: Redis backend URL

    Returns:
        Configured Celery app
    """
    if broker_url is None:
        broker_url = os.getenv(
            "CELERY_BROKER_URL",
            "redis://localhost:6379/0"
        )

    if backend_url is None:
        backend_url = os.getenv(
            "CELERY_RESULT_BACKEND",
            "redis://localhost:6379/0"
        )

    app = Celery(
        "ml_platform",
        broker=broker_url,
        backend=backend_url
    )

    # Configure Celery
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=86400,  # 24 hours
        task_soft_time_limit=86000,
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=10,
    )

    return app


# Global Celery app instance
celery_app = create_celery_app()


@celery_app.task(base=MLPlatformTask, bind=True, name="ml_platform.train")
def train_model_task(self, job_config_dict: Dict) -> Dict:
    """
    Celery task for model training

    Args:
        job_config_dict: Job configuration dictionary

    Returns:
        Job result dictionary
    """
    job_config = JobConfig(**job_config_dict)
    job_id = job_config.job_id

    logger.info(f"Starting training job {job_id}")

    started_at = time.time()
    result = JobResult(
        job_id=job_id,
        status=JobStatus.RUNNING,
        started_at=started_at
    )

    try:
        # Import here to avoid circular dependencies
        from ..pipeline.trainer import Trainer
        from ..resources.gpu_manager import get_gpu_manager

        # Get GPU manager
        gpu_manager = get_gpu_manager()

        # Allocate GPUs for this job
        allocation = gpu_manager.allocate_gpus(
            job_id=job_id,
            num_gpus=job_config.num_gpus
        )

        # Set visible devices
        gpu_manager.set_cuda_visible_devices(allocation.gpu_ids)

        # Store allocation for cleanup
        self.allocation = allocation

        logger.info(
            f"Job {job_id} allocated GPUs: {allocation.gpu_ids}"
        )

        # Create and run trainer
        trainer = Trainer(
            model_name=job_config.model_name,
            output_dir=job_config.output_dir,
            num_gpus=job_config.num_gpus,
            config=job_config.config
        )

        # Run training
        training_result = trainer.train(
            dataset_path=job_config.dataset_path,
            max_steps=job_config.config.get("max_steps"),
            save_checkpoints=True
        )

        # Update result
        completed_at = time.time()
        result.status = JobStatus.COMPLETED
        result.completed_at = completed_at
        result.duration = completed_at - started_at
        result.output = training_result

        logger.info(
            f"Job {job_id} completed in {result.duration:.1f}s"
        )

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        completed_at = time.time()
        result.status = JobStatus.FAILED
        result.completed_at = completed_at
        result.duration = completed_at - started_at
        result.error = str(e)

        raise JobExecutionError(job_id, str(e))

    return asdict(result)


class JobQueue:
    """
    Manages job queue and submission
    Integrates with Celery and priority manager
    """

    def __init__(
        self,
        celery_app: Celery,
        priority_manager: Optional[PriorityManager] = None,
        redis_url: str = "redis://localhost:6379/0"
    ):
        """
        Initialize job queue

        Args:
            celery_app: Celery application
            priority_manager: Priority manager instance
            redis_url: Redis connection URL
        """
        self.celery_app = celery_app
        self.priority_manager = priority_manager or get_priority_manager()

        # Redis for job metadata storage
        self.redis_client = Redis.from_url(redis_url)

        logger.info("Job queue initialized")

    def submit_job(
        self,
        job_config: JobConfig,
        priority: Priority = Priority.MEDIUM
    ) -> str:
        """
        Submit a job to the queue

        Args:
            job_config: Job configuration
            priority: Job priority

        Returns:
            Job ID

        Raises:
            JobSubmissionError: If submission fails
        """
        try:
            # Submit to priority manager
            self.priority_manager.submit_job(
                job_id=job_config.job_id,
                user_id=job_config.user_id,
                num_gpus=job_config.num_gpus,
                priority=priority,
                estimated_duration=job_config.estimated_duration
            )

            # Store job config in Redis
            self._store_job_config(job_config)

            # Submit to Celery with routing based on priority
            queue_name = self._get_queue_name(priority)

            task = train_model_task.apply_async(
                args=[asdict(job_config)],
                queue=queue_name,
                task_id=job_config.job_id,
                countdown=0,  # Execute immediately when picked up
                expires=job_config.timeout
            )

            logger.info(
                f"Submitted job {job_config.job_id} to queue '{queue_name}' "
                f"(priority={priority.name})"
            )

            # Update job status
            self._update_job_status(job_config.job_id, JobStatus.QUEUED)

            return job_config.job_id

        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise JobSubmissionError(f"Job submission failed: {e}")

    def _get_queue_name(self, priority: Priority) -> str:
        """Get Celery queue name based on priority"""
        if priority == Priority.HIGH:
            return "high_priority"
        elif priority == Priority.MEDIUM:
            return "default"
        else:
            return "low_priority"

    def _store_job_config(self, job_config: JobConfig):
        """Store job configuration in Redis"""
        key = f"job_config:{job_config.job_id}"
        self.redis_client.setex(
            key,
            86400 * 7,  # 7 days expiry
            json.dumps(asdict(job_config))
        )

    def _update_job_status(self, job_id: str, status: JobStatus):
        """Update job status in Redis"""
        key = f"job_status:{job_id}"
        self.redis_client.setex(
            key,
            86400 * 7,  # 7 days expiry
            status.value
        )

    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get current job status

        Args:
            job_id: Job identifier

        Returns:
            JobStatus
        """
        # Check Redis first
        key = f"job_status:{job_id}"
        status_str = self.redis_client.get(key)

        if status_str:
            return JobStatus(status_str.decode())

        # Check Celery task state
        task_result = AsyncResult(job_id, app=self.celery_app)

        state_mapping = {
            "PENDING": JobStatus.PENDING,
            "STARTED": JobStatus.RUNNING,
            "SUCCESS": JobStatus.COMPLETED,
            "FAILURE": JobStatus.FAILED,
            "RETRY": JobStatus.RUNNING,
            "REVOKED": JobStatus.CANCELLED
        }

        return state_mapping.get(task_result.state, JobStatus.PENDING)

    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """
        Get job result

        Args:
            job_id: Job identifier

        Returns:
            JobResult or None if not completed
        """
        task_result = AsyncResult(job_id, app=self.celery_app)

        if task_result.ready():
            try:
                result_dict = task_result.get()
                return JobResult(**result_dict)
            except Exception as e:
                logger.error(f"Failed to get result for job {job_id}: {e}")
                return None

        return None

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled successfully
        """
        try:
            # Cancel in priority manager
            self.priority_manager.cancel_job(job_id)

            # Revoke Celery task
            self.celery_app.control.revoke(job_id, terminate=True)

            # Update status
            self._update_job_status(job_id, JobStatus.CANCELLED)

            logger.info(f"Cancelled job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            "priority_queue": self.priority_manager.get_queue_summary(),
            "celery_active": self._get_celery_stats()
        }

    def _get_celery_stats(self) -> Dict:
        """Get Celery worker statistics"""
        try:
            inspect = self.celery_app.control.inspect()
            active = inspect.active()
            scheduled = inspect.scheduled()

            return {
                "active_tasks": sum(len(tasks) for tasks in (active or {}).values()),
                "scheduled_tasks": sum(len(tasks) for tasks in (scheduled or {}).values())
            }
        except Exception as e:
            logger.error(f"Failed to get Celery stats: {e}")
            return {"error": str(e)}


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get global job queue instance"""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(celery_app)
    return _job_queue


def submit_job_cli():
    """CLI entry point for job submission"""
    import click

    @click.command()
    @click.option("--job-id", required=True, help="Unique job identifier")
    @click.option("--user-id", required=True, help="User identifier")
    @click.option("--model", default="bert-base-uncased", help="Model name")
    @click.option("--dataset", required=True, help="Dataset path")
    @click.option("--output-dir", default="./output", help="Output directory")
    @click.option("--num-gpus", default=1, type=int, help="Number of GPUs")
    @click.option("--priority", type=click.Choice(["LOW", "MEDIUM", "HIGH"]),
                  default="MEDIUM", help="Job priority")
    def submit(job_id, user_id, model, dataset, output_dir, num_gpus, priority):
        """Submit a training job to the queue"""

        job_config = JobConfig(
            job_id=job_id,
            user_id=user_id,
            job_type="fine_tuning",
            model_name=model,
            dataset_path=dataset,
            output_dir=output_dir,
            num_gpus=num_gpus
        )

        queue = get_job_queue()
        priority_enum = Priority[priority]

        try:
            job_id = queue.submit_job(job_config, priority_enum)
            click.echo(f"Job submitted successfully: {job_id}")
        except Exception as e:
            click.echo(f"Failed to submit job: {e}", err=True)
            raise click.Abort()

    submit()

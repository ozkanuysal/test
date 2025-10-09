import logging
import signal
import sys
from typing import Optional
import os

from celery import Celery
from celery.signals import (
    worker_ready,
    worker_shutdown,
    task_prerun,
    task_postrun,
    task_failure
)

from .job_queue import celery_app
from ..resources.gpu_manager import get_gpu_manager
from ..resources.health_checker import get_health_checker
from ..monitoring.logger import setup_logging
from ..utils.config import get_config_manager

logger = logging.getLogger(__name__)


class WorkerManager:
    """
    Manages Celery worker lifecycle and resources
    Coordinates with GPU manager and health checker
    """

    def __init__(
        self,
        app: Celery,
        worker_name: Optional[str] = None,
        concurrency: int = 1,
        queues: Optional[list] = None
    ):
        """
        Initialize worker manager

        Args:
            app: Celery application
            worker_name: Worker identifier
            concurrency: Number of concurrent tasks
            queues: List of queues to consume from
        """
        self.app = app
        self.worker_name = worker_name or f"worker-{os.getpid()}"
        self.concurrency = concurrency
        self.queues = queues or ["default", "high_priority", "low_priority"]

        self.gpu_manager = get_gpu_manager()
        self.health_checker = get_health_checker()

        self._setup_signal_handlers()
        self._setup_celery_signals()

        logger.info(
            f"Worker manager initialized: {self.worker_name} "
            f"(concurrency={concurrency}, queues={queues})"
        )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _setup_celery_signals(self):
        """Setup Celery signal handlers"""

        @worker_ready.connect
        def on_worker_ready(sender, **kwargs):
            """Called when worker is ready"""
            logger.info(f"Worker {self.worker_name} is ready")

            # Start health monitoring
            if not self.health_checker._running:
                self.health_checker.start_monitoring()

            # Log GPU information
            gpu_info = self.gpu_manager.get_all_gpu_info()
            logger.info(f"Available GPUs: {len(gpu_info)}")
            for info in gpu_info:
                logger.info(
                    f"  GPU {info.id}: {info.name} "
                    f"({info.total_memory / (1024**3):.1f}GB)"
                )

        @worker_shutdown.connect
        def on_worker_shutdown(sender, **kwargs):
            """Called when worker is shutting down"""
            logger.info(f"Worker {self.worker_name} shutting down")

            # Stop health monitoring
            self.health_checker.stop_monitoring()

            # Release any remaining allocations
            allocations = self.gpu_manager.get_all_allocations()
            for job_id in list(allocations.keys()):
                try:
                    self.gpu_manager.release_gpus(job_id)
                    logger.info(f"Released GPUs for job {job_id}")
                except Exception as e:
                    logger.error(f"Failed to release GPUs for job {job_id}: {e}")

        @task_prerun.connect
        def on_task_prerun(sender, task_id, task, args, kwargs, **extra):
            """Called before task execution"""
            logger.info(f"Starting task {task.name} (id={task_id})")

            # Check GPU health before starting task
            unhealthy_gpus = []
            for gpu_id in range(self.gpu_manager.num_gpus):
                if not self.health_checker.is_gpu_healthy(gpu_id):
                    unhealthy_gpus.append(gpu_id)

            if unhealthy_gpus:
                logger.warning(
                    f"Unhealthy GPUs detected before task {task_id}: "
                    f"{unhealthy_gpus}"
                )

        @task_postrun.connect
        def on_task_postrun(sender, task_id, task, args, kwargs, retval, **extra):
            """Called after task execution"""
            logger.info(f"Completed task {task.name} (id={task_id})")

        @task_failure.connect
        def on_task_failure(sender, task_id, exception, args, kwargs,
                           traceback, einfo, **extra):
            """Called on task failure"""
            logger.error(
                f"Task {sender.name} (id={task_id}) failed: {exception}",
                exc_info=einfo
            )

    def start(self):
        """Start the Celery worker"""
        logger.info(f"Starting worker {self.worker_name}")

        # Build worker arguments
        worker_args = [
            "worker",
            f"--hostname={self.worker_name}@%h",
            f"--concurrency={self.concurrency}",
            f"--queues={','.join(self.queues)}",
            "--loglevel=INFO",
            "--pool=prefork",  # Use prefork pool for better isolation
            "--autoscale=10,1",  # Auto-scale between 1 and 10 workers
            "--max-tasks-per-child=10",  # Restart worker after N tasks
        ]

        # Start worker
        self.app.worker_main(worker_args)

    def shutdown(self):
        """Gracefully shutdown the worker"""
        logger.info(f"Shutting down worker {self.worker_name}")

        # Stop accepting new tasks
        self.app.control.cancel_consumer(self.queues, destination=[self.worker_name])

        # Wait for running tasks to complete (handled by Celery)
        # Cleanup is handled by worker_shutdown signal


def main():
    """Main entry point for worker"""
    import click

    @click.command()
    @click.option(
        "--name",
        default=None,
        help="Worker name (defaults to worker-<pid>)"
    )
    @click.option(
        "--concurrency",
        default=1,
        type=int,
        help="Number of concurrent tasks"
    )
    @click.option(
        "--queues",
        default="default,high_priority,low_priority",
        help="Comma-separated list of queues"
    )
    @click.option(
        "--config",
        default="default",
        help="Configuration file name"
    )
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
        help="Logging level"
    )
    def start_worker(name, concurrency, queues, config, log_level):
        """Start a Celery worker for the ML platform"""

        # Setup logging
        setup_logging(level=log_level)

        # Load configuration
        config_manager = get_config_manager()
        try:
            config_manager.load_config(config)
            logger.info(f"Loaded configuration: {config}")
        except Exception as e:
            logger.warning(f"Failed to load config '{config}': {e}")
            logger.info("Using default configuration")

        # Parse queues
        queue_list = [q.strip() for q in queues.split(",")]

        # Create and start worker
        worker = WorkerManager(
            app=celery_app,
            worker_name=name,
            concurrency=concurrency,
            queues=queue_list
        )

        try:
            worker.start()
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
            worker.shutdown()
        except Exception as e:
            logger.error(f"Worker failed: {e}", exc_info=True)
            worker.shutdown()
            raise

    start_worker()


if __name__ == "__main__":
    main()

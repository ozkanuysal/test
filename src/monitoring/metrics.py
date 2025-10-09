import logging
import time
from typing import Dict, Optional
from threading import Thread
import os

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    start_http_server,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from ..resources.gpu_manager import get_gpu_manager
from ..resources.health_checker import get_health_checker, HealthStatus
from ..scheduler.priority_manager import get_priority_manager

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and exports metrics for the ML platform
    Integrates with Prometheus for monitoring
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector

        Args:
            registry: Prometheus registry (uses default if None)
        """
        self.registry = registry or CollectorRegistry()

        # GPU Metrics
        self.gpu_utilization = Gauge(
            "ml_platform_gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id", "gpu_name"],
            registry=self.registry
        )

        self.gpu_memory_used = Gauge(
            "ml_platform_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["gpu_id", "gpu_name"],
            registry=self.registry
        )

        self.gpu_memory_total = Gauge(
            "ml_platform_gpu_memory_total_bytes",
            "GPU total memory in bytes",
            ["gpu_id", "gpu_name"],
            registry=self.registry
        )

        self.gpu_temperature = Gauge(
            "ml_platform_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["gpu_id", "gpu_name"],
            registry=self.registry
        )

        self.gpu_power_usage = Gauge(
            "ml_platform_gpu_power_watts",
            "GPU power usage in watts",
            ["gpu_id", "gpu_name"],
            registry=self.registry
        )

        self.gpu_health_status = Gauge(
            "ml_platform_gpu_health_status",
            "GPU health status (1=healthy, 0.5=degraded, 0=unhealthy)",
            ["gpu_id", "gpu_name"],
            registry=self.registry
        )

        # Job Metrics
        self.jobs_submitted_total = Counter(
            "ml_platform_jobs_submitted_total",
            "Total number of jobs submitted",
            ["user_id", "priority"],
            registry=self.registry
        )

        self.jobs_completed_total = Counter(
            "ml_platform_jobs_completed_total",
            "Total number of jobs completed",
            ["user_id", "status"],
            registry=self.registry
        )

        self.jobs_running = Gauge(
            "ml_platform_jobs_running",
            "Number of currently running jobs",
            registry=self.registry
        )

        self.jobs_queued = Gauge(
            "ml_platform_jobs_queued",
            "Number of jobs in queue",
            ["priority"],
            registry=self.registry
        )

        self.job_duration_seconds = Histogram(
            "ml_platform_job_duration_seconds",
            "Job execution duration in seconds",
            ["job_type", "user_id"],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400],
            registry=self.registry
        )

        self.job_wait_time_seconds = Histogram(
            "ml_platform_job_wait_time_seconds",
            "Job queue wait time in seconds",
            ["priority"],
            buckets=[1, 5, 10, 30, 60, 300, 900, 1800, 3600],
            registry=self.registry
        )

        self.job_gpu_hours = Summary(
            "ml_platform_job_gpu_hours",
            "GPU-hours consumed by jobs",
            ["user_id"],
            registry=self.registry
        )

        # Resource Metrics
        self.gpus_allocated = Gauge(
            "ml_platform_gpus_allocated",
            "Number of GPUs currently allocated",
            registry=self.registry
        )

        self.gpus_available = Gauge(
            "ml_platform_gpus_available",
            "Number of GPUs available for allocation",
            registry=self.registry
        )

        # System Metrics
        self.worker_count = Gauge(
            "ml_platform_workers_active",
            "Number of active Celery workers",
            registry=self.registry
        )

        self.collection_errors_total = Counter(
            "ml_platform_metrics_collection_errors_total",
            "Total number of metrics collection errors",
            ["component"],
            registry=self.registry
        )

        # Get manager instances
        self.gpu_manager = get_gpu_manager()
        self.health_checker = get_health_checker()
        self.priority_manager = get_priority_manager()

        # Collection state
        self._collection_thread: Optional[Thread] = None
        self._running = False
        self._collection_interval = 30  # seconds

        logger.info("Metrics collector initialized")

    def collect_gpu_metrics(self):
        """Collect GPU-related metrics"""
        try:
            gpu_infos = self.gpu_manager.get_all_gpu_info()

            for info in gpu_infos:
                labels = {"gpu_id": str(info.id), "gpu_name": info.name}

                # Utilization
                self.gpu_utilization.labels(**labels).set(info.utilization)

                # Memory
                self.gpu_memory_used.labels(**labels).set(info.used_memory)
                self.gpu_memory_total.labels(**labels).set(info.total_memory)

                # Temperature
                if info.temperature is not None:
                    self.gpu_temperature.labels(**labels).set(info.temperature)

                # Power
                if info.power_usage is not None:
                    self.gpu_power_usage.labels(**labels).set(info.power_usage)

                # Health status
                health_value = 1.0 if info.is_healthy else 0.0
                self.gpu_health_status.labels(**labels).set(health_value)

            # Allocation metrics
            utilization = self.gpu_manager.get_utilization_summary()
            self.gpus_allocated.set(utilization["allocated_gpus"])
            self.gpus_available.set(utilization["available_gpus"])

        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
            self.collection_errors_total.labels(component="gpu").inc()

    def collect_health_metrics(self):
        """Collect health-related metrics"""
        try:
            health_summary = self.health_checker.get_health_summary()

            for gpu_id, gpu_health in health_summary["gpus"].items():
                if gpu_health.get("status") == "unknown":
                    continue

                # Get GPU name
                try:
                    info = self.gpu_manager.get_gpu_info(int(gpu_id))
                    gpu_name = info.name
                except:
                    gpu_name = "unknown"

                labels = {"gpu_id": str(gpu_id), "gpu_name": gpu_name}

                # Map status to numeric value
                status = gpu_health["status"]
                if status == "healthy":
                    health_value = 1.0
                elif status == "degraded":
                    health_value = 0.5
                else:  # unhealthy
                    health_value = 0.0

                self.gpu_health_status.labels(**labels).set(health_value)

        except Exception as e:
            logger.error(f"Error collecting health metrics: {e}")
            self.collection_errors_total.labels(component="health").inc()

    def collect_queue_metrics(self):
        """Collect job queue metrics"""
        try:
            queue_summary = self.priority_manager.get_queue_summary()

            # Queue size by priority
            priority_breakdown = queue_summary.get("priority_breakdown", {})
            for priority, count in priority_breakdown.items():
                self.jobs_queued.labels(priority=priority.lower()).set(count)

        except Exception as e:
            logger.error(f"Error collecting queue metrics: {e}")
            self.collection_errors_total.labels(component="queue").inc()

    def record_job_submitted(self, user_id: str, priority: str):
        """
        Record job submission

        Args:
            user_id: User who submitted the job
            priority: Job priority level
        """
        self.jobs_submitted_total.labels(
            user_id=user_id,
            priority=priority.lower()
        ).inc()

    def record_job_completed(
        self,
        user_id: str,
        status: str,
        duration: float,
        job_type: str = "training",
        num_gpus: int = 1
    ):
        """
        Record job completion

        Args:
            user_id: User who ran the job
            status: Job completion status
            duration: Job duration in seconds
            job_type: Type of job
            num_gpus: Number of GPUs used
        """
        # Completion counter
        self.jobs_completed_total.labels(
            user_id=user_id,
            status=status.lower()
        ).inc()

        # Duration histogram
        self.job_duration_seconds.labels(
            job_type=job_type,
            user_id=user_id
        ).observe(duration)

        # GPU-hours
        gpu_hours = (num_gpus * duration) / 3600.0
        self.job_gpu_hours.labels(user_id=user_id).observe(gpu_hours)

    def record_job_wait_time(self, priority: str, wait_time: float):
        """
        Record job wait time in queue

        Args:
            priority: Job priority
            wait_time: Wait time in seconds
        """
        self.job_wait_time_seconds.labels(
            priority=priority.lower()
        ).observe(wait_time)

    def collect_all_metrics(self):
        """Collect all metrics"""
        logger.debug("Collecting metrics...")

        self.collect_gpu_metrics()
        self.collect_health_metrics()
        self.collect_queue_metrics()

        logger.debug("Metrics collection completed")

    def start_collection(self, interval: int = 30):
        """
        Start background metrics collection

        Args:
            interval: Collection interval in seconds
        """
        if self._running:
            logger.warning("Metrics collection already running")
            return

        self._collection_interval = interval
        self._running = True

        def collection_loop():
            logger.info(f"Starting metrics collection (interval={interval}s)")
            while self._running:
                try:
                    self.collect_all_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")

                time.sleep(self._collection_interval)

            logger.info("Metrics collection stopped")

        self._collection_thread = Thread(
            target=collection_loop,
            daemon=True,
            name="MetricsCollector"
        )
        self._collection_thread.start()

    def stop_collection(self):
        """Stop background metrics collection"""
        if not self._running:
            return

        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=10)

        logger.info("Stopped metrics collection")

    def get_metrics_text(self) -> str:
        """
        Get metrics in Prometheus text format

        Returns:
            Metrics text
        """
        return generate_latest(self.registry).decode("utf-8")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def start_metrics_server(port: int = 9090):
    """
    Start Prometheus metrics HTTP server

    Args:
        port: Port to listen on
    """
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise


def main():
    """Main entry point for standalone metrics server"""
    import click

    @click.command()
    @click.option(
        "--port",
        default=9090,
        type=int,
        help="Port for Prometheus metrics endpoint"
    )
    @click.option(
        "--interval",
        default=30,
        type=int,
        help="Metrics collection interval in seconds"
    )
    def run_metrics_server(port, interval):
        """Run standalone Prometheus metrics server"""

        from .logger import setup_logging
        setup_logging(level="INFO")

        # Get metrics collector
        collector = get_metrics_collector()

        # Start collection
        collector.start_collection(interval=interval)

        # Start HTTP server
        start_metrics_server(port=port)

        logger.info(
            f"Metrics server running at http://0.0.0.0:{port}/metrics "
            f"(collection interval: {interval}s)"
        )

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down metrics server")
            collector.stop_collection()

    run_metrics_server()


if __name__ == "__main__":
    main()

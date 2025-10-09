import logging
import time
import threading
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

from .gpu_manager import GPUManager, GPUInfo, get_gpu_manager
from ..utils.exceptions import ResourceException

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """GPU health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check"""
    gpu_id: int
    status: HealthStatus
    timestamp: float
    temperature: Optional[float] = None
    utilization: Optional[float] = None
    memory_used_percent: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class HealthThresholds:
    """Thresholds for health checks"""
    max_temperature: float = 85.0  # Celsius
    max_memory_percent: float = 95.0
    min_memory_free: int = 1024 * 1024 * 1024  # 1GB in bytes
    max_utilization: float = 100.0


class HealthChecker:
    """
    Monitors GPU health and handles failover scenarios
    Runs periodic health checks and triggers alerts
    """

    def __init__(
        self,
        gpu_manager: Optional[GPUManager] = None,
        check_interval: int = 60,
        thresholds: Optional[HealthThresholds] = None
    ):
        """
        Initialize health checker

        Args:
            gpu_manager: GPU manager instance
            check_interval: Seconds between health checks
            thresholds: Health check thresholds
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.check_interval = check_interval
        self.thresholds = thresholds or HealthThresholds()

        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._health_history: Dict[int, List[HealthCheck]] = {}
        self._alert_callbacks: List[Callable[[HealthCheck], None]] = []
        self._lock = threading.Lock()

        # Track consecutive failures for each GPU
        self._consecutive_failures: Dict[int, int] = {}
        self._max_consecutive_failures = 3

        logger.info(f"Health checker initialized with {check_interval}s interval")

    def register_alert_callback(self, callback: Callable[[HealthCheck], None]):
        """
        Register callback for health alerts

        Args:
            callback: Function to call when health issue detected
        """
        self._alert_callbacks.append(callback)
        logger.info(f"Registered health alert callback: {callback.__name__}")

    def check_gpu_health(self, gpu_id: int) -> HealthCheck:
        """
        Check health of a specific GPU

        Args:
            gpu_id: GPU device ID

        Returns:
            HealthCheck result
        """
        timestamp = time.time()

        try:
            info: GPUInfo = self.gpu_manager.get_gpu_info(gpu_id)

            # Determine health status
            status = HealthStatus.HEALTHY
            error_messages = []

            # Check temperature
            if info.temperature is not None:
                if info.temperature > self.thresholds.max_temperature:
                    status = HealthStatus.UNHEALTHY
                    error_messages.append(
                        f"Temperature too high: {info.temperature}°C > "
                        f"{self.thresholds.max_temperature}°C"
                    )
                elif info.temperature > self.thresholds.max_temperature - 10:
                    status = HealthStatus.DEGRADED
                    error_messages.append(
                        f"Temperature elevated: {info.temperature}°C"
                    )

            # Check memory
            memory_used_percent = (info.used_memory / info.total_memory * 100) \
                if info.total_memory > 0 else 0

            if memory_used_percent > self.thresholds.max_memory_percent:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                error_messages.append(
                    f"Memory usage high: {memory_used_percent:.1f}%"
                )

            if info.free_memory < self.thresholds.min_memory_free:
                status = HealthStatus.UNHEALTHY
                error_messages.append(
                    f"Low free memory: {info.free_memory / (1024**3):.2f}GB"
                )

            # Check if GPU reports as healthy
            if not info.is_healthy:
                status = HealthStatus.UNHEALTHY
                error_messages.append("GPU reported as unhealthy")

            health_check = HealthCheck(
                gpu_id=gpu_id,
                status=status,
                timestamp=timestamp,
                temperature=info.temperature,
                utilization=info.utilization,
                memory_used_percent=memory_used_percent,
                error_message="; ".join(error_messages) if error_messages else None
            )

            # Update consecutive failures counter
            if status == HealthStatus.UNHEALTHY:
                self._consecutive_failures[gpu_id] = \
                    self._consecutive_failures.get(gpu_id, 0) + 1
            else:
                self._consecutive_failures[gpu_id] = 0

            return health_check

        except Exception as e:
            logger.error(f"Health check failed for GPU {gpu_id}: {e}")
            self._consecutive_failures[gpu_id] = \
                self._consecutive_failures.get(gpu_id, 0) + 1

            return HealthCheck(
                gpu_id=gpu_id,
                status=HealthStatus.UNKNOWN,
                timestamp=timestamp,
                error_message=str(e)
            )

    def check_all_gpus(self) -> List[HealthCheck]:
        """
        Check health of all GPUs

        Returns:
            List of HealthCheck results
        """
        results = []

        for gpu_id in range(self.gpu_manager.num_gpus):
            health_check = self.check_gpu_health(gpu_id)
            results.append(health_check)

            # Store in history
            with self._lock:
                if gpu_id not in self._health_history:
                    self._health_history[gpu_id] = []

                history = self._health_history[gpu_id]
                history.append(health_check)

                # Keep only last 100 checks
                if len(history) > 100:
                    history.pop(0)

            # Trigger alerts if needed
            if health_check.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                self._trigger_alert(health_check)

            # Check for persistent failures
            if self._consecutive_failures.get(gpu_id, 0) >= self._max_consecutive_failures:
                logger.critical(
                    f"GPU {gpu_id} has failed {self._consecutive_failures[gpu_id]} "
                    f"consecutive health checks"
                )
                self._handle_gpu_failure(gpu_id)

        return results

    def _trigger_alert(self, health_check: HealthCheck):
        """
        Trigger alert callbacks for health issue

        Args:
            health_check: Health check that triggered alert
        """
        logger.warning(
            f"Health alert for GPU {health_check.gpu_id}: "
            f"{health_check.status.value} - {health_check.error_message}"
        )

        for callback in self._alert_callbacks:
            try:
                callback(health_check)
            except Exception as e:
                logger.error(f"Alert callback {callback.__name__} failed: {e}")

    def _handle_gpu_failure(self, gpu_id: int):
        """
        Handle persistent GPU failure

        Args:
            gpu_id: Failed GPU ID
        """
        logger.critical(f"Handling GPU {gpu_id} failure - initiating failover")

        # Get allocations on this GPU
        allocations = self.gpu_manager.get_all_allocations()

        failed_jobs = []
        for job_id, allocation in allocations.items():
            if gpu_id in allocation.gpu_ids:
                failed_jobs.append(job_id)

        if failed_jobs:
            logger.error(
                f"GPU {gpu_id} failure affects {len(failed_jobs)} jobs: {failed_jobs}"
            )

            # In a production system, we would:
            # 1. Checkpoint running jobs
            # 2. Release allocations
            # 3. Migrate jobs to healthy GPUs
            # 4. Notify job owners

            for job_id in failed_jobs:
                logger.warning(
                    f"Job {job_id} needs migration due to GPU {gpu_id} failure"
                )

    def start_monitoring(self):
        """Start background health monitoring"""
        if self._running:
            logger.warning("Health monitoring already running")
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._check_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop background health monitoring"""
        if not self._running:
            return

        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=10)
        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("Health monitoring loop started")

        while self._running:
            try:
                results = self.check_all_gpus()

                # Log summary
                unhealthy = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
                degraded = sum(1 for r in results if r.status == HealthStatus.DEGRADED)

                if unhealthy > 0 or degraded > 0:
                    logger.warning(
                        f"Health check: {unhealthy} unhealthy, {degraded} degraded GPUs"
                    )
                else:
                    logger.debug("Health check: All GPUs healthy")

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            # Wait for next check interval
            time.sleep(self.check_interval)

        logger.info("Health monitoring loop stopped")

    def get_health_summary(self) -> Dict:
        """
        Get summary of GPU health status

        Returns:
            Dictionary with health summary
        """
        with self._lock:
            summary = {
                "total_gpus": self.gpu_manager.num_gpus,
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0,
                "unknown": 0,
                "gpus": {}
            }

            for gpu_id in range(self.gpu_manager.num_gpus):
                if gpu_id in self._health_history and self._health_history[gpu_id]:
                    latest = self._health_history[gpu_id][-1]
                    status = latest.status.value

                    summary["gpus"][gpu_id] = {
                        "status": status,
                        "temperature": latest.temperature,
                        "utilization": latest.utilization,
                        "memory_used_percent": latest.memory_used_percent,
                        "error": latest.error_message,
                        "last_check": latest.timestamp,
                        "consecutive_failures": self._consecutive_failures.get(gpu_id, 0)
                    }

                    # Increment counter
                    summary[status] += 1
                else:
                    summary["unknown"] += 1
                    summary["gpus"][gpu_id] = {"status": "unknown"}

            return summary

    def get_gpu_history(self, gpu_id: int, limit: int = 10) -> List[HealthCheck]:
        """
        Get health check history for a GPU

        Args:
            gpu_id: GPU device ID
            limit: Maximum number of recent checks to return

        Returns:
            List of recent HealthCheck results
        """
        with self._lock:
            if gpu_id not in self._health_history:
                return []

            history = self._health_history[gpu_id]
            return history[-limit:]

    def is_gpu_healthy(self, gpu_id: int) -> bool:
        """
        Check if GPU is currently healthy

        Args:
            gpu_id: GPU device ID

        Returns:
            True if GPU is healthy
        """
        with self._lock:
            if gpu_id not in self._health_history or not self._health_history[gpu_id]:
                # No history, perform check now
                health_check = self.check_gpu_health(gpu_id)
                return health_check.status == HealthStatus.HEALTHY

            latest = self._health_history[gpu_id][-1]
            return latest.status == HealthStatus.HEALTHY


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

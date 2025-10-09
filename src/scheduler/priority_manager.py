import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import heapq

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Job priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class JobRequest:
    """Represents a job waiting in queue"""
    job_id: str
    user_id: str
    priority: Priority
    num_gpus: int
    submitted_at: float
    estimated_duration: int  # seconds
    metadata: Dict = field(default_factory=dict)

    def __lt__(self, other: 'JobRequest') -> bool:
        """
        Comparison for priority queue
        Higher priority first, then earlier submission time
        """
        # Higher priority value = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value

        # Same priority: earlier submission time wins
        return self.submitted_at < other.submitted_at


@dataclass
class UserStats:
    """Statistics for a user"""
    user_id: str
    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_gpu_hours: float = 0.0
    current_queue_size: int = 0
    fair_share_quota: float = 1.0  # Relative share (1.0 = equal share)


class PriorityManager:
    """
    Manages job priorities and implements fair scheduling
    Uses weighted fair queuing to balance priorities and fairness
    """

    def __init__(
        self,
        enable_fair_share: bool = True,
        starvation_timeout: int = 3600  # 1 hour
    ):
        """
        Initialize priority manager

        Args:
            enable_fair_share: Enable fair share scheduling
            starvation_timeout: Maximum wait time before priority boost (seconds)
        """
        self.enable_fair_share = enable_fair_share
        self.starvation_timeout = starvation_timeout

        self._lock = Lock()
        self._queue: List[JobRequest] = []
        self._user_stats: Dict[str, UserStats] = {}
        self._job_index: Dict[str, JobRequest] = {}

        logger.info(
            f"Priority manager initialized (fair_share={enable_fair_share}, "
            f"starvation_timeout={starvation_timeout}s)"
        )

    def submit_job(
        self,
        job_id: str,
        user_id: str,
        num_gpus: int,
        priority: Priority = Priority.MEDIUM,
        estimated_duration: int = 3600,
        metadata: Optional[Dict] = None
    ) -> JobRequest:
        """
        Submit a job to the priority queue

        Args:
            job_id: Unique job identifier
            user_id: User who submitted the job
            num_gpus: Number of GPUs requested
            priority: Job priority level
            estimated_duration: Estimated job duration in seconds
            metadata: Additional job metadata

        Returns:
            JobRequest object
        """
        with self._lock:
            # Check if job already in queue
            if job_id in self._job_index:
                logger.warning(f"Job {job_id} already in queue")
                return self._job_index[job_id]

            # Create job request
            job_request = JobRequest(
                job_id=job_id,
                user_id=user_id,
                priority=priority,
                num_gpus=num_gpus,
                submitted_at=time.time(),
                estimated_duration=estimated_duration,
                metadata=metadata or {}
            )

            # Add to priority queue
            heapq.heappush(self._queue, job_request)
            self._job_index[job_id] = job_request

            # Update user stats
            if user_id not in self._user_stats:
                self._user_stats[user_id] = UserStats(user_id=user_id)

            stats = self._user_stats[user_id]
            stats.total_jobs_submitted += 1
            stats.current_queue_size += 1

            logger.info(
                f"Submitted job {job_id} (user={user_id}, priority={priority.name}, "
                f"gpus={num_gpus}, queue_position={self.get_queue_position(job_id)})"
            )

            return job_request

    def get_next_job(
        self,
        available_gpus: int,
        user_limits: Optional[Dict[str, int]] = None
    ) -> Optional[JobRequest]:
        """
        Get next job to schedule based on priority and fairness

        Args:
            available_gpus: Number of GPUs currently available
            user_limits: Optional dict mapping user_id to max concurrent jobs

        Returns:
            Next JobRequest to schedule, or None if no eligible jobs
        """
        with self._lock:
            if not self._queue:
                return None

            current_time = time.time()

            # Apply starvation prevention
            self._prevent_starvation(current_time)

            # Try to find a schedulable job
            # We need to check multiple jobs because some might request
            # more GPUs than available or exceed user limits
            temp_queue = []

            while self._queue:
                job = heapq.heappop(self._queue)

                # Check if job can be scheduled
                if self._can_schedule(job, available_gpus, user_limits):
                    # Put remaining jobs back
                    for remaining_job in temp_queue:
                        heapq.heappush(self._queue, remaining_job)

                    # Remove from index (will be managed by scheduler now)
                    del self._job_index[job.job_id]

                    # Update user stats
                    stats = self._user_stats[job.user_id]
                    stats.current_queue_size -= 1

                    logger.info(
                        f"Scheduling job {job.job_id} (waited "
                        f"{current_time - job.submitted_at:.1f}s)"
                    )

                    return job

                # Can't schedule this job, keep it
                temp_queue.append(job)

            # No schedulable jobs found, restore queue
            for job in temp_queue:
                heapq.heappush(self._queue, job)

            return None

    def _can_schedule(
        self,
        job: JobRequest,
        available_gpus: int,
        user_limits: Optional[Dict[str, int]]
    ) -> bool:
        """
        Check if a job can be scheduled

        Args:
            job: Job to check
            available_gpus: Available GPU count
            user_limits: User job limits

        Returns:
            True if job can be scheduled
        """
        # Check GPU availability
        if job.num_gpus > available_gpus:
            return False

        # Check user limits if provided
        if user_limits and job.user_id in user_limits:
            user_limit = user_limits[job.user_id]
            stats = self._user_stats[job.user_id]
            # This would need to track current running jobs
            # For now, we'll assume this is checked externally
            pass

        # Check fair share if enabled
        if self.enable_fair_share:
            if not self._check_fair_share(job.user_id):
                return False

        return True

    def _check_fair_share(self, user_id: str) -> bool:
        """
        Check if user is within fair share quota

        Args:
            user_id: User to check

        Returns:
            True if within quota
        """
        if user_id not in self._user_stats:
            return True

        stats = self._user_stats[user_id]

        # Calculate total GPU hours used by all users
        total_gpu_hours = sum(
            s.total_gpu_hours for s in self._user_stats.values()
        )

        if total_gpu_hours == 0:
            return True

        # Calculate user's share
        user_share = stats.total_gpu_hours / total_gpu_hours

        # Check if user has exceeded their fair share significantly
        # Allow some flexibility (2x fair share)
        num_users = len(self._user_stats)
        fair_share = stats.fair_share_quota / num_users
        max_share = fair_share * 2

        if user_share > max_share:
            logger.debug(
                f"User {user_id} exceeds fair share: "
                f"{user_share:.2%} > {max_share:.2%}"
            )
            return False

        return True

    def _prevent_starvation(self, current_time: float):
        """
        Boost priority of jobs waiting too long

        Args:
            current_time: Current timestamp
        """
        for job in self._queue:
            wait_time = current_time - job.submitted_at

            if wait_time > self.starvation_timeout:
                # Boost priority if not already HIGH
                if job.priority != Priority.HIGH:
                    old_priority = job.priority
                    job.priority = Priority.HIGH
                    logger.info(
                        f"Boosted priority of job {job.job_id} "
                        f"from {old_priority.name} to HIGH due to long wait "
                        f"({wait_time:.0f}s)"
                    )

        # Re-heapify after priority changes
        heapq.heapify(self._queue)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job in the queue

        Args:
            job_id: Job to cancel

        Returns:
            True if job was cancelled
        """
        with self._lock:
            if job_id not in self._job_index:
                return False

            job = self._job_index[job_id]

            # Remove from queue (rebuild heap without this job)
            self._queue = [j for j in self._queue if j.job_id != job_id]
            heapq.heapify(self._queue)

            # Remove from index
            del self._job_index[job_id]

            # Update user stats
            if job.user_id in self._user_stats:
                stats = self._user_stats[job.user_id]
                stats.current_queue_size -= 1

            logger.info(f"Cancelled job {job_id}")
            return True

    def record_job_completion(
        self,
        job_id: str,
        user_id: str,
        num_gpus: int,
        duration: float
    ):
        """
        Record job completion for fair share accounting

        Args:
            job_id: Completed job ID
            user_id: User who ran the job
            num_gpus: Number of GPUs used
            duration: Job duration in seconds
        """
        with self._lock:
            if user_id not in self._user_stats:
                self._user_stats[user_id] = UserStats(user_id=user_id)

            stats = self._user_stats[user_id]
            stats.total_jobs_completed += 1
            stats.total_gpu_hours += (num_gpus * duration) / 3600.0

            logger.debug(
                f"Recorded completion of job {job_id}: "
                f"{stats.total_gpu_hours:.2f} total GPU-hours for user {user_id}"
            )

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get position of job in queue (1-indexed)

        Args:
            job_id: Job to check

        Returns:
            Queue position or None if not in queue
        """
        with self._lock:
            if job_id not in self._job_index:
                return None

            # Sort queue to get positions
            sorted_queue = sorted(self._queue)
            for i, job in enumerate(sorted_queue):
                if job.job_id == job_id:
                    return i + 1

            return None

    def get_queue_size(self) -> int:
        """Get total number of jobs in queue"""
        with self._lock:
            return len(self._queue)

    def get_user_stats(self, user_id: str) -> Optional[UserStats]:
        """Get statistics for a user"""
        with self._lock:
            return self._user_stats.get(user_id)

    def get_queue_summary(self) -> Dict:
        """
        Get summary of queue state

        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            priority_counts = {
                Priority.HIGH.name: 0,
                Priority.MEDIUM.name: 0,
                Priority.LOW.name: 0
            }

            total_requested_gpus = 0

            for job in self._queue:
                priority_counts[job.priority.name] += 1
                total_requested_gpus += job.num_gpus

            return {
                "total_jobs": len(self._queue),
                "priority_breakdown": priority_counts,
                "total_requested_gpus": total_requested_gpus,
                "unique_users": len(set(j.user_id for j in self._queue)),
                "oldest_job_wait_time": (
                    time.time() - min(j.submitted_at for j in self._queue)
                    if self._queue else 0
                )
            }

    def set_user_fair_share(self, user_id: str, quota: float):
        """
        Set fair share quota for a user

        Args:
            user_id: User identifier
            quota: Relative quota (1.0 = equal share)
        """
        with self._lock:
            if user_id not in self._user_stats:
                self._user_stats[user_id] = UserStats(user_id=user_id)

            self._user_stats[user_id].fair_share_quota = quota
            logger.info(f"Set fair share quota for user {user_id}: {quota}")


# Global priority manager instance
_priority_manager: Optional[PriorityManager] = None


def get_priority_manager() -> PriorityManager:
    """Get global priority manager instance"""
    global _priority_manager
    if _priority_manager is None:
        _priority_manager = PriorityManager()
    return _priority_manager

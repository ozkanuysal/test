import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from threading import Lock
import time

from .gpu_manager import GPUManager, GPUAllocation, get_gpu_manager
from ..utils.exceptions import (
    ResourcePoolExhaustedError,
    GPUNotAvailableError,
    ResourceException
)

logger = logging.getLogger(__name__)


class PoolType(Enum):
    """Types of resource pools"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"


@dataclass
class UserQuota:
    """Resource quota for a user"""
    user_id: str
    max_concurrent_gpus: int = 2
    max_concurrent_jobs: int = 5
    priority: int = 1  # 1=low, 2=medium, 3=high
    current_gpu_usage: int = 0
    current_job_count: int = 0


@dataclass
class PoolConfig:
    """Configuration for a resource pool"""
    name: str
    pool_type: PoolType
    gpu_ids: List[int]
    max_job_duration: int = 86400  # 24 hours in seconds
    allow_preemption: bool = False


@dataclass
class PoolAllocation:
    """Represents an allocation from a resource pool"""
    job_id: str
    user_id: str
    pool_name: str
    gpu_allocation: GPUAllocation
    allocated_at: float
    max_duration: int
    is_preemptible: bool = False


class ResourcePool:
    """
    Manages a pool of GPU resources
    Handles allocation, quotas, and preemption
    """

    def __init__(
        self,
        name: str,
        pool_type: PoolType,
        gpu_ids: List[int],
        gpu_manager: Optional[GPUManager] = None,
        max_job_duration: int = 86400,
        allow_preemption: bool = False
    ):
        """
        Initialize resource pool

        Args:
            name: Pool identifier
            pool_type: Type of pool (development or production)
            gpu_ids: List of GPU IDs in this pool
            gpu_manager: GPU manager instance
            max_job_duration: Maximum job duration in seconds
            allow_preemption: Whether to allow job preemption
        """
        self.name = name
        self.pool_type = pool_type
        self.gpu_ids = set(gpu_ids)
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.max_job_duration = max_job_duration
        self.allow_preemption = allow_preemption

        self._lock = Lock()
        self._allocations: Dict[str, PoolAllocation] = {}
        self._user_quotas: Dict[str, UserQuota] = {}
        self._reserved_gpus: Set[int] = set()

        logger.info(
            f"Initialized {pool_type.value} pool '{name}' with {len(gpu_ids)} GPUs: {gpu_ids}"
        )

    def set_user_quota(
        self,
        user_id: str,
        max_concurrent_gpus: int = 2,
        max_concurrent_jobs: int = 5,
        priority: int = 1
    ):
        """
        Set resource quota for a user

        Args:
            user_id: User identifier
            max_concurrent_gpus: Maximum concurrent GPUs
            max_concurrent_jobs: Maximum concurrent jobs
            priority: User priority level (1-3)
        """
        with self._lock:
            if user_id not in self._user_quotas:
                quota = UserQuota(
                    user_id=user_id,
                    max_concurrent_gpus=max_concurrent_gpus,
                    max_concurrent_jobs=max_concurrent_jobs,
                    priority=priority
                )
                self._user_quotas[user_id] = quota
            else:
                quota = self._user_quotas[user_id]
                quota.max_concurrent_gpus = max_concurrent_gpus
                quota.max_concurrent_jobs = max_concurrent_jobs
                quota.priority = priority

            logger.info(f"Set quota for user {user_id}: {quota}")

    def get_user_quota(self, user_id: str) -> UserQuota:
        """
        Get quota for a user

        Args:
            user_id: User identifier

        Returns:
            UserQuota object
        """
        if user_id not in self._user_quotas:
            # Create default quota
            self.set_user_quota(user_id)
        return self._user_quotas[user_id]

    def get_available_gpus(self) -> List[int]:
        """
        Get available GPUs in this pool

        Returns:
            List of available GPU IDs
        """
        return list(self.gpu_ids - self._reserved_gpus)

    def can_allocate(self, user_id: str, num_gpus: int) -> bool:
        """
        Check if allocation is possible for user

        Args:
            user_id: User identifier
            num_gpus: Number of GPUs requested

        Returns:
            True if allocation is possible
        """
        quota = self.get_user_quota(user_id)

        # Check if available GPUs in pool
        if len(self.get_available_gpus()) < num_gpus:
            return False

        # Check user GPU quota
        if quota.current_gpu_usage + num_gpus > quota.max_concurrent_gpus:
            logger.debug(
                f"User {user_id} would exceed GPU quota: "
                f"{quota.current_gpu_usage + num_gpus} > {quota.max_concurrent_gpus}"
            )
            return False

        # Check user job quota
        if quota.current_job_count >= quota.max_concurrent_jobs:
            logger.debug(
                f"User {user_id} would exceed job quota: "
                f"{quota.current_job_count} >= {quota.max_concurrent_jobs}"
            )
            return False

        return True

    def allocate(
        self,
        job_id: str,
        user_id: str,
        num_gpus: int,
        is_preemptible: bool = False,
        min_memory_per_gpu: Optional[int] = None
    ) -> PoolAllocation:
        """
        Allocate GPUs from the pool

        Args:
            job_id: Job identifier
            user_id: User identifier
            num_gpus: Number of GPUs to allocate
            is_preemptible: Whether job can be preempted
            min_memory_per_gpu: Minimum memory per GPU

        Returns:
            PoolAllocation object

        Raises:
            ResourcePoolExhaustedError: If pool is exhausted
            ResourceException: If quota exceeded
        """
        with self._lock:
            # Check if already allocated
            if job_id in self._allocations:
                logger.warning(f"Job {job_id} already has allocation in pool {self.name}")
                return self._allocations[job_id]

            # Check quota
            if not self.can_allocate(user_id, num_gpus):
                # Try preemption if allowed
                if self.allow_preemption and not is_preemptible:
                    self._try_preempt(num_gpus)

                # Recheck after preemption
                if not self.can_allocate(user_id, num_gpus):
                    quota = self.get_user_quota(user_id)
                    raise ResourceException(
                        f"Cannot allocate {num_gpus} GPUs for user {user_id}. "
                        f"Current usage: {quota.current_gpu_usage}/{quota.max_concurrent_gpus} GPUs, "
                        f"{quota.current_job_count}/{quota.max_concurrent_jobs} jobs"
                    )

            # Get available GPUs
            available = self.get_available_gpus()
            if len(available) < num_gpus:
                raise ResourcePoolExhaustedError(self.name)

            # Select GPUs from pool
            selected_gpu_ids = available[:num_gpus]

            # Allocate via GPU manager
            try:
                gpu_allocation = self.gpu_manager.allocate_gpus(
                    job_id=job_id,
                    num_gpus=num_gpus,
                    min_memory_per_gpu=min_memory_per_gpu
                )

                # Create pool allocation
                pool_allocation = PoolAllocation(
                    job_id=job_id,
                    user_id=user_id,
                    pool_name=self.name,
                    gpu_allocation=gpu_allocation,
                    allocated_at=time.time(),
                    max_duration=self.max_job_duration,
                    is_preemptible=is_preemptible
                )

                # Update tracking
                self._allocations[job_id] = pool_allocation
                self._reserved_gpus.update(selected_gpu_ids)

                # Update user quota
                quota = self.get_user_quota(user_id)
                quota.current_gpu_usage += num_gpus
                quota.current_job_count += 1

                logger.info(
                    f"Allocated {num_gpus} GPUs {selected_gpu_ids} from pool '{self.name}' "
                    f"to job {job_id} (user: {user_id})"
                )

                return pool_allocation

            except Exception as e:
                logger.error(f"Failed to allocate GPUs: {e}")
                raise

    def release(self, job_id: str):
        """
        Release allocation from pool

        Args:
            job_id: Job identifier

        Raises:
            ResourceException: If job has no allocation
        """
        with self._lock:
            if job_id not in self._allocations:
                raise ResourceException(
                    f"Job {job_id} has no allocation in pool {self.name}"
                )

            allocation = self._allocations[job_id]

            # Release from GPU manager
            self.gpu_manager.release_gpus(job_id)

            # Update tracking
            gpu_ids = allocation.gpu_allocation.gpu_ids
            self._reserved_gpus.difference_update(gpu_ids)
            del self._allocations[job_id]

            # Update user quota
            quota = self.get_user_quota(allocation.user_id)
            quota.current_gpu_usage -= len(gpu_ids)
            quota.current_job_count -= 1

            logger.info(
                f"Released {len(gpu_ids)} GPUs {gpu_ids} from pool '{self.name}' "
                f"for job {job_id}"
            )

    def _try_preempt(self, needed_gpus: int) -> bool:
        """
        Try to preempt low-priority jobs to free up resources

        Args:
            needed_gpus: Number of GPUs needed

        Returns:
            True if enough resources were freed
        """
        # Find preemptible jobs sorted by priority
        preemptible = [
            (job_id, alloc) for job_id, alloc in self._allocations.items()
            if alloc.is_preemptible
        ]

        if not preemptible:
            return False

        # Sort by user priority (lower priority first)
        preemptible.sort(
            key=lambda x: self.get_user_quota(x[1].user_id).priority
        )

        freed_gpus = 0
        preempted_jobs = []

        for job_id, allocation in preemptible:
            preempted_jobs.append(job_id)
            freed_gpus += len(allocation.gpu_allocation.gpu_ids)

            if freed_gpus >= needed_gpus:
                break

        # Preempt jobs
        for job_id in preempted_jobs:
            logger.info(f"Preempting job {job_id} from pool {self.name}")
            self.release(job_id)

        return freed_gpus >= needed_gpus

    def get_stats(self) -> Dict:
        """
        Get pool statistics

        Returns:
            Dictionary with pool stats
        """
        with self._lock:
            return {
                "name": self.name,
                "type": self.pool_type.value,
                "total_gpus": len(self.gpu_ids),
                "available_gpus": len(self.get_available_gpus()),
                "allocated_gpus": len(self._reserved_gpus),
                "active_jobs": len(self._allocations),
                "total_users": len(self._user_quotas),
                "allow_preemption": self.allow_preemption
            }


class ResourcePoolManager:
    """
    Manages multiple resource pools
    Routes requests to appropriate pools
    """

    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize pool manager

        Args:
            gpu_manager: GPU manager instance
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self._pools: Dict[str, ResourcePool] = {}
        self._lock = Lock()

    def create_pool(
        self,
        name: str,
        pool_type: PoolType,
        gpu_ids: List[int],
        max_job_duration: int = 86400,
        allow_preemption: bool = False
    ) -> ResourcePool:
        """
        Create a new resource pool

        Args:
            name: Pool identifier
            pool_type: Type of pool
            gpu_ids: GPUs assigned to pool
            max_job_duration: Maximum job duration
            allow_preemption: Allow preemption

        Returns:
            Created ResourcePool
        """
        with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = ResourcePool(
                name=name,
                pool_type=pool_type,
                gpu_ids=gpu_ids,
                gpu_manager=self.gpu_manager,
                max_job_duration=max_job_duration,
                allow_preemption=allow_preemption
            )

            self._pools[name] = pool
            return pool

    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get pool by name"""
        return self._pools.get(name)

    def get_pools_by_type(self, pool_type: PoolType) -> List[ResourcePool]:
        """Get all pools of a specific type"""
        return [
            pool for pool in self._pools.values()
            if pool.pool_type == pool_type
        ]

    def allocate_from_best_pool(
        self,
        job_id: str,
        user_id: str,
        num_gpus: int,
        pool_type: PoolType,
        is_preemptible: bool = False
    ) -> PoolAllocation:
        """
        Allocate from the best available pool of given type

        Args:
            job_id: Job identifier
            user_id: User identifier
            num_gpus: Number of GPUs
            pool_type: Preferred pool type
            is_preemptible: Whether job can be preempted

        Returns:
            PoolAllocation

        Raises:
            ResourcePoolExhaustedError: If no suitable pool found
        """
        pools = self.get_pools_by_type(pool_type)

        if not pools:
            raise ResourcePoolExhaustedError(f"No {pool_type.value} pools available")

        # Try each pool
        for pool in pools:
            try:
                return pool.allocate(job_id, user_id, num_gpus, is_preemptible)
            except (ResourcePoolExhaustedError, ResourceException) as e:
                logger.debug(f"Failed to allocate from pool {pool.name}: {e}")
                continue

        raise ResourcePoolExhaustedError(
            f"All {pool_type.value} pools exhausted for {num_gpus} GPUs"
        )

    def get_global_stats(self) -> Dict:
        """Get statistics for all pools"""
        return {
            "total_pools": len(self._pools),
            "pools": {name: pool.get_stats() for name, pool in self._pools.items()}
        }

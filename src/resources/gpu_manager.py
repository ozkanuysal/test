import os
import logging
from typing import List, Optional, Dict, Set
from dataclasses import dataclass
from threading import Lock
import time

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available. GPU monitoring will be disabled.")

import torch

from ..utils.exceptions import (
    GPUNotAvailableError,
    GPUAllocationError,
    ResourceException
)

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device"""
    id: int
    name: str
    total_memory: int  # in bytes
    free_memory: int
    used_memory: int
    utilization: float  # percentage
    temperature: Optional[float] = None
    power_usage: Optional[float] = None
    is_healthy: bool = True


@dataclass
class GPUAllocation:
    """Represents an allocated GPU resource"""
    gpu_ids: List[int]
    job_id: str
    allocated_at: float
    memory_reserved: int  # in bytes per GPU


class GPUManager:
    """
    Manages GPU resources for the ML platform
    Handles detection, allocation, monitoring, and health checking
    """

    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize GPU Manager

        Args:
            enable_monitoring: Enable NVIDIA GPU monitoring via pynvml
        """
        self.enable_monitoring = enable_monitoring and PYNVML_AVAILABLE
        self._allocation_lock = Lock()
        self._allocations: Dict[str, GPUAllocation] = {}
        self._reserved_gpus: Set[int] = set()

        # Initialize pynvml if available
        if self.enable_monitoring:
            try:
                pynvml.nvmlInit()
                logger.info("NVIDIA GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize pynvml: {e}")
                self.enable_monitoring = False

        # Detect available GPUs
        self._detect_gpus()

    def _detect_gpus(self):
        """Detect available GPUs"""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {self.num_gpus} CUDA-capable GPU(s)")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {gpu_name}")
        else:
            self.num_gpus = 0
            logger.warning("No CUDA-capable GPUs detected")

    def get_gpu_info(self, gpu_id: int) -> GPUInfo:
        """
        Get detailed information about a specific GPU

        Args:
            gpu_id: GPU device ID

        Returns:
            GPUInfo object with current GPU state

        Raises:
            ResourceException: If GPU info cannot be retrieved
        """
        if gpu_id >= self.num_gpus:
            raise ResourceException(f"Invalid GPU ID: {gpu_id}")

        try:
            # Get basic info from PyTorch
            name = torch.cuda.get_device_name(gpu_id)
            properties = torch.cuda.get_device_properties(gpu_id)
            total_memory = properties.total_memory

            # Get detailed info from pynvml if available
            if self.enable_monitoring:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    temperature = None

                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_usage = None

                free_memory = mem_info.free
                used_memory = mem_info.used
                gpu_utilization = utilization.gpu
            else:
                # Fallback to PyTorch memory stats
                free_memory = total_memory - torch.cuda.memory_allocated(gpu_id)
                used_memory = torch.cuda.memory_allocated(gpu_id)
                gpu_utilization = 0.0
                temperature = None
                power_usage = None

            # Check if GPU is healthy
            is_healthy = True
            if temperature and temperature > 85:  # High temperature threshold
                is_healthy = False
                logger.warning(f"GPU {gpu_id} temperature too high: {temperature}°C")

            return GPUInfo(
                id=gpu_id,
                name=name,
                total_memory=total_memory,
                free_memory=free_memory,
                used_memory=used_memory,
                utilization=gpu_utilization,
                temperature=temperature,
                power_usage=power_usage,
                is_healthy=is_healthy
            )

        except Exception as e:
            raise ResourceException(f"Failed to get GPU info for GPU {gpu_id}: {e}")

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """
        Get information about all GPUs

        Returns:
            List of GPUInfo objects
        """
        return [self.get_gpu_info(i) for i in range(self.num_gpus)]

    def get_available_gpus(self, min_free_memory: Optional[int] = None) -> List[int]:
        """
        Get list of available (unallocated and healthy) GPUs

        Args:
            min_free_memory: Minimum free memory required in bytes

        Returns:
            List of available GPU IDs
        """
        available = []

        for i in range(self.num_gpus):
            # Check if already allocated
            if i in self._reserved_gpus:
                continue

            # Check GPU health and memory
            try:
                info = self.get_gpu_info(i)
                if not info.is_healthy:
                    continue

                if min_free_memory and info.free_memory < min_free_memory:
                    continue

                available.append(i)
            except Exception as e:
                logger.warning(f"Failed to check GPU {i}: {e}")
                continue

        return available

    def allocate_gpus(
        self,
        job_id: str,
        num_gpus: int,
        min_memory_per_gpu: Optional[int] = None
    ) -> GPUAllocation:
        """
        Allocate GPUs for a job

        Args:
            job_id: Unique job identifier
            num_gpus: Number of GPUs to allocate
            min_memory_per_gpu: Minimum memory required per GPU in bytes

        Returns:
            GPUAllocation object

        Raises:
            GPUNotAvailableError: If requested GPUs not available
            GPUAllocationError: If allocation fails
        """
        with self._allocation_lock:
            # Check if job already has allocation
            if job_id in self._allocations:
                logger.warning(f"Job {job_id} already has GPU allocation")
                return self._allocations[job_id]

            # Get available GPUs
            available_gpus = self.get_available_gpus(min_memory_per_gpu)

            if len(available_gpus) < num_gpus:
                raise GPUNotAvailableError(
                    requested_gpus=num_gpus,
                    available_gpus=len(available_gpus)
                )

            # Allocate the first N available GPUs
            allocated_gpu_ids = available_gpus[:num_gpus]

            # Create allocation record
            allocation = GPUAllocation(
                gpu_ids=allocated_gpu_ids,
                job_id=job_id,
                allocated_at=time.time(),
                memory_reserved=min_memory_per_gpu or 0
            )

            # Mark GPUs as reserved
            self._reserved_gpus.update(allocated_gpu_ids)
            self._allocations[job_id] = allocation

            # Set CUDA_VISIBLE_DEVICES for this job
            gpu_ids_str = ",".join(map(str, allocated_gpu_ids))
            logger.info(
                f"Allocated {num_gpus} GPU(s) {allocated_gpu_ids} to job {job_id}"
            )

            return allocation

    def release_gpus(self, job_id: str):
        """
        Release GPUs allocated to a job

        Args:
            job_id: Job identifier

        Raises:
            ResourceException: If job has no allocation
        """
        with self._allocation_lock:
            if job_id not in self._allocations:
                raise ResourceException(f"Job {job_id} has no GPU allocation")

            allocation = self._allocations[job_id]

            # Remove from reserved set
            self._reserved_gpus.difference_update(allocation.gpu_ids)

            # Remove allocation record
            del self._allocations[job_id]

            logger.info(
                f"Released {len(allocation.gpu_ids)} GPU(s) {allocation.gpu_ids} "
                f"from job {job_id}"
            )

    def get_allocation(self, job_id: str) -> Optional[GPUAllocation]:
        """
        Get GPU allocation for a job

        Args:
            job_id: Job identifier

        Returns:
            GPUAllocation if exists, None otherwise
        """
        return self._allocations.get(job_id)

    def get_all_allocations(self) -> Dict[str, GPUAllocation]:
        """
        Get all current GPU allocations

        Returns:
            Dictionary mapping job_id to GPUAllocation
        """
        return self._allocations.copy()

    def get_utilization_summary(self) -> Dict[str, any]:
        """
        Get summary of GPU utilization across all devices

        Returns:
            Dictionary with utilization statistics
        """
        gpu_infos = self.get_all_gpu_info()

        if not gpu_infos:
            return {
                "total_gpus": 0,
                "allocated_gpus": 0,
                "available_gpus": 0,
                "average_utilization": 0.0,
                "total_memory": 0,
                "used_memory": 0,
                "free_memory": 0
            }

        total_memory = sum(info.total_memory for info in gpu_infos)
        used_memory = sum(info.used_memory for info in gpu_infos)
        free_memory = sum(info.free_memory for info in gpu_infos)
        avg_utilization = sum(info.utilization for info in gpu_infos) / len(gpu_infos)

        return {
            "total_gpus": self.num_gpus,
            "allocated_gpus": len(self._reserved_gpus),
            "available_gpus": self.num_gpus - len(self._reserved_gpus),
            "average_utilization": avg_utilization,
            "total_memory": total_memory,
            "used_memory": used_memory,
            "free_memory": free_memory,
            "memory_utilization_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0
        }

    def set_cuda_visible_devices(self, gpu_ids: List[int]):
        """
        Set CUDA_VISIBLE_DEVICES environment variable

        Args:
            gpu_ids: List of GPU IDs to make visible
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    def cleanup(self):
        """Cleanup resources and shutdown pynvml"""
        if self.enable_monitoring:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVIDIA GPU monitoring shut down")
            except Exception as e:
                logger.warning(f"Failed to shutdown pynvml: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager

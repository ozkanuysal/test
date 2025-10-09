import pytest
from unittest.mock import Mock, patch
from src.resources.gpu_manager import GPUManager, GPUAllocation
from src.utils.exceptions import GPUNotAvailableError


class TestGPUManager:
    """Test GPU Manager functionality"""

    def test_gpu_detection(self, mock_gpu_manager):
        """Test GPU detection"""
        assert mock_gpu_manager.num_gpus == 2

    def test_get_gpu_info(self, mock_gpu_manager, mock_gpu_info):
        """Test getting GPU info"""
        info = mock_gpu_manager.get_gpu_info(0)
        assert info.id == 0
        assert info.total_memory > 0
        assert info.is_healthy

    def test_get_all_gpu_info(self, mock_gpu_manager):
        """Test getting all GPU info"""
        infos = mock_gpu_manager.get_all_gpu_info()
        assert len(infos) == 2

    def test_get_available_gpus(self, mock_gpu_manager):
        """Test getting available GPUs"""
        available = mock_gpu_manager.get_available_gpus()
        assert len(available) == 2
        assert 0 in available
        assert 1 in available

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_gpu_allocation(self, mock_count, mock_available):
        """Test GPU allocation"""
        manager = GPUManager(enable_monitoring=False)

        # Allocate 1 GPU
        allocation = manager.allocate_gpus("job-1", num_gpus=1)
        assert isinstance(allocation, GPUAllocation)
        assert len(allocation.gpu_ids) == 1
        assert allocation.job_id == "job-1"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_gpu_allocation_not_available(self, mock_count, mock_available):
        """Test GPU allocation when not enough GPUs"""
        manager = GPUManager(enable_monitoring=False)

        # Allocate all GPUs
        manager.allocate_gpus("job-1", num_gpus=2)

        # Try to allocate more
        with pytest.raises(GPUNotAvailableError):
            manager.allocate_gpus("job-2", num_gpus=1)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_gpu_release(self, mock_count, mock_available):
        """Test GPU release"""
        manager = GPUManager(enable_monitoring=False)

        # Allocate and release
        allocation = manager.allocate_gpus("job-1", num_gpus=1)
        manager.release_gpus("job-1")

        # Should be available again
        available = manager.get_available_gpus()
        assert len(available) == 2

    def test_utilization_summary(self, mock_gpu_manager):
        """Test GPU utilization summary"""
        summary = mock_gpu_manager.get_utilization_summary()

        assert "total_gpus" in summary
        assert "allocated_gpus" in summary
        assert "available_gpus" in summary
        assert "average_utilization" in summary
        assert summary["total_gpus"] == 2

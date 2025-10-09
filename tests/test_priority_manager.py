import pytest
import time
from src.scheduler.priority_manager import PriorityManager, Priority, JobRequest


class TestPriorityManager:
    """Test Priority Manager functionality"""

    @pytest.fixture
    def priority_manager(self):
        """Create priority manager instance"""
        return PriorityManager(enable_fair_share=True, starvation_timeout=60)

    def test_submit_job(self, priority_manager):
        """Test job submission"""
        job = priority_manager.submit_job(
            job_id="test-job-1",
            user_id="user-1",
            num_gpus=1,
            priority=Priority.MEDIUM
        )

        assert isinstance(job, JobRequest)
        assert job.job_id == "test-job-1"
        assert job.priority == Priority.MEDIUM

    def test_job_priority_ordering(self, priority_manager):
        """Test that jobs are ordered by priority"""
        # Submit jobs with different priorities
        priority_manager.submit_job("job-low", "user-1", 1, Priority.LOW)
        priority_manager.submit_job("job-high", "user-1", 1, Priority.HIGH)
        priority_manager.submit_job("job-med", "user-1", 1, Priority.MEDIUM)

        # Get next job should be HIGH priority
        next_job = priority_manager.get_next_job(available_gpus=1)
        assert next_job.job_id == "job-high"

    def test_queue_position(self, priority_manager):
        """Test getting queue position"""
        priority_manager.submit_job("job-1", "user-1", 1, Priority.MEDIUM)
        priority_manager.submit_job("job-2", "user-1", 1, Priority.MEDIUM)
        priority_manager.submit_job("job-3", "user-1", 1, Priority.MEDIUM)

        # Check positions
        assert priority_manager.get_queue_position("job-1") == 1
        assert priority_manager.get_queue_position("job-2") == 2
        assert priority_manager.get_queue_position("job-3") == 3

    def test_cancel_job(self, priority_manager):
        """Test job cancellation"""
        priority_manager.submit_job("job-1", "user-1", 1, Priority.MEDIUM)

        # Cancel job
        result = priority_manager.cancel_job("job-1")
        assert result is True

        # Job should not be in queue
        assert priority_manager.get_queue_position("job-1") is None

    def test_queue_size(self, priority_manager):
        """Test getting queue size"""
        assert priority_manager.get_queue_size() == 0

        priority_manager.submit_job("job-1", "user-1", 1, Priority.MEDIUM)
        priority_manager.submit_job("job-2", "user-1", 1, Priority.MEDIUM)

        assert priority_manager.get_queue_size() == 2

    def test_starvation_prevention(self, priority_manager):
        """Test starvation prevention mechanism"""
        # Submit a LOW priority job
        job = priority_manager.submit_job(
            "job-low", "user-1", 1, Priority.LOW, estimated_duration=60
        )

        # Simulate passage of time
        job.submitted_at = time.time() - 3700  # 1 hour + 100 seconds ago

        # Trigger starvation prevention
        priority_manager._prevent_starvation(time.time())

        # Job should now be HIGH priority
        assert job.priority == Priority.HIGH

    def test_fair_share(self, priority_manager):
        """Test fair share accounting"""
        # Set quotas
        priority_manager.set_user_fair_share("user-1", quota=1.0)
        priority_manager.set_user_fair_share("user-2", quota=1.0)

        # Record usage
        priority_manager.record_job_completion(
            "job-1", "user-1", num_gpus=2, duration=3600
        )

        # Check stats
        stats = priority_manager.get_user_stats("user-1")
        assert stats.total_jobs_completed == 1
        assert stats.total_gpu_hours == 2.0

    def test_get_queue_summary(self, priority_manager):
        """Test queue summary"""
        priority_manager.submit_job("job-1", "user-1", 1, Priority.HIGH)
        priority_manager.submit_job("job-2", "user-2", 2, Priority.MEDIUM)
        priority_manager.submit_job("job-3", "user-1", 1, Priority.LOW)

        summary = priority_manager.get_queue_summary()

        assert summary["total_jobs"] == 3
        assert summary["priority_breakdown"]["HIGH"] == 1
        assert summary["priority_breakdown"]["MEDIUM"] == 1
        assert summary["priority_breakdown"]["LOW"] == 1
        assert summary["total_requested_gpus"] == 4
        assert summary["unique_users"] == 2

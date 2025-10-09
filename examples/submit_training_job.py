"""
Example: Submit a Training Job
Demonstrates how to submit a fine-tuning job to the platform
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scheduler.job_queue import JobConfig, JobQueue, Priority, get_job_queue
from src.scheduler.priority_manager import get_priority_manager
from src.monitoring.logger import setup_logging

def main():
    # Setup logging
    setup_logging(level="INFO")

    # Create job configuration
    job_config = JobConfig(
        job_id="training-job-001",
        user_id="user-123",
        job_type="fine_tuning",

        # Resource requirements
        num_gpus=1,
        pool_type="development",
        is_preemptible=False,

        # Model and data
        model_name="bert-base-uncased",
        dataset_path="./data/train.csv",
        output_dir="./output/job-001",

        # Job metadata
        priority="MEDIUM",
        estimated_duration=3600,

        # Training config
        config={
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "max_seq_length": 128
        }
    )

    # Get job queue
    job_queue = get_job_queue()

    # Submit job
    print(f"Submitting job: {job_config.job_id}")
    job_id = job_queue.submit_job(job_config, Priority.MEDIUM)

    print(f"✓ Job submitted successfully: {job_id}")
    print(f"  User: {job_config.user_id}")
    print(f"  GPUs: {job_config.num_gpus}")
    print(f"  Priority: {job_config.priority}")
    print(f"  Output: {job_config.output_dir}")

    # Check queue position
    priority_manager = get_priority_manager()
    position = priority_manager.get_queue_position(job_id)
    print(f"  Queue position: {position}")

    # Get queue stats
    stats = job_queue.get_queue_stats()
    print(f"\nQueue Statistics:")
    print(f"  Total jobs in queue: {stats['priority_queue']['total_jobs']}")
    print(f"  Priority breakdown: {stats['priority_queue']['priority_breakdown']}")

if __name__ == "__main__":
    main()

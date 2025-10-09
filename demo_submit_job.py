import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.scheduler.job_queue import JobConfig, get_job_queue, Priority
from src.scheduler.priority_manager import get_priority_manager
from src.monitoring.logger import setup_logging

def print_banner():
    """Print demo banner"""
    print("\n" + "="*60)
    print("   ML GPU PLATFORM - DEMO")
    print("   Production-Ready GPU Resource Management")
    print("="*60 + "\n")

def submit_demo_job():
    """Submit a demo training job"""
    print("📝 Creating job configuration...")

    # Create job configuration
    job_config = JobConfig(
        job_id="demo-job-001",
        user_id="demo-user",
        job_type="fine_tuning",

        # Resource requirements
        num_gpus=1,
        pool_type="development",
        is_preemptible=False,

        # Model and data
        model_name="bert-base-uncased",
        dataset_path="/app/data/train.csv",
        output_dir="/app/output/demo-job-001",

        # Job metadata
        priority="HIGH",
        estimated_duration=300,  # 5 minutes
        max_retries=3,

        # Training config
        config={
            "batch_size": 8,
            "learning_rate": 2e-5,
            "num_epochs": 1,
            "max_seq_length": 128,
            "max_steps": 10  # Just 10 steps for demo
        }
    )

    print(f"✓ Job ID: {job_config.job_id}")
    print(f"✓ User: {job_config.user_id}")
    print(f"✓ GPUs: {job_config.num_gpus}")
    print(f"✓ Priority: {job_config.priority}")
    print(f"✓ Model: {job_config.model_name}")
    print(f"✓ Dataset: {job_config.dataset_path}")

    # Get job queue
    print("\n🚀 Submitting job to queue...")
    job_queue = get_job_queue()

    try:
        job_id = job_queue.submit_job(job_config, Priority.HIGH)
        print(f"✅ Job submitted successfully: {job_id}\n")

        # Get queue position
        priority_manager = get_priority_manager()
        position = priority_manager.get_queue_position(job_id)
        print(f"📊 Queue position: {position if position else 'Processing'}")

        # Get queue stats
        print("\n📈 Queue Statistics:")
        stats = job_queue.get_queue_stats()
        pq_stats = stats.get('priority_queue', {})
        print(f"   Total jobs in queue: {pq_stats.get('total_jobs', 0)}")
        print(f"   Priority breakdown: {pq_stats.get('priority_breakdown', {})}")

        return job_id, job_queue

    except Exception as e:
        print(f"❌ Error submitting job: {e}")
        return None, None

def monitor_job(job_id, job_queue, duration=30):
    """Monitor job status"""
    print(f"\n👁️  Monitoring job {job_id} for {duration} seconds...")
    print("-" * 60)

    start_time = time.time()
    last_status = None

    while time.time() - start_time < duration:
        try:
            status = job_queue.get_job_status(job_id)

            if status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed:3d}s] Status: {status.value.upper()}")
                last_status = status

                # Check if job completed or failed
                if status.value in ['completed', 'failed', 'cancelled']:
                    print(f"\n🏁 Job finished with status: {status.value.upper()}")

                    # Try to get result
                    result = job_queue.get_job_result(job_id)
                    if result:
                        print(f"\n📊 Job Result:")
                        print(f"   Duration: {result.duration:.2f}s" if result.duration else "   Duration: N/A")
                        print(f"   Status: {result.status.value}")
                        if result.error:
                            print(f"   Error: {result.error}")
                    break

            time.sleep(2)

        except KeyboardInterrupt:
            print("\n⏸️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Error monitoring job: {e}")
            break

    print("-" * 60)

def show_system_info():
    """Show system information"""
    print("\n🔧 System Information:")
    print("-" * 60)

    try:
        from src.resources.gpu_manager import get_gpu_manager
        gpu_manager = get_gpu_manager()

        print(f"   Total GPUs detected: {gpu_manager.num_gpus}")

        if gpu_manager.num_gpus > 0:
            summary = gpu_manager.get_utilization_summary()
            print(f"   Available GPUs: {summary['available_gpus']}")
            print(f"   Allocated GPUs: {summary['allocated_gpus']}")
        else:
            print("   ⚠️  No CUDA GPUs detected (running in CPU mode)")

    except Exception as e:
        print(f"   ⚠️  Could not get GPU info: {e}")

    print("-" * 60)

def main():
    """Main demo function"""
    # Setup logging
    setup_logging(level="INFO")

    # Print banner
    print_banner()

    # Show system info
    show_system_info()

    # Submit job
    job_id, job_queue = submit_demo_job()

    if job_id and job_queue:
        # Monitor job
        monitor_job(job_id, job_queue, duration=60)

        print("\n" + "="*60)
        print("✅ Demo completed!")
        print("\nNext steps:")
        print("  1. Check output directory: /app/output/demo-job-001")
        print("  2. View logs in the worker container")
        print("  3. Submit more jobs with different priorities")
        print("="*60 + "\n")
    else:
        print("\n❌ Demo failed to complete")
        sys.exit(1)

if __name__ == "__main__":
    main()

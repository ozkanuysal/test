"""
Example: Monitoring Dashboard
Simple CLI dashboard for monitoring system status
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resources.gpu_manager import get_gpu_manager
from src.resources.health_checker import get_health_checker
from src.scheduler.priority_manager import get_priority_manager
from src.monitoring.metrics import get_metrics_collector

def print_separator():
    print("=" * 80)

def display_gpu_status():
    """Display GPU utilization and health"""
    gpu_manager = get_gpu_manager()

    print("\n📊 GPU STATUS")
    print_separator()

    gpu_infos = gpu_manager.get_all_gpu_info()

    if not gpu_infos:
        print("No GPUs detected")
        return

    for info in gpu_infos:
        status_emoji = "✅" if info.is_healthy else "❌"
        mem_used_pct = (info.used_memory / info.total_memory * 100) if info.total_memory > 0 else 0

        print(f"{status_emoji} GPU {info.id}: {info.name}")
        print(f"   Utilization: {info.utilization:.1f}%")
        print(f"   Memory: {mem_used_pct:.1f}% ({info.used_memory/(1024**3):.1f}GB / {info.total_memory/(1024**3):.1f}GB)")

        if info.temperature:
            temp_emoji = "🔥" if info.temperature > 80 else "🌡️"
            print(f"   {temp_emoji} Temperature: {info.temperature:.1f}°C")

        if info.power_usage:
            print(f"   ⚡ Power: {info.power_usage:.1f}W")

        print()

    # Summary
    summary = gpu_manager.get_utilization_summary()
    print(f"Summary: {summary['allocated_gpus']}/{summary['total_gpus']} GPUs allocated, "
          f"{summary['average_utilization']:.1f}% avg utilization")

def display_health_status():
    """Display health check status"""
    health_checker = get_health_checker()

    print("\n🏥 HEALTH STATUS")
    print_separator()

    summary = health_checker.get_health_summary()

    print(f"Healthy: {summary['healthy']} | Degraded: {summary['degraded']} | "
          f"Unhealthy: {summary['unhealthy']} | Unknown: {summary['unknown']}")

    for gpu_id, health in summary["gpus"].items():
        if health.get("status") == "unknown":
            continue

        status = health["status"]
        emoji_map = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
        emoji = emoji_map.get(status, "❓")

        print(f"  {emoji} GPU {gpu_id}: {status.upper()}")
        if health.get("error"):
            print(f"      Error: {health['error']}")

def display_queue_status():
    """Display job queue status"""
    priority_manager = get_priority_manager()

    print("\n📋 QUEUE STATUS")
    print_separator()

    summary = priority_manager.get_queue_summary()

    print(f"Total jobs in queue: {summary['total_jobs']}")
    print(f"Unique users: {summary['unique_users']}")
    print(f"Oldest job wait time: {summary['oldest_job_wait_time']:.1f}s")

    print("\nPriority breakdown:")
    for priority, count in summary['priority_breakdown'].items():
        print(f"  {priority}: {count} jobs")

def main():
    """Main dashboard function"""
    import argparse

    parser = argparse.ArgumentParser(description="ML Platform Monitoring Dashboard")
    parser.add_argument("--refresh", type=int, default=0,
                      help="Auto-refresh interval in seconds (0=no refresh)")
    args = parser.parse_args()

    try:
        while True:
            # Clear screen (works on Unix-like systems)
            print("\033[2J\033[H", end="")

            print("=" * 80)
            print(" " * 25 + "ML PLATFORM DASHBOARD")
            print("=" * 80)
            print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            display_gpu_status()
            display_health_status()
            display_queue_status()

            print("\n" + "=" * 80)

            if args.refresh > 0:
                print(f"Refreshing in {args.refresh} seconds... (Ctrl+C to stop)")
                time.sleep(args.refresh)
            else:
                break

    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")

if __name__ == "__main__":
    main()

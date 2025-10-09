"""
Example: Full Production Pipeline
Demonstrates complete pipeline from data loading to model deployment
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.trainer import Trainer
from src.resources.resource_pool import ResourcePoolManager, PoolType
from src.resources.gpu_manager import get_gpu_manager
from src.monitoring.logger import setup_logging
from src.utils.config import load_config

def main():
    # Setup
    setup_logging(level="INFO")
    print("=" * 60)
    print("ML Platform - Production Pipeline Example")
    print("=" * 60)

    # Load production configuration
    print("\n[1/6] Loading configuration...")
    config = load_config("production")
    print(f"✓ Loaded config: {config.environment} environment")

    # Initialize resource managers
    print("\n[2/6] Initializing resource managers...")
    gpu_manager = get_gpu_manager()
    pool_manager = ResourcePoolManager(gpu_manager)

    # Create resource pools
    if gpu_manager.num_gpus >= 4:
        dev_pool = pool_manager.create_pool(
            name="development",
            pool_type=PoolType.DEVELOPMENT,
            gpu_ids=[0, 1],
            allow_preemption=True
        )

        prod_pool = pool_manager.create_pool(
            name="production",
            pool_type=PoolType.PRODUCTION,
            gpu_ids=list(range(2, gpu_manager.num_gpus)),
            allow_preemption=False
        )

        print(f"✓ Created development pool: {dev_pool.get_stats()}")
        print(f"✓ Created production pool: {prod_pool.get_stats()}")
    else:
        print("⚠ Not enough GPUs for separate pools, using default")

    # Create trainer
    print("\n[3/6] Setting up trainer...")
    trainer = Trainer(
        model_name=config.training.model_name,
        output_dir="./output/production_run",
        num_gpus=config.resource.max_gpus_per_job,
        config={
            "batch_size": config.training.batch_size,
            "max_seq_length": config.training.max_seq_length,
            "save_total_limit": config.training.save_total_limit
        }
    )

    trainer.setup(
        num_labels=2,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps
    )

    print("✓ Trainer configured")
    print(f"  Model: {config.training.model_name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")

    # Prepare data
    print("\n[4/6] Preparing data...")
    print("  (In production, load from S3/GCS)")
    print("  Example: s3://ml-platform-data-prod/datasets/train.csv")

    # Train model
    print("\n[5/6] Starting training...")
    print("  This would run the full training pipeline")
    print("  - Data loading from cloud storage")
    print("  - Model fine-tuning with checkpointing")
    print("  - Distributed training across GPUs")
    print("  - Metrics collection and monitoring")

    # For demo, we'll skip actual training
    print("  (Skipping actual training in this demo)")

    # Model deployment
    print("\n[6/6] Model deployment...")
    print("  After training completes:")
    print("  - Model saved to: ./output/production_run/final_model")
    print("  - Checkpoints in: ./output/production_run/checkpoints")
    print("  - Metrics exported to Prometheus")
    print("  - Ready for inference deployment")

    print("\n" + "=" * 60)
    print("Production pipeline example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

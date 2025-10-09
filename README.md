# ML GPU Platform - GPU Resource Management System

## 🎯 Overview

This platform addresses the challenge of efficiently managing limited GPU resources across multiple users and workloads. It provides:

- **Job Scheduling**: Priority-based queue with fair resource allocation
- **Resource Management**: Separate pools for development and production workloads
- **Scalability**: Same codebase from single-GPU experiments to multi-node distributed training
- **Monitoring**: Comprehensive metrics and health checking
- **Fault Tolerance**: Automatic retry, checkpointing, and failover

## ✨ Key Features

### Resource Management
- 🎮 **GPU Detection & Allocation**: Automatic GPU discovery and efficient allocation
- 🔄 **Resource Pools**: Separate dev/prod pools with different quotas and SLAs
- ❤️ **Health Monitoring**: Real-time GPU health tracking with automatic failover
- 📊 **Utilization Tracking**: Detailed metrics on GPU memory, temperature, and usage

### Job Scheduling
- 📋 **Priority Queue**: 3-level priority system (High, Medium, Low)
- ⚖️ **Fair Scheduling**: Per-user quotas prevent resource monopolization
- 🔁 **Automatic Retry**: Exponential backoff retry for failed jobs
- ⏰ **Starvation Prevention**: Auto-boost priority for long-waiting jobs

### ML Pipeline
- 🤖 **Model Support**: BERT and other HuggingFace transformers
- 💾 **Multi-Source Data**: Load from local files, S3, GCS, or HuggingFace hub
- 💿 **Checkpointing**: Automatic checkpoint saving with best model tracking
- 🚀 **Distributed Training**: Multi-GPU and multi-node support via Ray

### Monitoring & Observability
- 📈 **Prometheus Metrics**: GPU utilization, queue stats, job metrics
- 📝 **Structured Logging**: JSON logs with job context
- 🎨 **CLI Dashboard**: Real-time system status display
- 🔔 **Health Alerts**: Callback system for GPU failures

## 📁 Project Structure

```
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── src/
│   ├── scheduler/              # Job scheduling and queue management
│   │   ├── job_queue.py        # Celery-based job queue
│   │   ├── priority_manager.py # Priority scheduling logic
│   │   └── worker.py           # Celery worker management
│   ├── resources/              # GPU and resource management
│   │   ├── gpu_manager.py      # GPU allocation and monitoring
│   │   ├── resource_pool.py    # Resource pool management
│   │   └── health_checker.py   # Health monitoring and failover
│   ├── pipeline/               # ML training pipeline
│   │   ├── model.py            # Model loading and management
│   │   ├── data_loader.py      # Multi-source data loading
│   │   ├── trainer.py          # Training orchestration
│   │   └── checkpoint_manager.py # Checkpoint management
│   ├── monitoring/             # Monitoring and logging
│   │   ├── metrics.py          # Prometheus metrics
│   │   └── logger.py           # Structured logging
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration management
│       └── exceptions.py       # Custom exceptions
├── config/
│   ├── default.yaml            # Base configuration
│   ├── development.yaml        # Dev environment
│   └── production.yaml         # Production environment
├── examples/
│   ├── submit_training_job.py  # Job submission example
│   ├── production_pipeline.py  # Full pipeline demo
│   └── monitoring_dashboard.py # CLI monitoring tool
├── docs/
│   ├── architecture.md         # System architecture
│   ├── setup.md                # Installation guide
│   ├── technology_choices.md   # Technology rationale
│   ├── scaling_strategy.md     # Scaling to 100+ users
│   └── cost_analysis.md        # Cost breakdown
└── notebooks/                  # Jupyter notebooks
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/ml-platform.git
cd ml-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start Redis (required)
docker run -d -p 6379:6379 redis:latest
```

### Run a Training Job

```bash
# Terminal 1: Start Celery worker
celery -A src.scheduler.job_queue worker --loglevel=info

# Terminal 2: Submit job
python examples/submit_training_job.py

# Terminal 3: Monitor system
python examples/monitoring_dashboard.py --refresh 5
```

### Python API Example

```python
from src.scheduler.job_queue import JobConfig, get_job_queue, Priority

# Configure job
job = JobConfig(
    job_id="bert-finetune-001",
    user_id="researcher-1",
    num_gpus=2,
    model_name="bert-base-uncased",
    dataset_path="./data/train.csv",
    output_dir="./output"
)

# Submit to queue
queue = get_job_queue()
job_id = queue.submit_job(job, Priority.HIGH)
print(f"Job submitted: {job_id}")
```

### 🎬 Run Comprehensive Demo

**NEW!** Automated demo that showcases all features:

```bash
# Quick start with Docker Compose
docker-compose --profile demo up

# The demo automatically:
# ✅ Answers all 4 key questions
# ✅ Demonstrates priority scheduling
# ✅ Shows fault tolerance
# ✅ Explains scaling architecture
# ✅ Displays monitoring capabilities
# ✅ Submits and monitors a live job
```

**What you'll see:**
- 🖥️ System information (GPU status, health)
- 📋 Priority queue demonstration (HIGH > MEDIUM > LOW)
- 🛡️ Fault tolerance mechanisms
- 🚀 Scaling to 100+ users explanation
- 📈 Monitoring and observability
- 🎯 Live job submission and monitoring


**Comprehensive Monitoring Stack**:

| Component | Technology | Metrics |
|-----------|-----------|---------|
| **GPU Metrics** | Prometheus + pynvml | Utilization, memory, temperature, power |
| **Job Metrics** | Prometheus | Queue depth, wait times, success rates |
| **Training Metrics** | TensorBoard/WandB | Loss, accuracy, checkpoints |
| **System Logs** | Structured JSON | Job lifecycle, errors, resource allocation |
| **Health Checks** | Custom checker | GPU health, automatic failover |

**Prometheus Endpoint**: http://localhost:9090/metrics

## 🏗️ Architecture

```
┌─────────────┐
│   Users     │
└──────┬──────┘
       │
┌──────┴───────────────────────┐
│  Job Queue (Celery + Redis)  │  ← Priority scheduling
└──────┬───────────────────────┘
       │
┌──────┴───────────────────────┐
│  Resource Manager            │  ← GPU allocation, health checking
│  - GPU Manager               │
│  - Resource Pools            │
│  - Health Checker            │
└──────┬───────────────────────┘
       │
┌──────┴───────────────────────┐
│  Worker Nodes (Celery)       │  ← Distributed execution
│  ┌──────────┐  ┌──────────┐ │
│  │ Worker 1 │  │ Worker 2 │ │
│  │ 8x GPUs  │  │ 8x GPUs  │ │
│  └──────────┘  └──────────┘ │
└──────┬───────────────────────┘
       │
┌──────┴───────────────────────┐
│  ML Pipeline                 │  ← Training execution
│  - Model (BERT)              │
│  - Data Loader               │
│  - Trainer                   │
│  - Checkpoint Manager        │
└──────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for detailed design.

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and component overview |
| [Setup Guide](docs/setup.md) | Installation and configuration |
| [Technology Choices](docs/technology_choices.md) | Rationale for tech stack |
| [Scaling Strategy](docs/scaling_strategy.md) | 100+ user scaling design |
| [Cost Analysis](docs/cost_analysis.md) | Cloud and on-prem cost breakdown |

## 💰 Cost Analysis

**Cloud (AWS) - Optimized**:
- Production: Reserved instances (p4d.24xlarge, 8x A100)
- Development: Spot instances (p3.8xlarge, 4x V100)
- **Total**: ~$22,246/month ($266,952/year)



See [docs/cost_analysis.md](docs/cost_analysis.md) for full breakdown.

## 🛠️ Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| ML Framework | PyTorch + Transformers | Industry standard, extensive model support |
| Job Queue | Celery + Redis | Mature, scalable, Python-native |
| Distributed Compute | Ray | Built for ML, fault-tolerant |
| Monitoring | Prometheus | Industry standard, rich ecosystem |
| Configuration | Hydra + OmegaConf | Type-safe, hierarchical configs |
| Data Loading | HuggingFace Datasets | Multi-source support |

See [docs/technology_choices.md](docs/technology_choices.md) for rationale.

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Code formatting
black src/ tests/

# Type checking
mypy src/
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for efficient ML infrastructure**

# Setup and Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU(s) (NVIDIA)
- Redis server
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ml-platform.git
cd ml-platform
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -e ".[dev]"

# Cloud storage support (optional)
pip install -e ".[cloud]"

# Visualization tools (optional)
pip install -e ".[viz]"

# Jupyter support (optional)
pip install -e ".[jupyter]"
```

### 4. Install the Package

```bash
pip install -e .
```

## Redis Setup

### Local Development

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis
redis-server

# Verify Redis is running
redis-cli ping  # Should return "PONG"
```

### Docker (Recommended)

```bash
docker run -d -p 6379:6379 redis:latest
```

## Configuration

### 1. Copy Default Configuration

```bash
cp config/default.yaml config/local.yaml
```

### 2. Edit Configuration

Edit `config/local.yaml` to match your environment:

```yaml
resource:
  max_gpus_per_job: 2  # Adjust based on available GPUs

scheduler:
  redis_host: localhost  # Change if Redis is remote
  redis_port: 6379

data:
  data_dir: ./data  # Your data directory
```

### 3. Environment Variables (Optional)

```bash
export MLPLATFORM_SCHEDULER_REDIS_HOST=localhost
export MLPLATFORM_SCHEDULER_REDIS_PORT=6379
export MLPLATFORM_MONITORING_LOG_LEVEL=INFO
```

## Running the Platform

### 1. Start Celery Worker

```bash
# Terminal 1: Start worker
mlflow-worker --config development

# Or using celery directly
celery -A src.scheduler.job_queue worker --loglevel=info
```

### 2. Start Metrics Server (Optional)

```bash
# Terminal 2: Start Prometheus metrics server
python -m src.monitoring.metrics --port 9090
```

### 3. Submit a Job

```bash
# Terminal 3: Submit test job
python examples/submit_training_job.py
```

## Quick Start Example

```python
from src.scheduler.job_queue import JobConfig, get_job_queue
from src.scheduler.priority_manager import Priority

# Create job config
job = JobConfig(
    job_id="my-first-job",
    user_id="my-user",
    num_gpus=1,
    model_name="bert-base-uncased",
    dataset_path="./data/train.csv",
    output_dir="./output"
)

# Submit job
queue = get_job_queue()
job_id = queue.submit_job(job, Priority.MEDIUM)
print(f"Job submitted: {job_id}")
```

## Monitoring

### View Metrics

```bash
# View Prometheus metrics
curl http://localhost:9090/metrics

# View dashboard
python examples/monitoring_dashboard.py --refresh 5
```

### Check Logs

```bash
# View logs
tail -f logs/ml_platform.log

# View errors
tail -f logs/errors.log
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU info
nvidia-smi
```

### Redis Connection Failed

```bash
# Test Redis connection
redis-cli ping

# Check Redis is listening
netstat -an | grep 6379
```

### Import Errors

```bash
# Reinstall package
pip install -e .

# Verify installation
python -c "import src; print('OK')"
```



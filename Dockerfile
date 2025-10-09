# ML GPU Platform - Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY examples/ ./examples/

# Copy demo scripts
COPY demo_comprehensive.py .
COPY demo_submit_job.py .

# Install the package in development mode
RUN pip install -e .

# Create directories for data and output
RUN mkdir -p /app/data /app/output /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0

# Expose ports
EXPOSE 9090 8000

# Default command
CMD ["python", "-m", "celery", "-A", "src.scheduler.job_queue", "worker", "--loglevel=info"]

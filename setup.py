"""
MLFlow - ML Platform with GPU Resource Management
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mlflow-gpu-platform",
    version="0.1.0",
    author="ML Platform Team",
    description="GPU Resource Management Platform for ML Workloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "celery>=5.3.0",
        "redis>=4.5.0",
        "ray[default]>=2.5.0",
        "hydra-core>=1.3.0",
        "prometheus-client>=0.17.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "pynvml>=11.5.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "cloud": [
            "boto3>=1.26.0",
            "google-cloud-storage>=2.9.0",
            "s3fs>=2023.6.0",
            "gcsfs>=2023.6.0",
        ],
        "viz": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "jupyter": [
            "ipykernel>=6.24.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlflow-worker=scheduler.worker:main",
            "mlflow-submit=scheduler.job_queue:submit_job_cli",
            "mlflow-monitor=monitoring.metrics:main",
        ],
    },
)

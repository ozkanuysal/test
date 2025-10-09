class MLPlatformException(Exception):
    """Base exception for all ML Platform errors"""
    pass


class ResourceException(MLPlatformException):
    """Base exception for resource-related errors"""
    pass


class GPUNotAvailableError(ResourceException):
    """Raised when no GPU resources are available"""
    def __init__(self, requested_gpus: int, available_gpus: int):
        self.requested_gpus = requested_gpus
        self.available_gpus = available_gpus
        super().__init__(
            f"Requested {requested_gpus} GPUs but only {available_gpus} available"
        )


class GPUAllocationError(ResourceException):
    """Raised when GPU allocation fails"""
    pass


class ResourcePoolExhaustedError(ResourceException):
    """Raised when resource pool has no capacity"""
    def __init__(self, pool_name: str):
        self.pool_name = pool_name
        super().__init__(f"Resource pool '{pool_name}' is exhausted")


class JobException(MLPlatformException):
    """Base exception for job-related errors"""
    pass


class JobNotFoundError(JobException):
    """Raised when a job cannot be found"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Job '{job_id}' not found")


class JobSubmissionError(JobException):
    """Raised when job submission fails"""
    pass


class JobExecutionError(JobException):
    """Raised when job execution fails"""
    def __init__(self, job_id: str, reason: str):
        self.job_id = job_id
        self.reason = reason
        super().__init__(f"Job '{job_id}' execution failed: {reason}")


class JobTimeoutError(JobException):
    """Raised when job exceeds timeout"""
    def __init__(self, job_id: str, timeout: int):
        self.job_id = job_id
        self.timeout = timeout
        super().__init__(f"Job '{job_id}' exceeded timeout of {timeout}s")


class CheckpointException(MLPlatformException):
    """Base exception for checkpoint-related errors"""
    pass


class CheckpointNotFoundError(CheckpointException):
    """Raised when checkpoint cannot be found"""
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        super().__init__(f"Checkpoint not found at '{checkpoint_path}'")


class CheckpointLoadError(CheckpointException):
    """Raised when checkpoint loading fails"""
    pass


class CheckpointSaveError(CheckpointException):
    """Raised when checkpoint saving fails"""
    pass


class ConfigurationException(MLPlatformException):
    """Base exception for configuration-related errors"""
    pass


class InvalidConfigError(ConfigurationException):
    """Raised when configuration is invalid"""
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Invalid configuration for '{field}': {reason}")


class ConfigNotFoundError(ConfigurationException):
    """Raised when configuration file is not found"""
    pass


class DataException(MLPlatformException):
    """Base exception for data-related errors"""
    pass


class DataLoadError(DataException):
    """Raised when data loading fails"""
    pass


class DataValidationError(DataException):
    """Raised when data validation fails"""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Data validation failed: {reason}")


class ModelException(MLPlatformException):
    """Base exception for model-related errors"""
    pass


class ModelLoadError(ModelException):
    """Raised when model loading fails"""
    pass


class ModelSaveError(ModelException):
    """Raised when model saving fails"""
    pass


class TrainingException(MLPlatformException):
    """Base exception for training-related errors"""
    pass


class TrainingFailedError(TrainingException):
    """Raised when training fails"""
    def __init__(self, reason: str, epoch: int = None):
        self.reason = reason
        self.epoch = epoch
        msg = f"Training failed: {reason}"
        if epoch is not None:
            msg += f" at epoch {epoch}"
        super().__init__(msg)


class MonitoringException(MLPlatformException):
    """Base exception for monitoring-related errors"""
    pass


class MetricsCollectionError(MonitoringException):
    """Raised when metrics collection fails"""
    pass

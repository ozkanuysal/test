import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
import logging

from .exceptions import ConfigurationException, InvalidConfigError, ConfigNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """GPU and resource configuration"""
    max_gpus_per_job: int = 4
    development_pool_size: int = 2
    production_pool_size: int = 8
    gpu_memory_fraction: float = 0.9
    enable_mixed_precision: bool = True


@dataclass
class SchedulerConfig:
    """Job scheduler configuration"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    max_retries: int = 3
    retry_backoff_base: int = 2
    job_timeout: int = 86400  # 24 hours
    queue_name: str = "ml_jobs"
    worker_concurrency: int = 4
    priority_levels: int = 3


@dataclass
class TrainingConfig:
    """ML training configuration"""
    model_name: str = "bert-base-uncased"
    max_seq_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    prometheus_port: int = 9090
    log_level: str = "INFO"
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = "ml-platform"
    metrics_collection_interval: int = 60


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    use_gcs: bool = False
    gcs_bucket: Optional[str] = None
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class PlatformConfig:
    """Complete platform configuration"""
    environment: str = "development"
    debug: bool = False
    resource: ResourceConfig = None
    scheduler: SchedulerConfig = None
    training: TrainingConfig = None
    monitoring: MonitoringConfig = None
    data: DataConfig = None

    def __post_init__(self):
        """Initialize nested configs if not provided"""
        if self.resource is None:
            self.resource = ResourceConfig()
        if self.scheduler is None:
            self.scheduler = SchedulerConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.data is None:
            self.data = DataConfig()


class ConfigManager:
    """
    Configuration manager for the ML Platform
    Handles loading from YAML files and environment variables
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to project root config directory
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self._config: Optional[DictConfig] = None
        self._platform_config: Optional[PlatformConfig] = None

    def load_config(
        self,
        config_name: str = "default",
        overrides: Optional[Dict[str, Any]] = None
    ) -> DictConfig:
        """
        Load configuration from YAML file

        Args:
            config_name: Name of config file (without .yaml extension)
            overrides: Dictionary of config overrides

        Returns:
            Loaded configuration

        Raises:
            ConfigNotFoundError: If config file not found
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise ConfigNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        try:
            # Load base configuration
            config = OmegaConf.load(config_path)

            # Apply environment variable overrides
            env_config = self._load_from_env()
            config = OmegaConf.merge(config, env_config)

            # Apply manual overrides
            if overrides:
                override_config = OmegaConf.create(overrides)
                config = OmegaConf.merge(config, override_config)

            self._config = config
            logger.info(f"Loaded configuration from {config_path}")

            return config

        except Exception as e:
            raise ConfigurationException(
                f"Failed to load configuration: {str(e)}"
            )

    def _load_from_env(self) -> DictConfig:
        """
        Load configuration overrides from environment variables

        Environment variables should be prefixed with MLPLATFORM_
        Example: MLPLATFORM_SCHEDULER_REDIS_HOST=redis.example.com

        Returns:
            Configuration from environment variables
        """
        env_config = {}
        prefix = "MLPLATFORM_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix):].lower()
                parts = config_key.split("_")

                # Build nested dictionary
                current = env_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Try to parse value as int, float, or bool
                parsed_value = self._parse_env_value(value)
                current[parts[-1]] = parsed_value

        return OmegaConf.create(env_config)

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get_platform_config(self) -> PlatformConfig:
        """
        Get strongly-typed platform configuration

        Returns:
            PlatformConfig object
        """
        if self._config is None:
            raise ConfigurationException(
                "Configuration not loaded. Call load_config() first."
            )

        if self._platform_config is None:
            # Convert OmegaConf to structured config
            config_dict = OmegaConf.to_container(self._config, resolve=True)

            self._platform_config = PlatformConfig(
                environment=config_dict.get("environment", "development"),
                debug=config_dict.get("debug", False),
                resource=ResourceConfig(**config_dict.get("resource", {})),
                scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
                training=TrainingConfig(**config_dict.get("training", {})),
                monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
                data=DataConfig(**config_dict.get("data", {}))
            )

        return self._platform_config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key

        Args:
            key: Configuration key (e.g., 'scheduler.redis_host')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self._config is None:
            raise ConfigurationException(
                "Configuration not loaded. Call load_config() first."
            )

        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if valid

        Raises:
            InvalidConfigError: If configuration is invalid
        """
        if self._config is None:
            raise ConfigurationException(
                "Configuration not loaded. Call load_config() first."
            )

        platform_config = self.get_platform_config()

        # Validate resource config
        if platform_config.resource.max_gpus_per_job < 1:
            raise InvalidConfigError(
                "resource.max_gpus_per_job",
                "Must be at least 1"
            )

        if platform_config.resource.gpu_memory_fraction <= 0 or \
           platform_config.resource.gpu_memory_fraction > 1:
            raise InvalidConfigError(
                "resource.gpu_memory_fraction",
                "Must be between 0 and 1"
            )

        # Validate scheduler config
        if platform_config.scheduler.max_retries < 0:
            raise InvalidConfigError(
                "scheduler.max_retries",
                "Must be non-negative"
            )

        # Validate training config
        if platform_config.training.batch_size < 1:
            raise InvalidConfigError(
                "training.batch_size",
                "Must be at least 1"
            )

        if platform_config.training.learning_rate <= 0:
            raise InvalidConfigError(
                "training.learning_rate",
                "Must be positive"
            )

        logger.info("Configuration validation passed")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary

        Returns:
            Configuration as dictionary
        """
        if self._config is None:
            raise ConfigurationException(
                "Configuration not loaded. Call load_config() first."
            )

        return OmegaConf.to_container(self._config, resolve=True)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(
    config_name: str = "default",
    overrides: Optional[Dict[str, Any]] = None
) -> PlatformConfig:
    """
    Convenience function to load and get platform configuration

    Args:
        config_name: Name of config file
        overrides: Configuration overrides

    Returns:
        PlatformConfig object
    """
    manager = get_config_manager()
    manager.load_config(config_name, overrides)
    manager.validate()
    return manager.get_platform_config()

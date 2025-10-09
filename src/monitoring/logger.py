import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    Outputs logs in JSON format for easy parsing
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "job_id"):
            log_data["job_id"] = record.job_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "gpu_id"):
            log_data["gpu_id"] = record.gpu_id

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output
    Adds color codes for different log levels
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors

        Args:
            record: Log record to format

        Returns:
            Colored log string
        """
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            )

        # Format the message
        formatted = super().format(record)

        # Reset levelname for future use
        record.levelname = levelname

        return formatted


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_json: bool = False,
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the ML platform

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to ./logs)
        enable_json: Enable JSON formatting for file logs
        enable_console: Enable console logging
        enable_file: Enable file logging

    Returns:
        Root logger instance
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Use colored formatter for console
        console_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(message)s [%(filename)s:%(lineno)d]"
        )
        console_formatter = ColoredFormatter(
            console_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    if enable_file:
        if log_dir is None:
            log_dir = Path("./logs")

        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file (all messages)
        if enable_json:
            # JSON format for structured logging
            main_log = log_dir / "ml_platform.json"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            main_handler.setFormatter(JSONFormatter())
        else:
            # Standard text format
            main_log = log_dir / "ml_platform.log"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            log_format = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(message)s [%(filename)s:%(lineno)d]"
            )
            main_handler.setFormatter(
                logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
            )

        main_handler.setLevel(level)
        root_logger.addHandler(main_handler)

        # Error log file (errors and critical only)
        error_log = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n"
            "%(pathname)s:%(lineno)d in %(funcName)s\n"
            "%(exc_info)s\n"
        )
        error_handler.setFormatter(
            logging.Formatter(error_format, datefmt="%Y-%m-%d %H:%M:%S")
        )
        root_logger.addHandler(error_handler)

    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    logging.info(f"Logging initialized at level {level}")

    return root_logger


class JobLogger:
    """
    Logger wrapper for job-specific logging
    Automatically adds job context to log messages
    """

    def __init__(self, job_id: str, user_id: str, logger_name: Optional[str] = None):
        """
        Initialize job logger

        Args:
            job_id: Job identifier
            user_id: User identifier
            logger_name: Logger name (defaults to module name)
        """
        self.job_id = job_id
        self.user_id = user_id
        self.logger = logging.getLogger(logger_name or __name__)

    def _add_context(self, extra: Optional[dict] = None) -> dict:
        """Add job context to extra fields"""
        context = {"job_id": self.job_id, "user_id": self.user_id}
        if extra:
            context.update(extra)
        return context

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with job context"""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.debug(msg, *args, extra=extra, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message with job context"""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.info(msg, *args, extra=extra, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with job context"""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.warning(msg, *args, extra=extra, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message with job context"""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.error(msg, *args, extra=extra, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with job context"""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.critical(msg, *args, extra=extra, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception with job context"""
        extra = self._add_context(kwargs.pop("extra", None))
        self.logger.exception(msg, *args, extra=extra, **kwargs)


def get_job_logger(job_id: str, user_id: str, logger_name: Optional[str] = None) -> JobLogger:
    """
    Get a job-specific logger

    Args:
        job_id: Job identifier
        user_id: User identifier
        logger_name: Logger name

    Returns:
        JobLogger instance
    """
    return JobLogger(job_id, user_id, logger_name)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level="DEBUG", enable_json=False)

    # Regular logging
    logger = logging.getLogger(__name__)
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Job-specific logging
    job_logger = get_job_logger("job-123", "user-456", __name__)
    job_logger.info("Job started")
    job_logger.warning("GPU temperature high")
    job_logger.error("Job failed")

    try:
        raise ValueError("Test exception")
    except Exception:
        job_logger.exception("Exception occurred during job execution")

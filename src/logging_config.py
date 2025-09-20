# src/logging_config.py
import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from structlog import get_logger

# Configure structlog for better logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Project root for log files
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> None:
    """Setup comprehensive logging configuration."""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)
    else:
        # Default file handler
        file_handler = logging.FileHandler(LOG_DIR / "paraphrase_system.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return get_logger(name)

# Error handling decorator
def with_error_handling(logger_name: str = "error_handler"):
    """Decorator to add comprehensive error handling to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise
        return wrapper
    return decorator

# Performance monitoring decorator
def with_performance_logging(logger_name: str = "performance"):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            logger = get_logger(logger_name)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    "Function completed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s",
                    error=str(e)
                )
                raise
        return wrapper
    return decorator
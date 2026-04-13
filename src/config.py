"""
Configuration management system for the paraphrase generation application.

This module provides comprehensive configuration management including environment
variables, file-based configuration, model resolution, and component factory
functions. It uses OpenRouter's chat completions API for paraphrase generation.

Key Features:
- Environment variable configuration with fallbacks
- File-based configuration (API keys, model names)
- Dynamic model resolution
- Component factory functions for dependency injection
- Path management and project structure configuration
- Feature flags and system settings
- Configuration validation and error handling
"""
# src/config.py
from pathlib import Path
import os
from typing import Optional

# --- General ---
SEED = 42

# --- Paths ---
# Determine project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "model_output"

# Custom Dataset Files
CUSTOM_TRAIN_FILE = "custom_train.tsv"
CUSTOM_EVAL_FILE = "custom_eval.tsv"

# --- Data Processing ---
DEFAULT_MAX_LEN = 128  # Default max sequence length for tokenizers (adjust as needed)

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY_FILE_PATH = Path("~/.api-openrouter").expanduser()
DEFAULT_OPENROUTER_MODEL_NAME = "openrouter/free"
OPENROUTER_MODEL_FILE_PATH = Path("~/.model-openrouter").expanduser()

def _read_text_file(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            v = path.read_text(encoding="utf-8").strip()
            return v if v else None
    except Exception:
        return None
    return None

def resolve_openrouter_model_name() -> str:
    """
    Resolution order for OpenRouter model:
    1) Env MODEL_OPENROUTER if set and non-empty
    2) ~/.model-openrouter file single line
    3) DEFAULT_OPENROUTER_MODEL_NAME
    """
    env_val = os.getenv("MODEL_OPENROUTER")
    if env_val and env_val.strip():
        return env_val.strip()
    file_val = _read_text_file(OPENROUTER_MODEL_FILE_PATH)
    if file_val:
        return file_val
    return DEFAULT_OPENROUTER_MODEL_NAME

# --- Training Defaults (for Hugging Face Trainer) ---
# These are now obsolete but kept for reference during refactoring
# DEFAULT_LEARNING_RATE = 2e-5
# DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
# DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 8
# DEFAULT_NUM_TRAIN_EPOCHS = 3
# DEFAULT_WEIGHT_DECAY = 0.01
# DEFAULT_LOGGING_STEPS = 10

# --- Model ---
# Default model checkpoint (now OpenRouter model name)
# DEFAULT_MODEL_CHECKPOINT = "t5-small"

# --- New System Configuration ---

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = False
API_WORKERS = 1

# Cache Configuration
CACHE_TYPE = "redis"  # "redis" or "memory"
REDIS_URL = "redis://localhost:6379/0"
CACHE_TTL = 3600  # Default TTL in seconds

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_TOKENS_PER_MINUTE = 100000

# Batch Processing Configuration
BATCH_MAX_SIZE = 10
BATCH_MAX_WORKERS = 4
BATCH_MAX_RETRIES = 3
BATCH_RETRY_DELAY = 1.0

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = None  # None for console only, or path to file
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Evaluation Configuration
EVALUATION_INCLUDE_SEMANTIC = True
EVALUATION_MODEL_CACHE_SIZE = 1000

# Security Configuration
API_KEY_REQUIRED = False
ALLOWED_ORIGINS = ["*"]  # Configure for production

# Monitoring Configuration
SENTRY_DSN = None  # Set to enable Sentry error tracking
METRICS_ENABLED = False

# Feature Flags
ENABLE_CACHING = True
ENABLE_RATE_LIMITING = True
ENABLE_EVALUATION = True
ENABLE_BATCH_PROCESSING = True

# --- Component Factory Functions ---

def get_cache():
    """Get configured cache instance."""
    if not ENABLE_CACHING:
        from src.cache import MemoryCache
        return MemoryCache()

    from src.cache import get_cache as get_cache_instance
    return get_cache_instance()

def get_rate_limiter():
    """Get configured rate limiter instance."""
    if not ENABLE_RATE_LIMITING:
        # Return a no-op rate limiter
        class NoOpRateLimiter:
            def wait_if_needed(self, *args, **kwargs): pass
            def check_request(self, *args, **kwargs): return True, 0.0
        return NoOpRateLimiter()

    from src.rate_limiter import get_rate_limiter as get_rate_limiter_instance
    return get_rate_limiter_instance()

def get_batch_processor():
    """Get configured batch processor instance."""
    if not ENABLE_BATCH_PROCESSING:
        # Return a simple processor
        class SimpleProcessor:
            def process_sync(self, texts): return [f"Processed: {text}" for text in texts]
            def process_with_cache(self, texts): return self.process_sync(texts)
        return SimpleProcessor()

    from src.batch_processor import get_batch_processor as get_batch_processor_instance
    return get_batch_processor_instance()

def get_evaluator():
    """Get configured evaluator instance."""
    if not ENABLE_EVALUATION:
        # Return a simple evaluator
        class SimpleEvaluator:
            def evaluate_paraphrase(self, orig, para): return {"overall_score": 0.5}
            def evaluate_paraphrase_batch(self, origs, paras): return [{"overall_score": 0.5} for _ in origs]
        return SimpleEvaluator()

    from src.evaluation import get_evaluator as get_evaluator_instance
    return get_evaluator_instance()

def setup_system():
    """Setup all system components with proper configuration."""
    from src.logging_config import setup_logging

    # Setup logging
    setup_logging(
        level=LOG_LEVEL,
        log_file=LOG_FILE
    )

    # Initialize components
    if ENABLE_CACHING:
        get_cache()

    if ENABLE_RATE_LIMITING:
        get_rate_limiter()

    if ENABLE_BATCH_PROCESSING:
        get_batch_processor()

    if ENABLE_EVALUATION:
        get_evaluator()

    print("System components initialized successfully")

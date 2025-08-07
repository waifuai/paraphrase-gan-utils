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

# --- Gemini API Configuration ---
GEMINI_API_KEY_PATH = Path("~/.api-gemini").expanduser()
# Default Gemini model if no override files/env present
DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-pro"
GEMINI_MODEL_FILE_PATH = Path("~/.model-gemini").expanduser()

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY_FILE_PATH = Path("~/.api-openrouter").expanduser()
DEFAULT_OPENROUTER_MODEL_NAME = "openrouter/horizon-beta"
OPENROUTER_MODEL_FILE_PATH = Path("~/.model-openrouter").expanduser()

def _read_text_file(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            v = path.read_text(encoding="utf-8").strip()
            return v if v else None
    except Exception:
        return None
    return None

def resolve_gemini_model_name() -> str:
    """
    Resolution order for Gemini model:
    1) Env MODEL_GEMINI if set and non-empty
    2) ~/.model-gemini file single line
    3) DEFAULT_GEMINI_MODEL_NAME
    """
    env_val = os.getenv("MODEL_GEMINI")
    if env_val and env_val.strip():
        return env_val.strip()
    file_val = _read_text_file(GEMINI_MODEL_FILE_PATH)
    if file_val:
        return file_val
    return DEFAULT_GEMINI_MODEL_NAME

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

def load_gemini_api_key() -> str:
    """
    Loads the Gemini API key using fallback file when env vars are not set.

    Preferred sources are environment variables GEMINI_API_KEY or GOOGLE_API_KEY
    which are handled in gemini_api.initialize_gemini_api. This function provides
    the legacy file fallback to maintain compatibility.
    """
    try:
        with open(GEMINI_API_KEY_PATH, "r") as f:
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty.")
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"Gemini API key file not found at {GEMINI_API_KEY_PATH}")
    except Exception as e:
        raise RuntimeError(f"Error loading Gemini API key: {e}")

# Backwards compatibility alias used by gemini_api
GEMINI_MODEL_NAME = resolve_gemini_model_name()

# --- Training Defaults (for Hugging Face Trainer) ---
# These are now obsolete but kept for reference during refactoring
# DEFAULT_LEARNING_RATE = 2e-5
# DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
# DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 8
# DEFAULT_NUM_TRAIN_EPOCHS = 3
# DEFAULT_WEIGHT_DECAY = 0.01
# DEFAULT_LOGGING_STEPS = 10

# --- Model ---
# Default model checkpoint (now Gemini model name)
# DEFAULT_MODEL_CHECKPOINT = "t5-small"

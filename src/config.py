# src/config.py
from pathlib import Path
import os

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
GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-03-25"

def load_gemini_api_key() -> str:
    """Loads the Gemini API key from the specified file path."""
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

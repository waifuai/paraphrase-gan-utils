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

# --- Training Defaults (for Hugging Face Trainer) ---
# These can be overridden by flags in main.py
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 8
DEFAULT_NUM_TRAIN_EPOCHS = 3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOGGING_STEPS = 10

# --- Model ---
# Default model checkpoint (can be overridden by flag in main.py)
DEFAULT_MODEL_CHECKPOINT = "t5-small"

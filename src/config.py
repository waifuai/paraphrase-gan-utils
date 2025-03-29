# src/config.py
from pathlib import Path
import os

# --- General ---
SEED = 42

# --- Paths ---
# Determine project root dynamically
# Assuming this file (config.py) is in src/
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = (
    PROJECT_ROOT / "model_output"
)  # Renamed from 'model' to avoid conflict with module name
CHUNK_DIR = DATA_DIR / "tmp"  # For multicore processing


# Dataset-specific subdirectories (can be overridden by main script logic)
# Using functions to allow dynamic creation if needed, though direct paths are fine too
def get_coco_char_data_dir():
    return DATA_DIR / "coco_char"


def get_coco_word_data_dir():
    return DATA_DIR / "coco_word"


def get_parabank_data_dir():
    # Parabank files expected directly in DATA_DIR by default
    return DATA_DIR


# Vocabulary files (relative to their respective data dirs)
COCO_CHAR_VOCAB_FILE = "vocab_char.txt"
COCO_WORD_VOCAB_FILE = "vocab_word.txt"
PARABANK_VOCAB_FILE = "vocab.subword"

# Data files (relative to their respective data dirs)
COCO_CHAR_TRAIN_FILE = "train_char.txt"
COCO_CHAR_VAL_FILE = "val_char.txt"
COCO_WORD_TRAIN_FILE = "train_word.txt"
COCO_WORD_VAL_FILE = "val_word.txt"
PARABANK_RAW_FILE = "parabank.tsv"  # Original parabank file
PARABANK_PROCESSED_FILE = f"processed_{PARABANK_RAW_FILE}"
PARABANK_TRAIN_FILE = "train.tsv"
PARABANK_EVAL_FILE = "eval.tsv"


# --- Data Processing ---
DEFAULT_MAX_LEN = 256  # Default max sequence length
PARABANK_EVAL_RATIO = 0.1
PARABANK_VOCAB_SAMPLE_SIZE = 1000
PARABANK_VOCAB_SIZE = 32000  # For subword vocab generation

# --- Training ---
DEFAULT_TRAIN_STEPS = 1000
DEFAULT_EVAL_STEPS = 10  # Number of steps per eval batch, or batches per eval run
DEFAULT_BATCH_SIZE = 128  # Default, might be overridden by dataset/model specifics
DEFAULT_LEARNING_RATE = 0.001  # Default, might be overridden
DEFAULT_N_STEPS_PER_CHECKPOINT = 500  # Default checkpoint frequency

# Specific overrides (examples, main.py will select based on task)
# COCO specific training params
COCO_TRAIN_STEPS = 1000
COCO_EVAL_STEPS = 10  # Used as n_steps_per_checkpoint and eval batches
COCO_BATCH_SIZE = 128
COCO_LEARNING_RATE = 0.0005

# Multicore/Parabank specific training params
MULTICORE_TRAIN_STEPS = 1000
MULTICORE_EVAL_STEPS = 10  # n_eval_batches
MULTICORE_BATCH_SIZE = 32
MULTICORE_LEARNING_RATE = 0.001
MULTICORE_N_STEPS_PER_CHECKPOINT = 500

# Phrase Generator specific training params
PHRASE_GENERATOR_TRAIN_STEPS = (
    1000  # Original script used 10000, using default for consistency
)
PHRASE_GENERATOR_EVAL_STEPS = 10  # Assuming n_eval_batches
PHRASE_GENERATOR_BATCH_SIZE = 32
PHRASE_GENERATOR_LEARNING_RATE = 0.05
PHRASE_GENERATOR_N_STEPS_PER_CHECKPOINT = 500

# --- Model Hyperparameters ---

# COCO Transformer Defaults
COCO_MODEL_NAME = "transformer"  # or 'transformer_encoder'
COCO_D_MODEL = 512
COCO_D_FF = 2048
COCO_N_HEADS = 8
COCO_N_ENCODER_LAYERS = 6
COCO_N_DECODER_LAYERS = 6
COCO_DROPOUT = 0.1

# Multicore/Parabank Transformer Defaults
# Note: vocab_size is determined dynamically or from PARABANK_VOCAB_SIZE
MULTICORE_D_MODEL = 512
MULTICORE_D_FF = 2048
MULTICORE_N_HEADS = 8
MULTICORE_N_ENCODER_LAYERS = 6
MULTICORE_N_DECODER_LAYERS = 6

# Phrase Generator Transformer Defaults
# Note: vocab_size is determined dynamically
PHRASE_GENERATOR_D_MODEL = 128
PHRASE_GENERATOR_D_FF = 512
PHRASE_GENERATOR_N_HEADS = 4
PHRASE_GENERATOR_N_ENCODER_LAYERS = 2
PHRASE_GENERATOR_N_DECODER_LAYERS = 2

# --- Multiprocessing ---
# Use all available cores by default, or 1 if detection fails
NUM_PROCESSES = os.cpu_count() or 1

# src/main.py
from pathlib import Path
import numpy as np

from absl import app
from absl import flags
from absl import logging

# Import refactored modules
import config
import gemini_api # Import the new gemini_api module

FLAGS = flags.FLAGS

# --- Flag Definitions ---

flags.DEFINE_enum(
    "task",
    None,
    ["train_custom", "decode_custom"], # train_custom will now be a validation step
    "The specific task to run: train (validate API) or decode using the custom dataset.",
)
flags.mark_flag_as_required("task")

# Remove Hugging Face specific flags
# flags.DEFINE_string(
#     "model_checkpoint",
#     "t5-small",
#     "Hugging Face model checkpoint name (e.g., 't5-small', 'google/bart-base').",
# )
flags.DEFINE_string(
    "output_dir",
    str(config.OUTPUT_DIR),
    "Base directory for output (less relevant now, but keep for consistency).",
)
flags.DEFINE_string(
    "custom_train_file",
    str(config.DATA_DIR / config.CUSTOM_TRAIN_FILE),
    "Path to the custom training data TSV file (used for data structure reference if needed).",
)
flags.DEFINE_string(
    "custom_eval_file",
    str(config.DATA_DIR / config.CUSTOM_EVAL_FILE),
    "Path to the custom evaluation data TSV file (used for data structure reference if needed).",
)

# --- Overrides for Config/Training Values ---
# These are now obsolete
# flags.DEFINE_integer(
#     "max_source_length",
#     config.DEFAULT_MAX_LEN,
#     "Maximum sequence length for source inputs.",
# )
# flags.DEFINE_integer(
#     "max_target_length",
#     config.DEFAULT_MAX_LEN,
#     "Maximum sequence length for target outputs.",
# )
flags.DEFINE_integer(
    "seed", config.SEED, "Random seed for reproducibility." # Keep seed for general randomness if needed
)

# Training arguments overrides (obsolete)
# flags.DEFINE_float("learning_rate", None, "Override learning rate for Trainer.")
# flags.DEFINE_integer(
#     "per_device_train_batch_size", None, "Override train batch size per device."
# )
# flags.DEFINE_integer(
#     "per_device_eval_batch_size", None, "Override eval batch size per device."
# )
# flags.DEFINE_integer("num_train_epochs", None, "Override number of training epochs.")
# flags.DEFINE_float("weight_decay", None, "Override weight decay.")
# flags.DEFINE_integer("logging_steps", None, "Override logging frequency.")

# Decode specific flags
flags.DEFINE_string("decode_input", None, "Input sentence for the decode_custom task.")
# Remove decode_checkpoint_dir as it's not needed for API calls
# flags.DEFINE_string(
#     "decode_checkpoint_dir",
#     None,
#     "Directory containing the trained model checkpoint for decoding (usually the task output_dir). Required for decode.",
# )


def main(argv):
    del argv  # Unused.
    # set_seed(FLAGS.seed) # Remove transformers set_seed
    logging.set_verbosity(logging.INFO)

    task = FLAGS.task
    # Output directory is less critical now, but keep for potential logs
    task_output_dir = Path(FLAGS.output_dir) / f"{task}_gemini" # Use gemini slug
    task_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running task: {task}")
    print(f"Output directory: {task_output_dir}")

    # --- Task Execution ---

    if task == "train_custom":
        # --- Validate API Connection Task ---
        print("Initializing and validating Gemini API connection...")
        try:
            gemini_api.initialize_gemini_api()
            # Optional: Make a small test call to validate
            test_response = gemini_api.generate_paraphrase("This is a test.")
            print(f"API test successful. Response sample: {test_response[:50]}...")
            print("Gemini API connection validated.")
        except Exception as e:
            print(f"Error validating Gemini API connection: {e}")
            print("Please ensure your API key is correctly placed at ~/.api-gemini")


    elif task == "decode_custom":
        # --- Decode Custom Task ---
        if not FLAGS.decode_input:
            raise ValueError("Flag --decode_input is required for task 'decode_custom'.")

        input_sentence = FLAGS.decode_input

        print("Initializing Gemini API...")
        gemini_api.initialize_gemini_api()

        print(f"Generating paraphrase for: '{input_sentence}' using {config.GEMINI_MODEL_NAME}")
        decoded_output = gemini_api.generate_paraphrase(input_sentence)

        print("-" * 40)
        print(f"Input:  {input_sentence}")
        print(f"Output: {decoded_output}")
        print("-" * 40)

    else:
        print(f"Unknown task: {task}")


if __name__ == "__main__":
    app.run(main)

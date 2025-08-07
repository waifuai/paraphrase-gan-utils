# src/main.py
from pathlib import Path
import numpy as np

from absl import app
from absl import flags
from absl import logging

# Import refactored modules
import config
import gemini_api  # existing gemini module
from provider_facade import generate_paraphrase, DEFAULT_PROVIDER

FLAGS = flags.FLAGS

# --- Flag Definitions ---

flags.DEFINE_enum(
    "task",
    None,
    ["train_custom", "decode_custom"],  # train_custom now validates provider connectivity
    "The specific task to run: train validate API or decode using the custom dataset.",
)
flags.mark_flag_as_required("task")

flags.DEFINE_string(
    "output_dir",
    str(config.OUTPUT_DIR),
    "Base directory for output.",
)
flags.DEFINE_string(
    "custom_train_file",
    str(config.DATA_DIR / config.CUSTOM_TRAIN_FILE),
    "Path to the custom training data TSV file.",
)
flags.DEFINE_string(
    "custom_eval_file",
    str(config.DATA_DIR / config.CUSTOM_EVAL_FILE),
    "Path to the custom evaluation data TSV file.",
)

flags.DEFINE_integer(
    "seed", config.SEED, "Random seed for reproducibility."
)

# New provider and model flags
flags.DEFINE_enum(
    "provider",
    DEFAULT_PROVIDER,
    ["openrouter", "gemini"],
    "Provider backend to use. Defaults to openrouter.",
)
flags.DEFINE_string(
    "model",
    None,
    "Optional explicit model name to override resolution from dotfiles and defaults.",
)

# Decode specific flags
flags.DEFINE_string("decode_input", None, "Input sentence for the decode_custom task.")

def main(argv):
    del argv  # Unused.
    logging.set_verbosity(logging.INFO)

    task = FLAGS.task
    task_output_dir = Path(FLAGS.output_dir) / f"{task}_{FLAGS.provider}"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running task: {task}")
    print(f"Output directory: {task_output_dir}")
    print(f"Provider: {FLAGS.provider}  Model: {FLAGS.model or '(auto-resolve)'}")

    # --- Task Execution ---

    if task == "train_custom":
        # Validate provider connectivity by making a tiny paraphrase call
        try:
            sample = generate_paraphrase("This is a test.", provider=FLAGS.provider, model=FLAGS.model)
            print(f"API test sample: {sample[:80] if isinstance(sample, str) else sample}")
            print(f"{FLAGS.provider} API connectivity validated.")
        except Exception as e:
            print(f"Error validating {FLAGS.provider} API connection: {e}")
            if FLAGS.provider == "gemini":
                print("Please ensure your API key is correctly placed at ~/.api-gemini or env var set.")
            else:
                print("Please ensure your API key is set in OPENROUTER_API_KEY or ~/.api-openrouter.")

    elif task == "decode_custom":
        if not FLAGS.decode_input:
            raise ValueError("Flag --decode_input is required for task 'decode_custom'.")

        input_sentence = FLAGS.decode_input

        print(f"Generating paraphrase for: '{input_sentence}'")
        decoded_output = generate_paraphrase(input_sentence, provider=FLAGS.provider, model=FLAGS.model)

        print("-" * 40)
        print(f"Input:  {input_sentence}")
        print(f"Output: {decoded_output}")
        print("-" * 40)

    else:
        print(f"Unknown task: {task}")

if __name__ == "__main__":
    app.run(main)

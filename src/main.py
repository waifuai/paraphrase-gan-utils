# src/main.py
from pathlib import Path
import numpy as np
import sys

from absl import app
from absl import flags
from absl import logging

# Import refactored modules
import config
import gemini_api  # existing gemini module
from provider_facade import generate_paraphrase, DEFAULT_PROVIDER

# Import new enhanced modules
from config import setup_system, get_batch_processor, get_evaluator
from logging_config import get_logger
from exceptions import ValidationError
from batch_processor import paraphrase_batch
from evaluation import evaluate_paraphrase
from cache import get_cache
from rate_limiter import get_rate_limiter

# Setup logging
logger = get_logger("main")

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

# New enhanced flags
flags.DEFINE_enum(
    "mode",
    "cli",
    ["cli", "api", "batch", "evaluate", "interactive"],
    "Operation mode: cli (single), api (server), batch (multiple), evaluate (quality), interactive (shell)"
)
flags.DEFINE_string("batch_input", None, "Path to file with texts to process in batch mode")
flags.DEFINE_string("batch_output", None, "Path to save batch results")
flags.DEFINE_string("evaluate_original", None, "Original text for evaluation")
flags.DEFINE_string("evaluate_paraphrase", None, "Paraphrased text for evaluation")
flags.DEFINE_boolean("include_semantic", True, "Include semantic similarity in evaluation")
flags.DEFINE_boolean("setup_only", False, "Only setup system components and exit")
flags.DEFINE_boolean("verbose", False, "Enable verbose output")
flags.DEFINE_integer("port", config.API_PORT, "Port for API server")

def setup_and_validate():
    """Setup system components and validate configuration."""
    try:
        setup_system()
        logger.info("System setup completed")

        # Test API connectivity
        if FLAGS.task == "train_custom":
            sample = generate_paraphrase("This is a test.", provider=FLAGS.provider, model=FLAGS.model)
            print(f"API test sample: {sample[:80] if isinstance(sample, str) else sample}")
            print(f"{FLAGS.provider} API connectivity validated.")

    except Exception as e:
        logger.error("System setup failed", error=str(e))
        if FLAGS.provider == "gemini":
            print("Please ensure your API key is correctly placed at ~/.api-gemini or env var set.")
        else:
            print("Please ensure your API key is set in OPENROUTER_API_KEY or ~/.api-openrouter.")
        sys.exit(1)

def handle_single_paraphrase():
    """Handle single paraphrase generation (legacy decode_custom task)."""
    if not FLAGS.decode_input:
        raise ValidationError("decode_input", FLAGS.decode_input, "Required for single paraphrase")

    input_sentence = FLAGS.decode_input.strip()
    if not input_sentence:
        raise ValidationError("decode_input", input_sentence, "Cannot be empty")

    print(f"Generating paraphrase for: '{input_sentence}'")
    decoded_output = generate_paraphrase(input_sentence, provider=FLAGS.provider, model=FLAGS.model)

    print("-" * 40)
    print(f"Input:  {input_sentence}")
    print(f"Output: {decoded_output}")
    print("-" * 40)

def handle_batch_paraphrase():
    """Handle batch paraphrase processing."""
    if FLAGS.batch_input:
        # Read from file
        input_path = Path(FLAGS.batch_input)
        if not input_path.exists():
            raise ValidationError("batch_input", str(input_path), "File not found")

        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif FLAGS.decode_input:
        # Single input for batch processing
        texts = [FLAGS.decode_input.strip()]
    else:
        raise ValidationError("batch_input", None, "Required for batch mode (file path or use decode_input)")

    if not texts:
        raise ValidationError("batch_input", "empty", "No valid texts found")

    print(f"Processing {len(texts)} texts in batch mode...")
    results = paraphrase_batch(texts)

    # Save results if output specified
    if FLAGS.batch_output:
        output_path = Path(FLAGS.batch_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for original, result in zip(texts, results):
                f.write(f"Original: {original}\n")
                f.write(f"Paraphrase: {result or 'FAILED'}\n")
                f.write("-" * 40 + "\n")

        print(f"Results saved to: {output_path}")

    # Print summary
    successful = sum(1 for r in results if r is not None)
    print(f"Batch processing complete: {successful}/{len(texts)} successful")

def handle_evaluation():
    """Handle paraphrase evaluation."""
    if not FLAGS.evaluate_original or not FLAGS.evaluate_paraphrase:
        raise ValidationError(
            "evaluation inputs",
            f"original='{FLAGS.evaluate_original}', paraphrase='{FLAGS.evaluate_paraphrase}'",
            "Both original and paraphrase required"
        )

    evaluation = evaluate_paraphrase(
        original=FLAGS.evaluate_original.strip(),
        paraphrase=FLAGS.evaluate_paraphrase.strip(),
        include_semantic=FLAGS.include_semantic
    )

    print("Evaluation Results:")
    print(f"Original: {FLAGS.evaluate_original}")
    print(f"Paraphrase: {FLAGS.evaluate_paraphrase}")
    print("-" * 40)
    for key, value in evaluation.items():
        if isinstance(value, float):
            print(".3f")
        else:
            print(f"{key}: {value}")

def handle_api_server():
    """Start the API server."""
    import uvicorn
    from src.api import app

    print(f"Starting API server on port {FLAGS.port}")
    print(f"API documentation: http://localhost:{FLAGS.port}/docs")
    print(f"Health check: http://localhost:{FLAGS.port}/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=FLAGS.port,
        reload=config.API_DEBUG,
        log_level=FLAGS.verbose and "debug" or "info"
    )

def handle_interactive():
    """Start interactive mode."""
    print("Paraphrase Generation Interactive Mode")
    print("Commands: 'quit' or 'exit' to quit, 'help' for help")
    print("-" * 50)

    while True:
        try:
            text = input("Enter text to paraphrase (or 'quit'): ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                break
            elif text.lower() == 'help':
                print("Commands:")
                print("  help  - Show this help")
                print("  quit  - Exit interactive mode")
                print("  <text> - Paraphrase the text")
                continue
            elif not text:
                continue

            paraphrase = generate_paraphrase(text, provider=FLAGS.provider, model=FLAGS.model)

            print(f"Original:  {text}")
            print(f"Paraphrase: {paraphrase}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main(argv):
    del argv  # Unused.

    # Setup system
    if FLAGS.setup_only:
        setup_and_validate()
        return

    setup_and_validate()

    mode = FLAGS.mode
    task = FLAGS.task

    # Handle legacy task-based execution
    if task in ["train_custom", "decode_custom"] and mode == "cli":
        if task == "train_custom":
            # Just setup and validate - already done above
            return
        elif task == "decode_custom":
            handle_single_paraphrase()
            return

    # Handle new mode-based execution
    try:
        if mode == "cli":
            if task == "decode_custom":
                handle_single_paraphrase()
            else:
                print(f"Unknown legacy task: {task}")

        elif mode == "batch":
            handle_batch_paraphrase()

        elif mode == "evaluate":
            handle_evaluation()

        elif mode == "api":
            handle_api_server()

        elif mode == "interactive":
            handle_interactive()

        else:
            print(f"Unknown mode: {mode}")

    except ValidationError as e:
        print(f"Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error("Main execution failed", error=str(e), exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app.run(main)

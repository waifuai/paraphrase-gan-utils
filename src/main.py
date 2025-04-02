# src/main.py
import random
from pathlib import Path
import numpy as np
import torch  # Assuming torch backend for HF

from absl import app
from absl import flags
from absl import logging

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    set_seed,
)
import datasets  # Import datasets library

# Import refactored modules
import config
from data_processing import custom_dataset
import training_utils

FLAGS = flags.FLAGS

# --- Flag Definitions ---

flags.DEFINE_enum(
    "task",
    None,
    ["train_custom", "decode_custom"],
    "The specific task to run: train or decode using the custom dataset.",
)
flags.mark_flag_as_required("task")

flags.DEFINE_string(
    "model_checkpoint",
    "t5-small",  # Default to a small, common seq2seq model
    "Hugging Face model checkpoint name (e.g., 't5-small', 'google/bart-base').",
)
flags.DEFINE_string(
    "output_dir",
    str(config.OUTPUT_DIR),
    "Base directory to save checkpoints and logs. A subdirectory based on the task/model will be created.",
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

# --- Overrides for Config/Training Values ---
flags.DEFINE_integer(
    "max_source_length",
    config.DEFAULT_MAX_LEN,
    "Maximum sequence length for source inputs.",
)
flags.DEFINE_integer(
    "max_target_length",
    config.DEFAULT_MAX_LEN,
    "Maximum sequence length for target outputs.",
)
flags.DEFINE_integer(
    "seed", config.SEED, "Random seed for reproducibility."
)

# Training arguments overrides
flags.DEFINE_float("learning_rate", None, "Override learning rate for Trainer.")
flags.DEFINE_integer(
    "per_device_train_batch_size", None, "Override train batch size per device."
)
flags.DEFINE_integer(
    "per_device_eval_batch_size", None, "Override eval batch size per device."
)
flags.DEFINE_integer("num_train_epochs", None, "Override number of training epochs.")
flags.DEFINE_float("weight_decay", None, "Override weight decay.")
flags.DEFINE_integer("logging_steps", None, "Override logging frequency.")

# Decode specific flags
flags.DEFINE_string("decode_input", None, "Input sentence for the decode_custom task.")
flags.DEFINE_string(
    "decode_checkpoint_dir",
    None,
    "Directory containing the trained model checkpoint for decoding (usually the task output_dir). Required for decode.",
)


def main(argv):
    del argv  # Unused.
    set_seed(FLAGS.seed)  # Use transformers set_seed
    logging.set_verbosity(logging.INFO)

    task = FLAGS.task
    # Create task-specific output dir based on model checkpoint name
    model_name_slug = FLAGS.model_checkpoint.split("/")[-1]  # Get model name like 't5-small'
    task_output_dir = Path(FLAGS.output_dir) / f"{task}_{model_name_slug}"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running task: {task}")
    print(f"Model checkpoint: {FLAGS.model_checkpoint}")
    print(f"Output directory: {task_output_dir}")

    # --- Task Execution ---

    if task == "train_custom":
        # --- Train Custom Task ---

        # 1. Load Tokenizer
        print(f"Loading tokenizer for {FLAGS.model_checkpoint}...")
        tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_checkpoint)

        # 2. Load and Tokenize Dataset
        print("Loading and tokenizing custom dataset...")
        train_path = Path(FLAGS.custom_train_file)
        eval_path = Path(FLAGS.custom_eval_file)
        tokenized_datasets = custom_dataset.load_and_tokenize_custom_dataset(
            tokenizer=tokenizer,
            train_path=train_path,
            eval_path=eval_path,
            max_source_length=FLAGS.max_source_length,
            max_target_length=FLAGS.max_target_length,
            # prefix="paraphrase: " # Add prefix if using T5 or similar
        )

        # 3. Load Model
        print(f"Loading model {FLAGS.model_checkpoint}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(FLAGS.model_checkpoint)

        # 4. Get Data Collator
        data_collator = training_utils.get_data_collator(tokenizer, model)

        # 5. Create Training Arguments
        # Collect overrides from flags
        training_args_overrides = {}
        if FLAGS.learning_rate is not None:
            training_args_overrides["learning_rate"] = FLAGS.learning_rate
        if FLAGS.per_device_train_batch_size is not None:
            training_args_overrides[
                "per_device_train_batch_size"
            ] = FLAGS.per_device_train_batch_size
        if FLAGS.per_device_eval_batch_size is not None:
            training_args_overrides[
                "per_device_eval_batch_size"
            ] = FLAGS.per_device_eval_batch_size
        if FLAGS.num_train_epochs is not None:
            training_args_overrides["num_train_epochs"] = FLAGS.num_train_epochs
        if FLAGS.weight_decay is not None:
            training_args_overrides["weight_decay"] = FLAGS.weight_decay
        if FLAGS.logging_steps is not None:
            training_args_overrides["logging_steps"] = FLAGS.logging_steps

        print("Creating training arguments...")
        training_args = training_utils.create_training_args(
            output_dir=str(task_output_dir),
            # Pass other defaults or overrides from config if needed
            **training_args_overrides,
        )

        # 6. Create Trainer
        # Wrap compute_metrics to pass the tokenizer
        compute_metrics_fn = (
            lambda p: training_utils.compute_metrics(p, tokenizer)
            if training_utils.metric is not None
            else None
        )

        print("Initializing Trainer...")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["eval"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )

        # 7. Train Model
        print("Starting training...")
        train_result = trainer.train()
        print("Training finished.")

        # Save training metrics and state
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model() # Save final model checkpoint

        # 8. Evaluate Model
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print("Evaluation finished.")
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)


    elif task == "decode_custom":
        # --- Decode Custom Task ---
        if not FLAGS.decode_input:
            raise ValueError("Flag --decode_input is required for task 'decode_custom'.")
        if not FLAGS.decode_checkpoint_dir:
            # Default to the output dir created by the training task if not specified
            FLAGS.decode_checkpoint_dir = str(task_output_dir).replace(task, "train_custom")
            print(f"Inferring checkpoint directory: {FLAGS.decode_checkpoint_dir}")
            # raise ValueError("Flag --decode_checkpoint_dir is required for task 'decode_custom'.")

        checkpoint_dir = Path(FLAGS.decode_checkpoint_dir)
        if not checkpoint_dir.exists() or not (checkpoint_dir / "pytorch_model.bin").exists(): # Check for common model file
             raise FileNotFoundError(
                 f"Checkpoint directory not found or doesn't contain model files: {checkpoint_dir}"
             )

        input_sentence = FLAGS.decode_input

        print(f"Loading model and tokenizer from checkpoint: {checkpoint_dir}")
        # Load model and tokenizer from the specified checkpoint directory
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)

        # Check for GPU availability (optional, but good practice)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Using device: {device}")

        print(f"Tokenizing input: '{input_sentence}'")
        # prefix = "paraphrase: " # Use the same prefix as during training if applicable
        # inputs = tokenizer(prefix + input_sentence, return_tensors="pt").to(device)
        inputs = tokenizer(input_sentence, return_tensors="pt").to(device)


        print("Generating paraphrase...")
        # Generate output using model.generate()
        outputs = model.generate(
            **inputs,
            max_length=FLAGS.max_target_length, # Use target length for generation
            num_beams=4, # Example: Use beam search
            early_stopping=True
            # Add other generation parameters as needed (temperature, top_k, etc.)
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("-" * 40)
        print(f"Input:  {input_sentence}")
        print(f"Output: {decoded_output}")
        print("-" * 40)

    else:
        print(f"Unknown task: {task}")


if __name__ == "__main__":
    app.run(main)

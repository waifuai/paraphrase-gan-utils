# src/training_utils.py
import os
from pathlib import Path
from typing import Callable, List, Optional, Any, Tuple

import jax
import trax
from trax import layers as tl
from trax import models
from trax import optimizers
from trax import data
from trax.supervised import training
from trax import lr as lr_lib

import config  # Centralized configuration

# Default Loss and Metrics (can be overridden)
DEFAULT_LOSS = tl.CrossEntropyLoss
DEFAULT_METRICS = [tl.CrossEntropyLoss(), tl.Accuracy()]
PHRASE_LOSS = tl.WeightedCategoryCrossEntropy  # Used by phrase_generator
PHRASE_METRICS = [tl.WeightedCategoryCrossEntropy(), tl.WeightedCategoryAccuracy()]


def create_tasks(
    train_stream: Callable,
    eval_stream: Callable,
    learning_rate: float,
    loss_layer: tl.Layer = DEFAULT_LOSS(),
    optimizer_cls: type = optimizers.Adam,
    lr_schedule_fn: Callable = lr_lib.warmup_and_rsqrt_decay,
    lr_warmup_steps: int = 1000,
    n_steps_per_checkpoint: int = config.DEFAULT_N_STEPS_PER_CHECKPOINT,
    eval_metrics: Optional[List[tl.Layer]] = None,
    n_eval_batches: Optional[int] = None,  # For EvalTask n_eval_batches
) -> Tuple[training.TrainTask, training.EvalTask]:
    """
    Creates Trax training and evaluation tasks.

    Args:
        train_stream: A function that returns the training data stream.
        eval_stream: A function that returns the evaluation data stream.
        learning_rate: The maximum learning rate.
        loss_layer: The loss layer instance to use (default: CrossEntropyLoss).
        optimizer_cls: The optimizer class (default: Adam).
        lr_schedule_fn: Function to create the learning rate schedule.
        lr_warmup_steps: Number of warmup steps for the LR schedule.
        n_steps_per_checkpoint: Frequency of saving checkpoints.
        eval_metrics: List of metrics for evaluation (default uses loss and accuracy).
        n_eval_batches: Number of batches to use for evaluation (optional).

    Returns:
        A tuple containing the TrainTask and EvalTask.
    """
    if eval_metrics is None:
        eval_metrics = DEFAULT_METRICS

    lr_schedule = lr_schedule_fn(
        n_warmup_steps=lr_warmup_steps, max_value=learning_rate
    )

    train_task = training.TrainTask(
        labeled_data=train_stream(),
        loss_layer=loss_layer,
        optimizer=optimizer_cls(
            learning_rate=learning_rate
        ),  # Pass LR here for Adam etc.
        lr_schedule=lr_schedule,
        n_steps_per_checkpoint=n_steps_per_checkpoint,
    )

    eval_task_args = {
        "labeled_data": eval_stream(),
        "metrics": eval_metrics,
    }
    if n_eval_batches is not None:
        eval_task_args["n_eval_batches"] = n_eval_batches

    eval_task = training.EvalTask(**eval_task_args)

    return train_task, eval_task


def create_training_loop(
    model: tl.Layer,
    train_task: training.TrainTask,
    eval_task: training.EvalTask,
    output_dir: Path,
    train_steps: int,  # Total training steps
    eval_steps: Optional[
        int
    ] = None,  # Used for legacy checkpoint_at calculation if needed
) -> training.Loop:
    """
    Creates a Trax training Loop.

    Args:
        model: The Trax model to train.
        train_task: The training task.
        eval_task: The evaluation task.
        output_dir: Directory to save checkpoints and logs.
        train_steps: Total number of steps to train.
        eval_steps: (Optional) Used for legacy checkpoint calculation. If None,
                    Loop's default checkpointing based on n_steps_per_checkpoint is used.

    Returns:
        A configured training.Loop instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    loop_args = {
        "model": model,
        "tasks": train_task,
        "eval_tasks": [eval_task],
        "output_dir": str(output_dir),
    }

    # Optional: Replicate old checkpoint_at logic if eval_steps is provided
    # Otherwise, rely on n_steps_per_checkpoint in TrainTask
    if eval_steps is not None and eval_steps > 0:
        # Ensure eval_steps is positive to avoid division by zero or infinite loops
        checkpoint_at = [train_steps // max(1, (eval_steps * 2)), train_steps]
        checkpoint_low_at = [train_steps // max(1, (eval_steps * 4))]
        loop_args["checkpoint_at"] = checkpoint_at
        loop_args["checkpoint_low_at"] = checkpoint_low_at
        print(f"Using legacy checkpoint_at logic based on eval_steps={eval_steps}")
    else:
        print(
            f"Using default checkpointing based on n_steps_per_checkpoint={train_task.n_steps_per_checkpoint}"
        )

    return training.Loop(**loop_args)


def run_training(loop: training.Loop, train_steps: int):
    """
    Runs the training loop for a specified number of steps.

    Args:
        loop: The configured training.Loop instance.
        train_steps: Total number of steps to train.
    """
    print(f"Starting training for {train_steps} steps...")
    loop.run(n_steps=train_steps)
    print("Training complete.")


# Note: This decode function is specific to models using trax.models.transformer.fast_decode
# It might need adjustments for other model types (like the manual phrase_transformer).
# It also assumes the model was created with a 'name' attribute.
def decode_sentence(
    model_class_creator: Callable,  # Function to recreate the model (e.g., create_coco_model)
    checkpoint_path: Path,
    input_sentence: str,
    vocab_path: Path,
    vocab_size: int,  # Needed to recreate model
    model_name: str,  # Needed to recreate model
    max_len: int = config.DEFAULT_MAX_LEN,
    temperature: float = 0.0,
    n_beams: int = 1,
    start_id: int = ord("\n"),  # Default EOS/SOS for char models
    eos_id: int = ord("\n"),
) -> str:
    """
    Decodes (paraphrases) an input sentence using a trained model checkpoint.
    Assumes the model uses trax.models.transformer.fast_decode.

    Args:
        model_class_creator: Function that takes (mode, vocab_size, model_name, **kwargs)
                             and returns a model instance.
        checkpoint_path: Path to the model checkpoint file (e.g., model.pkl.gz).
        input_sentence: The sentence to paraphrase.
        vocab_path: Path to the vocabulary file used for tokenization.
        vocab_size: The vocabulary size (including padding/reserved IDs).
        model_name: The name of the model architecture (passed to creator).
        max_len: Maximum sequence length for input padding and decoding.
        temperature: Sampling temperature for decoding.
        n_beams: Number of beams for beam search (1 for greedy).
        start_id: Token ID to initiate decoding.
        eos_id: Token ID that signifies end of sequence.

    Returns:
        The decoded (paraphrased) sentence as a string.

    Raises:
        FileNotFoundError: If checkpoint or vocab file not found.
        Exception: If model loading or decoding fails.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    print(f"Loading model {model_name} from {checkpoint_path} for decoding...")
    # Recreate the model in 'predict' mode (or 'eval' if predict causes issues)
    # Pass necessary args based on the specific creator function's signature
    model_predict = model_class_creator(
        mode="predict", vocab_size=vocab_size, model_name=model_name
    )

    # Initialize model state from the checkpoint
    model_predict.init_from_file(str(checkpoint_path), weights_only=True)

    print(f"Tokenizing input sentence: '{input_sentence}'")
    # Tokenize the input sentence using the provided vocabulary
    # Assuming vocab file is compatible with trax.data.tokenize
    token_gen = data.tokenize(
        iter([input_sentence]), vocab_file=str(vocab_path), n_reserved_ids=0
    )
    try:
        input_ids = next(token_gen)
    except StopIteration:
        print("Warning: Tokenizer produced no output for the input sentence.")
        return ""  # Or raise an error

    # Pad the input sequence
    padded_ids = input_ids + [0] * (max_len - len(input_ids))
    if len(padded_ids) > max_len:
        print(f"Warning: Input sentence truncated to max_len {max_len}")
        padded_ids = padded_ids[:max_len]

    # Convert to JAX array and add batch dimension
    inputs_batch = jax.numpy.array(padded_ids)[None, :]

    print(f"Decoding with temperature={temperature}, n_beams={n_beams}...")
    # Use Trax's fast decoding function
    output_ids_batch = models.transformer.fast_decode(
        model_predict,
        inputs_batch,
        start_id=start_id,
        eos_id=eos_id,
        max_len=max_len,
        temperature=temperature,
        beam_size=n_beams,  # Note: fast_decode uses beam_size parameter
    )

    # Process the output (assuming batch size 1)
    output_ids = output_ids_batch[0].tolist()

    # Remove padding tokens (ID 0) and EOS token
    # Be careful not to remove legitimate 0s if vocab uses them differently
    cleaned_ids = [idx for idx in output_ids if idx != 0 and idx != eos_id]

    print(f"Detokenizing output IDs: {cleaned_ids}")
    # Detokenize: Convert IDs back to characters/subwords
    # This part is highly dependent on the vocabulary type (char vs subword)
    # Assuming character-level for now based on original decode function
    # TODO: Make detokenization more robust (handle subwords, different vocab types)
    try:
        # Attempt char detokenization
        output_sentence = "".join([chr(c) for c in cleaned_ids])
    except ValueError:
        print("Warning: Failed to detokenize as characters. Outputting raw IDs.")
        output_sentence = " ".join(map(str, cleaned_ids))  # Fallback

    return output_sentence

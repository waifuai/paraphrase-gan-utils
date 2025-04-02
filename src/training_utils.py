# src/training_utils.py
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
)
import evaluate
from typing import Dict, Any, Optional

# Load the BLEU metric (can be changed or expanded)
# Using BLEU as a common seq2seq metric, SacreBLEU is often preferred for robustness.
# Let's use SacreBLEU.
try:
    metric = evaluate.load("sacrebleu")
except Exception as e:
    print(f"Warning: Failed to load 'sacrebleu' metric ({e}). Falling back to 'bleu'.")
    try:
        metric = evaluate.load("bleu")
    except Exception as e_bleu:
        print(f"Warning: Failed to load 'bleu' metric ({e_bleu}). Metrics will be disabled.")
        metric = None


def create_training_args(
    output_dir: str,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 10,
    predict_with_generate: bool = True,
    push_to_hub: bool = False,
    **kwargs,  # Allow passing other TrainingArguments
) -> Seq2SeqTrainingArguments:
    """
    Creates and returns Seq2SeqTrainingArguments.

    Args:
        output_dir: Directory to save checkpoints and logs.
        learning_rate: The initial learning rate.
        per_device_train_batch_size: Batch size per GPU/CPU for training.
        per_device_eval_batch_size: Batch size per GPU/CPU for evaluation.
        num_train_epochs: Total number of training epochs.
        weight_decay: Weight decay for optimization.
        evaluation_strategy: When to perform evaluation ('no', 'steps', 'epoch').
        save_strategy: When to save checkpoints ('no', 'steps', 'epoch').
        logging_steps: Log every N steps.
        predict_with_generate: Whether to use model.generate() for evaluation predictions.
        push_to_hub: Whether to push the model to the Hugging Face Hub.
        **kwargs: Additional arguments for Seq2SeqTrainingArguments.

    Returns:
        A Seq2SeqTrainingArguments instance.
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        predict_with_generate=predict_with_generate,
        push_to_hub=push_to_hub,
        **kwargs,
    )


def compute_metrics(
    eval_pred, tokenizer: PreTrainedTokenizerBase
) -> Optional[Dict[str, float]]:
    """
    Computes metrics (e.g., BLEU) for sequence-to-sequence evaluation.

    Args:
        eval_pred: A tuple containing predictions and labels.
        tokenizer: The tokenizer used for decoding.

    Returns:
        A dictionary containing the computed metrics, or None if metric loading failed.
    """
    if metric is None:
        print("Metric computation skipped as metric object failed to load.")
        return None

    predictions, labels = eval_pred
    # Decode generated summaries, replacing -100 in labels as it's used for padding
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple post-processing: remove leading/trailing whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    # SacreBLEU expects labels to be a list of lists of strings
    decoded_labels = [[label.strip()] for label in decoded_labels]

    try:
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # Extract main score (e.g., 'bleu' or 'score' depending on the metric)
        metric_key = "score" if "score" in result else "bleu"
        if metric_key in result:
            final_result = {metric_key: result[metric_key]}
        else:
            # Fallback if expected key isn't present
            final_result = result

        # Add prediction lengths (optional)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        final_result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in final_result.items()}

    except Exception as e:
        print(f"Error computing metrics: {e}")
        print("Predictions:", decoded_preds[:2]) # Print sample predictions/labels for debugging
        print("Labels:", decoded_labels[:2])
        return None


def get_data_collator(
    tokenizer: PreTrainedTokenizerBase, model
) -> DataCollatorForSeq2Seq:
    """
    Creates and returns a data collator suitable for sequence-to-sequence tasks.

    Args:
        tokenizer: The tokenizer instance.
        model: The model instance (used by the collator).

    Returns:
        A DataCollatorForSeq2Seq instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

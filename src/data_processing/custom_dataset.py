"""
Custom dataset loading and tokenization utilities.

This module provides functionality for loading custom TSV datasets and tokenizing
them for use with transformer models. It supports both training and evaluation
splits with configurable tokenization parameters.

Key Features:
- TSV dataset loading with proper column handling
- Integration with Hugging Face datasets library
- Tokenization with configurable sequence lengths
- Support for T5-style prefix instructions
- Column renaming and preprocessing
- Batch processing support
- Error handling for missing files
"""
# src/data_processing/custom_dataset.py
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Any

# Define expected column names for clarity
SOURCE_COLUMN = "text1"
TARGET_COLUMN = "text2"


def load_and_tokenize_custom_dataset(
    tokenizer: PreTrainedTokenizerBase,
    train_path: Path,
    eval_path: Path,
    max_source_length: int,
    max_target_length: int,
    prefix: str = "paraphrase: ",  # Optional prefix for T5-style models
) -> DatasetDict:
    """
    Loads the custom TSV dataset, renames columns, and tokenizes it.

    Args:
        tokenizer: The Hugging Face tokenizer instance.
        train_path: Path to the training TSV file.
        eval_path: Path to the evaluation TSV file.
        max_source_length: Maximum length for tokenized source sequences.
        max_target_length: Maximum length for tokenized target sequences.
        prefix: Optional prefix to add to source sequences (e.g., for T5).

    Returns:
        A DatasetDict containing the tokenized 'train' and 'eval' splits.
    """
    if not train_path.is_file():
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not eval_path.is_file():
        raise FileNotFoundError(f"Evaluation data file not found: {eval_path}")

    # Load datasets using datasets.load_dataset for CSV/TSV
    # Specify separator, no header, and assign column names
    raw_datasets = load_dataset(
        "csv",
        data_files={"train": str(train_path), "eval": str(eval_path)},
        delimiter="\t",
        column_names=[SOURCE_COLUMN, TARGET_COLUMN],
        header=None,
    )

    def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, List[int]]:
        """Tokenizes source and target texts."""
        inputs = [prefix + doc for doc in examples[SOURCE_COLUMN]]
        model_inputs = tokenizer(
            inputs, max_length=max_source_length, truncation=True
        )

        # Setup the tokenizer for targets
        labels = tokenizer(
            text_target=examples[TARGET_COLUMN],
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,  # Remove original text columns
    )
    print("Tokenization complete.")
    print(f"Train dataset size: {len(tokenized_datasets['train'])}")
    print(f"Eval dataset size: {len(tokenized_datasets['eval'])}")

    return tokenized_datasets
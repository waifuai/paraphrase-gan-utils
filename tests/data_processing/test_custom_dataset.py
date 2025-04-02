# tests/data_processing/test_custom_dataset.py
import pytest
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import DatasetDict

# Import the function to test
from src.data_processing.custom_dataset import load_and_tokenize_custom_dataset

# Define paths relative to the project root (where pytest is usually run)
TEST_DATA_DIR = Path("tests/data")
TEST_TRAIN_FILE = TEST_DATA_DIR / "test_train.tsv"
TEST_EVAL_FILE = TEST_DATA_DIR / "test_eval.tsv"

# Use a small, fast tokenizer for testing
TEST_TOKENIZER_NAME = "t5-small"

@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    """Fixture to load the tokenizer once per module."""
    return AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)

@pytest.fixture(scope="module")
def tokenized_datasets(tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    """Fixture to load and tokenize the test dataset."""
    # Ensure test files exist before running
    if not TEST_TRAIN_FILE.is_file() or not TEST_EVAL_FILE.is_file():
        pytest.fail(f"Test data files not found in {TEST_DATA_DIR}")

    return load_and_tokenize_custom_dataset(
        tokenizer=tokenizer,
        train_path=TEST_TRAIN_FILE,
        eval_path=TEST_EVAL_FILE,
        max_source_length=128,
        max_target_length=128,
        prefix="" # No prefix for this test, adjust if needed for specific models
    )

def test_dataset_loading_and_structure(tokenized_datasets: DatasetDict):
    """Tests if the dataset loads and has the correct structure."""
    assert isinstance(tokenized_datasets, DatasetDict)
    assert "train" in tokenized_datasets
    assert "eval" in tokenized_datasets
    assert len(tokenized_datasets["train"]) == 2 # From test_train.tsv
    assert len(tokenized_datasets["eval"]) == 1  # From test_eval.tsv

    # Check structure of the first training example
    first_train_example = tokenized_datasets["train"][0]
    assert "input_ids" in first_train_example
    assert "attention_mask" in first_train_example
    assert "labels" in first_train_example
    assert isinstance(first_train_example["input_ids"], list)
    assert isinstance(first_train_example["attention_mask"], list)
    assert isinstance(first_train_example["labels"], list)

def test_tokenization_content(tokenized_datasets: DatasetDict, tokenizer: PreTrainedTokenizerBase):
    """Tests the content of the tokenization (decoding back)."""
    # Check first training example
    train_example = tokenized_datasets["train"][0]
    decoded_input = tokenizer.decode(train_example["input_ids"], skip_special_tokens=True)
    decoded_label = tokenizer.decode(train_example["labels"], skip_special_tokens=True)

    # Original data: "Test sentence one." -> "Test paraphrase one."
    # Note: Decoding might not be exact due to tokenization nuances, but should be close.
    # T5 adds a prefix space sometimes, so strip might be needed.
    assert "Test sentence one." in decoded_input.strip()
    assert "Test paraphrase one." in decoded_label.strip()

    # Check eval example
    eval_example = tokenized_datasets["eval"][0]
    decoded_input_eval = tokenizer.decode(eval_example["input_ids"], skip_special_tokens=True)
    decoded_label_eval = tokenizer.decode(eval_example["labels"], skip_special_tokens=True)

    # Original data: "Eval sentence alpha." -> "Eval paraphrase alpha."
    assert "Eval sentence alpha." in decoded_input_eval.strip()
    assert "Eval paraphrase alpha." in decoded_label_eval.strip()

def test_file_not_found():
    """Tests that FileNotFoundError is raised for non-existent files."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_TOKENIZER_NAME)
    with pytest.raises(FileNotFoundError):
        load_and_tokenize_custom_dataset(
            tokenizer=tokenizer,
            train_path=Path("non_existent_train.tsv"),
            eval_path=TEST_EVAL_FILE,
            max_source_length=128,
            max_target_length=128,
        )
    with pytest.raises(FileNotFoundError):
        load_and_tokenize_custom_dataset(
            tokenizer=tokenizer,
            train_path=TEST_TRAIN_FILE,
            eval_path=Path("non_existent_eval.tsv"),
            max_source_length=128,
            max_target_length=128,
        )
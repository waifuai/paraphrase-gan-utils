# src/data_processing/coco.py
import os
from pathlib import Path
from trax import data
from trax.data import inputs
import config  # Import the centralized configuration


def download_and_process_coco_data(
    data_dir: Path,
    tmp_dir: Path,
    train_filename: str,
    val_filename: str,
    vocab_filename: str,
    max_len: int,
    mode: str,
):
    """
    Downloads and processes the MS COCO Captions data if not already present.

    Uses configuration for file paths and parameters.

    Args:
        data_dir: Directory to store processed data and vocab.
        tmp_dir: Temporary directory for downloads.
        train_filename: Name for the training data file.
        val_filename: Name for the validation data file.
        vocab_filename: Name for the vocabulary file.
        max_len: Maximum sequence length.
        mode: 'char' or 'word' level processing.
    """
    train_path = data_dir / train_filename
    val_path = data_dir / val_filename
    vocab_path = data_dir / vocab_filename

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not (train_path.exists() and val_path.exists() and vocab_path.exists()):
        print(f"COCO data not found in {data_dir}. Downloading and processing...")
        # Download and prepare the data using trax built-in function
        # Note: The first argument '.' might need adjustment depending on where MsCocoCaptions expects to run.
        # Assuming it downloads relative to the project root or a cache.
        inputs.MsCocoCaptions(
            str(config.PROJECT_ROOT), data_dir=str(data_dir), tmp_dir=str(tmp_dir)
        )

        if mode == "char":
            data_fn = inputs.bidi_characters
        elif mode == "word":
            data_fn = inputs.bidi_inputs
        else:
            raise ValueError(f"Unknown mode for COCO processing: {mode}")

        # Generate train/eval streams using the appropriate bidi function
        train_stream_fn = data_fn(
            data_dir=str(data_dir),
            mode="train",
            max_source_length=max_len,
            max_target_length=max_len,
        )
        eval_stream_fn = data_fn(
            data_dir=str(data_dir),
            mode="eval",
            max_source_length=max_len,
            max_target_length=max_len,
        )

        # Convert streams to lists to process for vocab and file writing
        train_data = list(train_stream_fn())
        val_data = list(eval_stream_fn())

        # Build vocabulary from training data
        print("Building vocabulary...")
        vocab = set()
        for source, target in train_data:
            vocab.update(source)
            vocab.update(target)
        vocab = sorted(list(vocab))

        # Write vocabulary file
        print(f"Writing vocabulary to {vocab_path}...")
        with vocab_path.open("w", encoding="utf-8") as f:
            for token_id in vocab:
                # For character-level, token_id is a code point; for word-level, it's also an integer ID.
                # Trax bidi functions yield integer IDs directly.
                # We need to write the *character* representation if it's char mode,
                # or handle appropriately if word mode (depends on how vocab is used later).
                # Assuming the original logic intended char for char mode.
                # If word mode uses subwords, this vocab build might be incorrect.
                # Sticking to original logic for now: write char for char mode.
                # For word mode, the integer IDs might not correspond directly to printable chars.
                # Let's assume downstream Tokenize handles the vocab file correctly based on IDs.
                # Writing the integer ID as string might be safer for word mode.
                if mode == "char":
                    # Check if token_id is a valid Unicode code point before converting
                    try:
                        f.write(chr(token_id) + "\n")
                    except ValueError:
                        print(
                            f"Warning: Skipping invalid code point {token_id} in vocab."
                        )
                else:  # Assuming word mode uses integer IDs directly in vocab file
                    f.write(str(token_id) + "\n")

        # Write train and validation files
        print(f"Writing train data to {train_path}...")
        with train_path.open("w", encoding="utf-8") as f:
            for source, target in train_data:
                f.write(
                    " ".join(map(str, source))
                    + "\t"
                    + " ".join(map(str, target))
                    + "\n"
                )

        print(f"Writing validation data to {val_path}...")
        with val_path.open("w", encoding="utf-8") as f:
            for source, target in val_data:
                f.write(
                    " ".join(map(str, source))
                    + "\t"
                    + " ".join(map(str, target))
                    + "\n"
                )
        print("COCO data processing complete.")
    else:
        print(f"Found existing COCO data in {data_dir}.")
    return


def create_coco_vocab(vocab_path: Path):
    """
    Creates a vocabulary list from a COCO vocab file (char or word IDs).

    Args:
        vocab_path: Path to the vocabulary file.

    Returns:
        A list of vocabulary items (integer IDs).
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        # Assuming vocab file contains one item (char or word ID string) per line
        # Need to handle potential errors if file format is unexpected
        try:
            # Attempt to read as chars first (original logic)
            # This might fail if word IDs are not valid char codes
            vocab_items = [ord(line.strip()) for line in f if line.strip()]
        except ValueError:
            # Fallback: try reading as integer strings (more robust for word IDs)
            vocab_path.seek(0)  # Reset file pointer
            vocab_items = [int(line.strip()) for line in f if line.strip()]

    # Append newline ID as EOS marker (consistent with original create_vocab)
    # Ensure it's not already present
    newline_id = ord("\n")
    if newline_id not in vocab_items:
        vocab_items.append(newline_id)
    return vocab_items


def create_coco_data_pipeline(
    data_path: Path, vocab_path: Path, batch_size: int, max_len: int, mode: str
):
    """
    Creates a Trax data pipeline for reading, tokenizing, and batching COCO data.

    Args:
        data_path: Path to the train or validation data file (.txt).
        vocab_path: Path to the vocabulary file.
        batch_size: Batch size for training/evaluation.
        max_len: Maximum sequence length.
        mode: 'train' or 'eval' (affects bucketing/shuffling if added).

    Returns:
        A Trax data pipeline instance.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    # Define the generator function to read pre-processed data
    def data_generator():
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    source, target = parts
                    # Data is already tokenized (integer IDs) in the file
                    source_ids = [int(c) for c in source.split()]
                    target_ids = [int(c) for c in target.split()]
                    yield (source_ids, target_ids)
                else:
                    print(
                        f"Warning: Skipping malformed line in {data_path}: {line.strip()}"
                    )

    # Define the Trax pipeline steps
    # Note: Tokenize step might be redundant if data files already contain integer IDs.
    # However, keeping it aligns with original `read_and_batch_data` which used it.
    # It might be necessary if the vocab file format requires it.
    pipeline = [
        # data.Tokenize(vocab_file=str(vocab_path), keys=[0, 1]), # Re-evaluate if needed
        data.FilterByLength(max_length=max_len, length_keys=[0, 1]),
        # BucketByLength boundaries might need adjustment based on data characteristics
        data.BucketByLength(
            boundaries=[32, 64, 128, 256],  # Example boundaries
            batch_sizes=[
                batch_size,
                batch_size,
                batch_size,
                batch_size,
                max(1, batch_size // 2),
            ],  # Ensure batch size > 0
            length_keys=[0, 1],
        ),
        data.AddLossWeights(id_to_mask=0),  # Mask padding tokens (ID 0)
    ]

    # Add shuffling for training mode
    if mode == "train":
        pipeline.insert(0, data.Shuffle())  # Shuffle before batching

    # Create the serial pipeline
    data_pipeline = data.Serial(*pipeline)

    # Return the pipeline applied to the generator
    return data_pipeline(data_generator())

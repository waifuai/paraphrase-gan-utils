# src/data_processing/parabank.py
import os
import shutil
import itertools
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Tuple, List, Optional

import trax
from trax.data import inputs

# Import from sibling modules and config
import config
from . import utils as data_utils


def generate_parabank_vocabulary(
    raw_filepath: Path,
    vocab_dir: Path,
    vocab_filename: str = config.PARABANK_VOCAB_FILE,
    vocab_size: int = config.PARABANK_VOCAB_SIZE,
    num_samples: int = config.PARABANK_VOCAB_SAMPLE_SIZE,
) -> Path:
    """
    Generates a subword vocabulary from a sample of the raw Parabank TSV file.

    Args:
        raw_filepath: Path to the raw Parabank TSV file (e.g., "parabank.tsv").
        vocab_dir: Directory where the vocabulary file will be saved.
        vocab_filename: Name of the vocabulary file.
        vocab_size: Target vocabulary size.
        num_samples: Number of lines to sample from the raw file for vocab generation.

    Returns:
        Path to the generated vocabulary file.

    Raises:
        FileNotFoundError: If the raw_filepath does not exist.
    """
    if not raw_filepath.is_file():
        raise FileNotFoundError(f"Raw Parabank file not found: {raw_filepath}")

    vocab_path = vocab_dir / vocab_filename
    vocab_dir.mkdir(parents=True, exist_ok=True)

    if vocab_path.exists():
        print(f"Vocabulary file already exists: {vocab_path}. Skipping generation.")
        return vocab_path

    print(f"Generating vocabulary from {num_samples} samples of {raw_filepath}...")
    sample_data: List[Tuple[str, str]] = []
    line_count = 0
    try:
        with raw_filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                if line_count > num_samples:
                    break
                parts = line.strip().split("\t")
                # Use permutations like phrase_generator? No, multicore.py reads pairs directly.
                # Assuming TSV has pairs, but handle lines with not exactly 2 parts.
                if len(parts) == 2:
                    sample_data.append(tuple(parts))
                # Original multicore.py script didn't use permutations, stick to that.
    except Exception as e:
        print(f"Error reading samples from {raw_filepath}: {e}")
        raise

    if not sample_data:
        raise ValueError(
            f"No valid data pairs found in the first {num_samples} lines of {raw_filepath}."
        )

    # Define a generator for Trax CreateVocabulary
    def sample_generator() -> Iterator[Tuple[str, str]]:
        yield from sample_data

    # Use Trax to create the subword vocabulary
    # Note: inputs.tokenize is used here just to feed strings to CreateVocabulary
    # It doesn't actually tokenize based on the vocab yet (since it's being created)
    vocab_task = trax.data.CreateVocabulary(
        tokenized_inputs=inputs.tokenize(
            sample_generator(), keys=(0, 1)
        ),  # Pass raw strings
        vocab_dir=str(vocab_dir),
        vocab_file=vocab_filename,
        vocab_size=vocab_size,
        # Add other options like min_count if needed
    )
    # Run the vocabulary creation task
    _ = list(vocab_task)  # Consume the generator to execute the task

    if not vocab_path.exists():
        raise RuntimeError(f"Vocabulary file {vocab_path} was not created.")

    print(f"Vocabulary generated and saved to {vocab_path}")
    return vocab_path


def _process_parabank_chunk_internal(
    chunk_filepath: Path,
    output_dir: Path,
    vocab_dir: Path,
    vocab_filename: str,
    tmp_chunk_dir: Path,
):
    """
    Internal function to process a single chunk: tokenizes and saves output.
    Designed to be called by multiprocessing.

    Args:
        chunk_filepath: Path to the input chunk file.
        output_dir: Directory to save the final tokenized chunk.
        vocab_dir: Directory containing the vocabulary file.
        vocab_filename: Name of the vocabulary file.
        tmp_chunk_dir: Temporary directory specific to this chunk process.
    """
    chunk_id = chunk_filepath.name.split(".")[-1]  # Assumes format like 'file.tsv.000'
    moved_chunk_path = tmp_chunk_dir / chunk_filepath.name
    tokenized_tmp_path = tmp_chunk_dir / f"tokenized.tsv.{chunk_id}"
    final_tokenized_path = output_dir / f"tokenized.tsv.{chunk_id}"

    try:
        # Ensure temporary directory exists
        tmp_chunk_dir.mkdir(parents=True, exist_ok=True)

        # Move chunk to temporary processing directory
        shutil.move(str(chunk_filepath), moved_chunk_path)

        # Initialize tokenizer
        # Ensure vocab path is correct
        tokenizer = trax.data.Tokenize(
            vocab_dir=str(vocab_dir), vocab_file=vocab_filename
        )

        # Process lines
        processed_lines = 0
        with moved_chunk_path.open(
            "r", encoding="utf-8"
        ) as infile, tokenized_tmp_path.open("w", encoding="utf-8") as outfile:
            for line in infile:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    source, target = parts
                    # Tokenize using the generated subword vocabulary
                    # Note: tokenizer expects an iterable, returns a generator
                    source_tokens_gen = tokenizer([source])
                    target_tokens_gen = tokenizer([target])
                    # Consume generator and join tokens with space
                    source_tokens_str = " ".join(map(str, next(source_tokens_gen)))
                    target_tokens_str = " ".join(map(str, next(target_tokens_gen)))
                    outfile.write(f"{source_tokens_str}\t{target_tokens_str}\n")
                    processed_lines += 1

        # Move final tokenized file to the main output directory
        shutil.move(str(tokenized_tmp_path), final_tokenized_path)
        # print(f"Processed chunk {chunk_id} ({processed_lines} lines) -> {final_tokenized_path}")

    except Exception as e:
        print(f"Error processing chunk {chunk_filepath.name}: {e}")
        # Optionally re-raise or handle cleanup
    finally:
        # Clean up temporary directory for this chunk
        if tmp_chunk_dir.exists():
            shutil.rmtree(tmp_chunk_dir)
        # Clean up original moved chunk file if it still exists (e.g., on error)
        if moved_chunk_path.exists():
            moved_chunk_path.unlink()


def process_parabank_multicore(
    raw_filepath: Path,
    processed_data_dir: Path,  # Where final tokenized chunks are placed
    tmp_dir: Path,  # Base temporary directory (e.g., data/tmp)
    vocab_dir: Path,
    vocab_filename: str = config.PARABANK_VOCAB_FILE,
    num_processes: int = config.NUM_PROCESSES,
):
    """
    Splits the raw Parabank file, processes chunks in parallel using multiprocessing,
    and saves tokenized chunks to the processed_data_dir.

    Args:
        raw_filepath: Path to the raw Parabank TSV file.
        processed_data_dir: Directory where final tokenized chunks (.tsv.XXX) will be saved.
        tmp_dir: Base directory for temporary chunk processing folders.
        vocab_dir: Directory containing the vocabulary file.
        vocab_filename: Name of the vocabulary file.
        num_processes: Number of CPU cores to use.
    """
    if not raw_filepath.is_file():
        raise FileNotFoundError(f"Raw Parabank file not found: {raw_filepath}")

    print(
        f"Starting parallel processing of {raw_filepath} using {num_processes} processes..."
    )
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Split the raw file into chunks in a temporary location
    split_tmp_dir = tmp_dir / "split_chunks"
    split_tmp_dir.mkdir(exist_ok=True)
    chunk_files = data_utils.split_file(
        input_filepath=raw_filepath,
        output_dir=split_tmp_dir,
        num_chunks=num_processes,
        filename_base=raw_filepath.name,
    )

    if not chunk_files:
        print("No chunk files were created. Exiting parallel processing.")
        return

    # 2. Prepare arguments for each process
    # Each process needs its own temporary directory under the main tmp_dir
    pool_args = [
        (
            chunk_file,
            processed_data_dir,  # Output dir for final tokenized files
            vocab_dir,
            vocab_filename,
            tmp_dir / f"process_{i:03d}",  # Unique temp dir per process
        )
        for i, chunk_file in enumerate(chunk_files)
    ]

    # 3. Run processing in parallel
    with Pool(processes=num_processes) as pool:
        pool.starmap(_process_parabank_chunk_internal, pool_args)

    # 4. Clean up the temporary directory where initial split chunks were stored
    if split_tmp_dir.exists():
        shutil.rmtree(split_tmp_dir)

    print("Parallel chunk processing complete.")


def create_parabank_data_pipeline(
    data_filepath: Path,  # Path to the tokenized train or eval file
    vocab_dir: Path,
    vocab_filename: str = config.PARABANK_VOCAB_FILE,
    batch_size: int = config.MULTICORE_BATCH_SIZE,
    max_len: int = config.DEFAULT_MAX_LEN,
    mode: str = "train",  # 'train' or 'eval'
):
    """
    Creates a Trax data pipeline for reading tokenized Parabank data.

    Args:
        data_filepath: Path to the tokenized data file (train.tsv or eval.tsv).
        vocab_dir: Directory containing the vocabulary file.
        vocab_filename: Name of the vocabulary file.
        batch_size: Batch size.
        max_len: Maximum sequence length.
        mode: 'train' or 'eval'.

    Returns:
        A Trax data pipeline instance.
    """
    if not data_filepath.is_file():
        raise FileNotFoundError(f"Data file not found: {data_filepath}")
    vocab_path = vocab_dir / vocab_filename
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    # Generator function to read the already tokenized data
    def data_generator():
        with data_filepath.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    source_str, target_str = parts
                    # Convert space-separated token IDs back to integers
                    try:
                        source_ids = [int(t) for t in source_str.split()]
                        target_ids = [int(t) for t in target_str.split()]
                        yield (source_ids, target_ids)
                    except ValueError:
                        print(
                            f"Warning: Skipping line with non-integer tokens in {data_filepath}: {line.strip()}"
                        )
                else:
                    print(
                        f"Warning: Skipping malformed line in {data_filepath}: {line.strip()}"
                    )

    # Define the Trax pipeline
    # No Tokenize step needed here as data is already tokenized integers
    pipeline = [
        data.FilterByLength(max_length=max_len, length_keys=[0, 1]),
        data.BucketByLength(  # Using default boundaries, adjust if needed
            boundaries=[32, 64, 128, 256],
            batch_sizes=[batch_size] * 4 + [max(1, batch_size // 2)],
            length_keys=[0, 1],
        ),
        data.AddLossWeights(id_to_mask=0),  # Mask padding
    ]

    if mode == "train":
        pipeline.insert(0, data.Shuffle())  # Shuffle training data

    data_pipeline = data.Serial(*pipeline)
    return data_pipeline(data_generator())

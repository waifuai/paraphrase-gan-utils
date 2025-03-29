# src/data_processing/utils.py
import subprocess
import shutil
from pathlib import Path
from typing import List


def count_lines(filepath: Path) -> int:
    """
    Counts the number of lines in a file using 'wc -l' if available,
    otherwise falls back to reading the file in Python.

    Args:
        filepath: Path object pointing to the file.

    Returns:
        The number of lines in the file.

    Raises:
        FileNotFoundError: If the filepath does not exist.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Use 'bash -c' to ensure wc runs in a bash environment if needed
        # Using check_output to capture the output
        command = ["bash", "-c", f"wc -l < '{filepath}'"]
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
        # wc output might have leading spaces, strip and take the first part
        return int(output.split()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(
            f"Warning: 'wc -l' failed ({e}), falling back to Python line count for {filepath}."
        )
        line_count = 0
        with filepath.open("r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1
        return line_count


def split_file(
    input_filepath: Path, output_dir: Path, num_chunks: int, filename_base: str
) -> List[Path]:
    """
    Splits a text file into approximately equal-sized smaller files (chunks).

    Args:
        input_filepath: Path to the input file to split.
        output_dir: Directory where the chunk files will be saved.
        num_chunks: The desired number of chunks.
        filename_base: The base name for the output chunk files (e.g., "train.tsv").
                       Chunks will be named like "train.tsv.000", "train.tsv.001", etc.

    Returns:
        A list of Path objects pointing to the created chunk files.

    Raises:
        FileNotFoundError: If the input_filepath does not exist.
        ValueError: If num_chunks is less than 1.
    """
    if not input_filepath.is_file():
        raise FileNotFoundError(f"Input file not found: {input_filepath}")
    if num_chunks < 1:
        raise ValueError("Number of chunks must be at least 1.")

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    num_lines = count_lines(input_filepath)
    if num_lines == 0:
        print(f"Warning: Input file {input_filepath} is empty. No chunks created.")
        return []

    lines_per_chunk = max(
        1, (num_lines + num_chunks - 1) // num_chunks
    )  # Avoid 0 lines per chunk
    chunk_num = 0
    writer = None
    created_files: List[Path] = []

    print(
        f"Splitting {input_filepath} ({num_lines} lines) into {num_chunks} chunks (~{lines_per_chunk} lines each)..."
    )

    try:
        with input_filepath.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % lines_per_chunk == 0 and chunk_num < num_chunks:
                    if writer:
                        writer.close()
                    # Format chunk number with leading zeros (adjust padding as needed)
                    chunk_filename = f"{filename_base}.{chunk_num:03d}"
                    chunk_filepath = output_dir / chunk_filename
                    created_files.append(chunk_filepath)
                    writer = chunk_filepath.open("w", encoding="utf-8")
                    chunk_num += 1
                if writer:  # Ensure writer is open before writing
                    writer.write(line)
    finally:
        if writer:
            writer.close()

    print(
        f"Finished splitting. Created {len(created_files)} chunk files in {output_dir}."
    )
    return created_files


def merge_files(
    input_files: List[Path], output_filepath: Path, delete_inputs: bool = False
):
    """
    Merges the content of multiple input files into a single output file.

    Args:
        input_files: A list of Path objects for the files to merge.
        output_filepath: Path to the output file where content will be merged.
        delete_inputs: If True, delete the input files after successful merging.
    """
    print(f"Merging {len(input_files)} files into {output_filepath}...")
    output_filepath.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure output directory exists

    with output_filepath.open("w", encoding="utf-8") as outfile:
        for input_file in input_files:
            if input_file.is_file():
                try:
                    with input_file.open("r", encoding="utf-8") as infile:
                        shutil.copyfileobj(infile, outfile)
                    if delete_inputs:
                        try:
                            input_file.unlink()
                        except OSError as e:
                            print(
                                f"Warning: Could not delete input file {input_file}: {e}"
                            )
                except FileNotFoundError:
                    print(
                        f"Warning: Input file {input_file} not found during merge. Skipping."
                    )
            else:
                print(f"Warning: Input path {input_file} is not a file. Skipping.")
    print("Merging complete.")


def split_train_eval(
    processed_filepath: Path,
    train_filepath: Path,
    eval_filepath: Path,
    eval_ratio: float = 0.1,
):
    """
    Splits a processed data file into training and evaluation sets based on a ratio.

    Args:
        processed_filepath: Path to the input file containing all processed data.
        train_filepath: Path where the training data subset will be saved.
        eval_filepath: Path where the evaluation data subset will be saved.
        eval_ratio: The approximate fraction of data to allocate to the evaluation set.
    """
    if not processed_filepath.is_file():
        raise FileNotFoundError(f"Processed file not found: {processed_filepath}")
    if eval_ratio <= 0 or eval_ratio >= 1:
        raise ValueError("Evaluation ratio must be between 0 and 1 (exclusive).")

    train_filepath.parent.mkdir(parents=True, exist_ok=True)
    eval_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine sampling rate based on evaluation ratio
    # e.g., ratio 0.1 means sample 1 out of every 10 lines for eval
    sample_rate = int(1 / eval_ratio)
    print(
        f"Splitting {processed_filepath} into train/eval sets (eval ratio ~{eval_ratio}, sample rate 1/{sample_rate})..."
    )

    line_count = 0
    train_count = 0
    eval_count = 0
    with processed_filepath.open("r", encoding="utf-8") as infile, train_filepath.open(
        "w", encoding="utf-8"
    ) as train_file, eval_filepath.open("w", encoding="utf-8") as eval_file:
        for i, line in enumerate(infile):
            line_count += 1
            # Assign every 'sample_rate'-th line to eval, others to train
            if i % sample_rate == (
                sample_rate - 1
            ):  # Use modulo result (sample_rate - 1) for 1-based selection
                eval_file.write(line)
                eval_count += 1
            else:
                train_file.write(line)
                train_count += 1

    print(
        f"Finished splitting: {train_count} training lines, {eval_count} evaluation lines (Total: {line_count})."
    )

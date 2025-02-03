import os
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import trax
from trax import layers as tl
from trax.data import inputs
from trax.supervised import training

# Constants
DATA_DIR = Path("data")
OUTPUT_DIR = Path("model")
CHUNK_DIR = DATA_DIR / "tmp"
TRAIN_STEPS = 1000
EVAL_STEPS = 10
VOCAB_FILE = "vocab.subword"
VOCAB_SIZE = 32000
BATCH_SIZE = 32

# Ensure necessary directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def transformer_model(
    vocab_size: int = 33300,
    d_model: int = 512,
    d_ff: int = 2048,
    n_heads: int = 8,
    n_encoder_layers: int = 6,
    n_decoder_layers: int = 6,
    mode: str = "train",
) -> tl.Transformer:
    """Return a Transformer model."""
    return tl.Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        mode=mode,
    )


def load_and_preprocess_data(filename: str, data_dir: Path = DATA_DIR, max_len: int = 256):
    """Load and preprocess data using Trax."""
    file_path = data_dir / filename
    if not file_path.exists():
        print(f"File {filename} not found in {data_dir}. Exiting.")
        return None

    def data_generator():
        with file_path.open("r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                yield tuple(parts)

    tokenized_data = inputs.tokenize(
        data_generator(),
        keys=(0, 1),
        vocab_dir=str(data_dir),
        vocab_file=VOCAB_FILE,
    )
    batched_data = inputs.batch(tokenized_data, batch_size=BATCH_SIZE)
    return inputs.add_loss_weights(batched_data, id_to_mask=0)


def train_model(train_data, eval_data, output_dir: Path = OUTPUT_DIR, train_steps: int = TRAIN_STEPS):
    """Train the Transformer model using Trax."""
    train_task = training.TrainTask(
        labeled_data=train_data,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.001),
        n_steps_per_checkpoint=500,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_data,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=EVAL_STEPS,
    )

    training_loop = training.Loop(
        transformer_model(),
        train_task,
        eval_tasks=[eval_task],
        output_dir=str(output_dir),
    )
    training_loop.run(n_steps=train_steps)


def count_lines(filepath: Path) -> int:
    """Count number of lines in a file using wc if available, else fallback to Python."""
    try:
        output = subprocess.check_output(["wc", "-l", str(filepath)])
        return int(output.split()[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        with filepath.open("r") as f:
            return sum(1 for _ in f)


def split_file(filename: str, num_chunks: int, data_dir: Path = DATA_DIR):
    """Split file into approximately equal-sized chunks."""
    filepath = data_dir / filename
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    num_lines = count_lines(filepath)
    lines_per_chunk = (num_lines + num_chunks - 1) // num_chunks
    chunk_num = 0
    writer = None

    with filepath.open("r") as f:
        for i, line in enumerate(f):
            if i % lines_per_chunk == 0:
                if writer:
                    writer.close()
                chunk_filename = data_dir / f"{filename}.{chunk_num:03}"
                writer = chunk_filename.open("w")
                chunk_num += 1
            writer.write(line)
        if writer:
            writer.close()


def process_chunk(chunk_filename: str, data_dir: Path = DATA_DIR):
    """
    Process a single chunk of data by tokenizing its contents.
    
    The function moves the chunk file to a temporary directory, tokenizes each line,
    and moves the tokenized output back to the data directory with a unique name.
    """
    chunk_file_path = data_dir / chunk_filename
    # Extract chunk identifier (the part after the last dot)
    chunk_id = chunk_file_path.suffix.lstrip(".")
    tmp_dir = data_dir / "tmp" / chunk_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Move the chunk file to the temporary directory
    shutil.move(str(chunk_file_path), tmp_dir)
    moved_chunk_path = tmp_dir / chunk_filename

    tokenized_file_path = tmp_dir / "tokenized.tsv"
    tokenizer = trax.data.Tokenize(vocab_dir=str(data_dir), vocab_file=VOCAB_FILE)

    with moved_chunk_path.open("r") as infile, tokenized_file_path.open("w") as outfile:
        for line in infile:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            source, target = parts
            source_tokens = " ".join(tokenizer([source]))
            target_tokens = " ".join(tokenizer([target]))
            outfile.write(f"{source_tokens}\t{target_tokens}\n")

    # Move the tokenized file back to the main data directory with a unique name
    final_tokenized_path = data_dir / f"tokenized.tsv.{chunk_id}"
    shutil.move(str(tokenized_file_path), final_tokenized_path)


def merge_processed_chunks(original_filename: str, data_dir: Path = DATA_DIR):
    """
    Merge all tokenized chunk files into a single processed file.
    
    Tokenized chunk files are expected to have the pattern 'tokenized.tsv.*'.
    """
    processed_file_path = data_dir / f"processed_{original_filename}"
    tokenized_files = sorted(data_dir.glob("tokenized.tsv.*"))
    with processed_file_path.open("w") as outfile:
        for tokenized_file in tokenized_files:
            try:
                with tokenized_file.open("r") as infile:
                    shutil.copyfileobj(infile, outfile)
                tokenized_file.unlink()  # Remove the processed chunk file
            except FileNotFoundError:
                print(f"Warning: {tokenized_file} not found. Skipping.")


def split_train_eval(processed_filename: str, data_dir: Path = DATA_DIR, eval_ratio: float = 0.1):
    """Split processed data into training and evaluation sets."""
    processed_file_path = data_dir / processed_filename
    train_file_path = data_dir / "train.tsv"
    eval_file_path = data_dir / "eval.tsv"

    # Determine sampling rate based on evaluation ratio
    sample_rate = int(1 / eval_ratio) if eval_ratio > 0 else 10
    with processed_file_path.open("r") as infile, \
         train_file_path.open("w") as train_file, \
         eval_file_path.open("w") as eval_file:
        for i, line in enumerate(infile):
            if i % sample_rate == 0:
                eval_file.write(line)
            else:
                train_file.write(line)


def generate_vocabulary(input_filename: str, data_dir: Path = DATA_DIR, num_samples: int = 1000):
    """Generate vocabulary using a sample from the input file."""
    input_file_path = Path(input_filename)
    sample_data = []
    with input_file_path.open("r") as f:
        for _ in range(num_samples):
            line = next(f, None)
            if line is None:
                break
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            sample_data.append(tuple(parts))
    trax.data.CreateVocabulary(
        inputs.tokenize(iter(sample_data), keys=(0, 1)),
        vocab_dir=str(data_dir),
        vocab_file=VOCAB_FILE,
        vocab_size=VOCAB_SIZE,
    )


def process_all_chunks(original_filename: str, data_dir: Path = DATA_DIR):
    """Split file and process chunks in parallel."""
    num_processes = os.cpu_count() or 1
    split_file(original_filename, num_processes, data_dir)
    # Identify chunk files created by split_file (e.g. parabank.tsv.000, parabank.tsv.001, …)
    chunk_files = [f for f in os.listdir(data_dir) if f.startswith(f"{original_filename}.")]
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_chunk, [(cf, data_dir) for cf in chunk_files])


def main():
    # Step 1: Generate vocabulary from a sample of the original data
    generate_vocabulary("parabank.tsv", DATA_DIR)

    # Step 2: Split the original file and process chunks in parallel
    process_all_chunks("parabank.tsv", DATA_DIR)

    # Step 3: Merge the processed tokenized chunks into one file
    merge_processed_chunks("parabank.tsv", DATA_DIR)

    # Step 4: Split merged data into training and evaluation sets
    split_train_eval("processed_parabank.tsv", DATA_DIR)

    # Step 5: Load and preprocess data for training
    train_data = load_and_preprocess_data("train.tsv", DATA_DIR)
    eval_data = load_and_preprocess_data("eval.tsv", DATA_DIR)
    if train_data is None or eval_data is None:
        print("Error loading or preprocessing data. Exiting.")
        return

    # Step 6: Train the Transformer model
    train_model(train_data, eval_data, OUTPUT_DIR)


if __name__ == "__main__":
    main()

# src/main.py
import random
from pathlib import Path
import numpy as np
import jax
import itertools  # Added for phrase_generator

from absl import app
from absl import flags
from absl import logging

import trax
from trax.supervised import trainer_lib

# Import refactored modules
import config
from data_processing import coco as coco_data
from data_processing import parabank as parabank_data
from data_processing import utils as data_utils
from models import coco_transformer
from models import phrase_transformer
import training_utils

FLAGS = flags.FLAGS

# --- Flag Definitions ---

flags.DEFINE_enum(
    "task",
    None,
    ["coco_char", "coco_word", "multicore_parabank", "phrase_generator", "decode"],
    "The specific task to run: training a model or decoding.",
)
flags.mark_flag_as_required("task")

flags.DEFINE_string(
    "output_dir",
    str(config.OUTPUT_DIR),
    "Directory to save checkpoints and logs. A subdirectory based on the task will be created.",
)

# --- Overrides for Config Values ---
# Training parameters
flags.DEFINE_integer("train_steps", None, "Override total training steps.")
flags.DEFINE_integer("eval_steps", None, "Override evaluation frequency/batches.")
flags.DEFINE_integer("batch_size", None, "Override batch size.")
flags.DEFINE_float("learning_rate", None, "Override learning rate.")
flags.DEFINE_integer("n_steps_per_checkpoint", None, "Override checkpoint frequency.")
flags.DEFINE_integer(
    "max_len", config.DEFAULT_MAX_LEN, "Override maximum sequence length."
)

# Model parameters (allow overriding specific model params)
flags.DEFINE_string(
    "coco_model_name",
    config.COCO_MODEL_NAME,
    "Override COCO model name (transformer or transformer_encoder).",
)
flags.DEFINE_integer(
    "d_model", None, "Override model dimension (applies to selected task)."
)
flags.DEFINE_integer(
    "d_ff", None, "Override feed-forward dimension (applies to selected task)."
)
flags.DEFINE_integer(
    "n_heads", None, "Override number of attention heads (applies to selected task)."
)
flags.DEFINE_integer(
    "n_encoder_layers",
    None,
    "Override number of encoder layers (applies to selected task).",
)
flags.DEFINE_integer(
    "n_decoder_layers",
    None,
    "Override number of decoder layers (applies to selected task).",
)
flags.DEFINE_float("dropout", None, "Override dropout rate (applies to selected task).")

# Decode specific flags
flags.DEFINE_string("decode_input", None, "Input sentence for the decode task.")
flags.DEFINE_string(
    "decode_checkpoint_dir",
    None,
    "Directory containing model.pkl.gz for decoding (usually the task output_dir).",
)
flags.DEFINE_string(
    "decode_model_task",
    None,
    "Specify which trained model to use for decoding (e.g., coco_char, phrase_generator). Required for decode task.",
)


def set_random_seed(seed=config.SEED):
    """Sets random seeds for reproducibility."""
    print(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    # JAX random keys are handled differently, but Trax trainer_lib initializes them.
    trainer_lib.init_random_number_generators(seed)


def main(argv):
    del argv  # Unused.
    set_random_seed()
    logging.set_verbosity(logging.INFO)

    task = FLAGS.task
    # Create task-specific output dir *within* the main output dir
    task_output_dir = Path(FLAGS.output_dir) / task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running task: {task}")
    print(f"Output directory: {task_output_dir}")

    # --- Task Execution ---

    if task == "coco_char" or task == "coco_word":
        # --- COCO Task ---
        mode = "char" if task == "coco_char" else "word"
        print(f"Mode: {mode}")

        # Determine paths and params using config, with overrides
        data_dir = (
            config.get_coco_char_data_dir()
            if mode == "char"
            else config.get_coco_word_data_dir()
        )
        tmp_dir = config.CHUNK_DIR / f"coco_{mode}_tmp"  # Specific tmp dir
        train_file = (
            config.COCO_CHAR_TRAIN_FILE
            if mode == "char"
            else config.COCO_WORD_TRAIN_FILE
        )
        val_file = (
            config.COCO_CHAR_VAL_FILE if mode == "char" else config.COCO_WORD_VAL_FILE
        )
        vocab_file = (
            config.COCO_CHAR_VOCAB_FILE
            if mode == "char"
            else config.COCO_WORD_VOCAB_FILE
        )
        train_steps = FLAGS.train_steps or config.COCO_TRAIN_STEPS
        eval_steps = (
            FLAGS.eval_steps or config.COCO_EVAL_STEPS
        )  # Used for checkpoint_at and n_eval_batches
        batch_size = FLAGS.batch_size or config.COCO_BATCH_SIZE
        learning_rate = FLAGS.learning_rate or config.COCO_LEARNING_RATE
        max_len = FLAGS.max_len
        model_name = FLAGS.coco_model_name  # Specific flag for coco model type
        d_model = FLAGS.d_model or config.COCO_D_MODEL
        d_ff = FLAGS.d_ff or config.COCO_D_FF
        n_heads = FLAGS.n_heads or config.COCO_N_HEADS
        n_encoder_layers = FLAGS.n_encoder_layers or config.COCO_N_ENCODER_LAYERS
        n_decoder_layers = FLAGS.n_decoder_layers or config.COCO_N_DECODER_LAYERS
        dropout = FLAGS.dropout if FLAGS.dropout is not None else config.COCO_DROPOUT

        # 1. Prepare Data
        coco_data.download_and_process_coco_data(
            data_dir=data_dir,
            tmp_dir=tmp_dir,
            train_filename=train_file,
            val_filename=val_file,
            vocab_filename=vocab_file,
            max_len=max_len,
            mode=mode,
        )
        vocab_path = data_dir / vocab_file
        train_path = data_dir / train_file
        val_path = data_dir / val_file

        # Get vocab size (needed for model)
        # Add 1 for padding ID 0
        vocab_list = coco_data.create_coco_vocab(vocab_path)
        vocab_size = len(vocab_list) + 1  # Ensure padding ID is accounted for

        # 2. Create Data Pipelines
        train_stream_fn = lambda: coco_data.create_coco_data_pipeline(
            train_path, vocab_path, batch_size, max_len, mode="train"
        )
        eval_stream_fn = lambda: coco_data.create_coco_data_pipeline(
            val_path, vocab_path, batch_size, max_len, mode="eval"
        )

        # 3. Create Model
        model = coco_transformer.create_coco_model(
            mode="train",
            vocab_size=vocab_size,
            model_name=model_name,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
        )

        # 4. Create Tasks and Loop
        # Use default loss/metrics for COCO
        train_task, eval_task = training_utils.create_tasks(
            train_stream=train_stream_fn,
            eval_stream=eval_stream_fn,
            learning_rate=learning_rate,
            n_eval_batches=eval_steps,  # Use eval_steps as n_eval_batches
        )
        loop = training_utils.create_training_loop(
            model=model,
            train_task=train_task,
            eval_task=eval_task,
            output_dir=task_output_dir,
            train_steps=train_steps,
            eval_steps=eval_steps,  # Pass eval_steps for legacy checkpointing
        )

        # 5. Run Training
        training_utils.run_training(loop, train_steps)

    elif task == "multicore_parabank":
        # --- Multicore Parabank Task ---
        # This task involves vocab generation, parallel processing, merging, splitting, then training
        data_dir = config.get_parabank_data_dir()
        tmp_dir = config.CHUNK_DIR / "parabank_multicore_tmp"
        raw_file_path = data_dir / config.PARABANK_RAW_FILE
        vocab_path = data_dir / config.PARABANK_VOCAB_FILE
        processed_data_dir = (
            tmp_dir / "tokenized_chunks"
        )  # Dir to store intermediate tokenized files
        merged_processed_file = data_dir / config.PARABANK_PROCESSED_FILE
        train_file_path = data_dir / config.PARABANK_TRAIN_FILE
        eval_file_path = data_dir / config.PARABANK_EVAL_FILE

        train_steps = FLAGS.train_steps or config.MULTICORE_TRAIN_STEPS
        eval_steps = (
            FLAGS.eval_steps or config.MULTICORE_EVAL_STEPS
        )  # Used as n_eval_batches
        batch_size = FLAGS.batch_size or config.MULTICORE_BATCH_SIZE
        learning_rate = FLAGS.learning_rate or config.MULTICORE_LEARNING_RATE
        n_steps_per_checkpoint = (
            FLAGS.n_steps_per_checkpoint or config.MULTICORE_N_STEPS_PER_CHECKPOINT
        )
        max_len = FLAGS.max_len
        d_model = FLAGS.d_model or config.MULTICORE_D_MODEL
        d_ff = FLAGS.d_ff or config.MULTICORE_D_FF
        n_heads = FLAGS.n_heads or config.MULTICORE_N_HEADS
        n_encoder_layers = FLAGS.n_encoder_layers or config.MULTICORE_N_ENCODER_LAYERS
        n_decoder_layers = FLAGS.n_decoder_layers or config.MULTICORE_N_DECODER_LAYERS
        # Dropout not specified in original multicore.py model func, using COCO default if needed
        dropout = FLAGS.dropout if FLAGS.dropout is not None else config.COCO_DROPOUT

        # 1. Generate Vocabulary (if needed)
        parabank_data.generate_parabank_vocabulary(
            raw_filepath=raw_file_path, vocab_dir=data_dir
        )

        # 2. Process Chunks in Parallel
        parabank_data.process_parabank_multicore(
            raw_filepath=raw_file_path,
            processed_data_dir=processed_data_dir,  # Save tokenized chunks here
            tmp_dir=tmp_dir,
            vocab_dir=data_dir,
            vocab_filename=config.PARABANK_VOCAB_FILE,
            num_processes=config.NUM_PROCESSES,
        )

        # 3. Merge Processed Chunks
        tokenized_files = sorted(processed_data_dir.glob("tokenized.tsv.*"))
        if not tokenized_files:
            raise RuntimeError(
                f"No tokenized chunk files found in {processed_data_dir} after parallel processing."
            )
        data_utils.merge_files(
            input_files=tokenized_files,
            output_filepath=merged_processed_file,
            delete_inputs=True,
        )

        # 4. Split into Train/Eval
        data_utils.split_train_eval(
            processed_filepath=merged_processed_file,
            train_filepath=train_file_path,
            eval_filepath=eval_file_path,
            eval_ratio=config.PARABANK_EVAL_RATIO,
        )

        # 5. Create Data Pipelines
        # Note: Parabank pipeline reads already tokenized data
        train_stream_fn = lambda: parabank_data.create_parabank_data_pipeline(
            train_file_path,
            data_dir,
            config.PARABANK_VOCAB_FILE,
            batch_size,
            max_len,
            mode="train",
        )
        eval_stream_fn = lambda: parabank_data.create_parabank_data_pipeline(
            eval_file_path,
            data_dir,
            config.PARABANK_VOCAB_FILE,
            batch_size,
            max_len,
            mode="eval",
        )

        # 6. Create Model (Using COCO Transformer architecture as per original multicore.py)
        # Vocab size for subword is defined in config
        vocab_size = config.PARABANK_VOCAB_SIZE
        model = coco_transformer.create_coco_model(  # Reusing COCO model structure
            mode="train",
            vocab_size=vocab_size,
            model_name="transformer",  # Assuming standard transformer
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
        )

        # 7. Create Tasks and Loop
        train_task, eval_task = training_utils.create_tasks(
            train_stream=train_stream_fn,
            eval_stream=eval_stream_fn,
            learning_rate=learning_rate,
            n_steps_per_checkpoint=n_steps_per_checkpoint,
            n_eval_batches=eval_steps,  # Use eval_steps as n_eval_batches
        )
        loop = training_utils.create_training_loop(
            model=model,
            train_task=train_task,
            eval_task=eval_task,
            output_dir=task_output_dir,
            train_steps=train_steps,
            # Not passing eval_steps here to use default checkpointing
        )

        # 8. Run Training
        training_utils.run_training(loop, train_steps)

    elif task == "phrase_generator":
        # --- Phrase Generator Task ---
        # This uses a different model architecture and loss function
        data_dir = config.get_parabank_data_dir()  # Uses parabank data
        raw_file_path = (
            data_dir / config.PARABANK_RAW_FILE
        )  # Assumes parabank.tsv exists

        train_steps = FLAGS.train_steps or config.PHRASE_GENERATOR_TRAIN_STEPS
        # eval_steps = FLAGS.eval_steps or config.PHRASE_GENERATOR_EVAL_STEPS # Not directly used in loop setup
        batch_size = FLAGS.batch_size or config.PHRASE_GENERATOR_BATCH_SIZE
        learning_rate = FLAGS.learning_rate or config.PHRASE_GENERATOR_LEARNING_RATE
        n_steps_per_checkpoint = (
            FLAGS.n_steps_per_checkpoint
            or config.PHRASE_GENERATOR_N_STEPS_PER_CHECKPOINT
        )
        max_len = FLAGS.max_len  # Use general max_len flag

        d_model = FLAGS.d_model or config.PHRASE_GENERATOR_D_MODEL
        d_ff = FLAGS.d_ff or config.PHRASE_GENERATOR_D_FF
        n_heads = FLAGS.n_heads or config.PHRASE_GENERATOR_N_HEADS
        n_encoder_layers = (
            FLAGS.n_encoder_layers or config.PHRASE_GENERATOR_N_ENCODER_LAYERS
        )
        n_decoder_layers = (
            FLAGS.n_decoder_layers or config.PHRASE_GENERATOR_N_DECODER_LAYERS
        )

        # 1. Load Data (Original phrase generator loaded all pairs into memory)
        # This is potentially very memory intensive. We might need to adapt this.
        # For now, replicating original logic.
        if not raw_file_path.is_file():
            raise FileNotFoundError(
                f"Parabank file not found for Phrase Generator: {raw_file_path}"
            )

        print("Loading and creating phrase permutations (may take time/memory)...")
        all_phrase_pairs = []
        with raw_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                phrases = line.strip().split("\t")
                if len(phrases) >= 2:
                    all_phrase_pairs.extend(itertools.permutations(phrases, 2))
        print(f"Loaded {len(all_phrase_pairs)} phrase pairs.")

        if not all_phrase_pairs:
            raise ValueError("No phrase pairs generated from the data file.")

        # Split data (in memory)
        random.shuffle(all_phrase_pairs)  # Shuffle before splitting
        train_size = int(0.9 * len(all_phrase_pairs))
        train_data = all_phrase_pairs[:train_size]
        eval_data = all_phrase_pairs[train_size:]

        # 2. Create Data Generators (Original phrase generator style)
        # These yield raw strings, tokenization happens inside the model? No, model expects IDs.
        # The original phrase_generator model used vocab_size=256, implying char-level.
        # We need a char-level vocab and tokenization here.
        # Let's build a simple char vocab from the data.
        print("Building character vocabulary for Phrase Generator...")
        char_vocab = set()
        for p1, p2 in all_phrase_pairs:
            char_vocab.update(p1)
            char_vocab.update(p2)
        sorted_chars = sorted(list(char_vocab))
        # Create char to ID mapping (add 1 for padding=0)
        ctoi = {c: i + 1 for i, c in enumerate(sorted_chars)}
        itoc = {i + 1: c for i, c in enumerate(sorted_chars)}
        vocab_size = len(sorted_chars) + 1  # Plus padding

        def _tokenize_pair(pair):
            s, t = pair
            s_ids = [ctoi.get(c, 0) for c in s]  # Use 0 for unknown? Or raise error?
            t_ids = [ctoi.get(c, 0) for c in t]
            return (s_ids, t_ids)

        def phrase_data_generator(data_subset, loop=True):
            # This generator yields tokenized pairs directly
            indices = list(range(len(data_subset)))
            if loop:
                while True:
                    random.shuffle(indices)
                    for i in indices:
                        yield _tokenize_pair(data_subset[i])
            else:
                # No shuffling for eval
                for i in indices:
                    yield _tokenize_pair(data_subset[i])

        # Create Trax pipelines from the in-memory tokenized data generator
        # This avoids manual padding but might be slow if data is huge.
        def train_stream_provider():
            # Need to wrap the generator for Trax pipeline
            return phrase_data_generator(train_data, loop=True)

        def eval_stream_provider():
            return phrase_data_generator(eval_data, loop=False)  # Loop=False for eval

        # Define pipeline (bucketing might be useful here too)
        pipeline_fn = lambda generator_fn: trax.data.Serial(
            # Assuming max_len applies here too
            trax.data.FilterByLength(max_length=max_len, length_keys=[0, 1]),
            trax.data.BucketByLength(
                boundaries=[32, 64, 128, 256],
                batch_sizes=[batch_size] * 4 + [max(1, batch_size // 2)],
                length_keys=[0, 1],
            ),
            trax.data.AddLossWeights(id_to_mask=0),  # Mask padding
        )(generator_fn())

        train_pipeline = pipeline_fn(train_stream_provider)
        eval_pipeline = pipeline_fn(eval_stream_provider)

        # 3. Create Model
        model = phrase_transformer.create_phrase_model(
            input_vocab_size=vocab_size,
            output_vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
        )

        # 4. Create Tasks and Loop (Using specific Phrase Generator loss/metrics)
        train_task, eval_task = training_utils.create_tasks(
            train_stream=lambda: train_pipeline,  # Pass the pipeline instance
            eval_stream=lambda: eval_pipeline,  # Pass the pipeline instance
            learning_rate=learning_rate,
            loss_layer=training_utils.PHRASE_LOSS(),
            eval_metrics=training_utils.PHRASE_METRICS,
            n_steps_per_checkpoint=n_steps_per_checkpoint,
            # n_eval_batches not specified in original, omit for now
        )
        loop = training_utils.create_training_loop(
            model=model,
            train_task=train_task,
            eval_task=eval_task,
            output_dir=task_output_dir,
            train_steps=train_steps,
        )

        # 5. Run Training
        training_utils.run_training(loop, train_steps)

    elif task == "decode":
        # --- Decode Task ---
        if not FLAGS.decode_input:
            raise ValueError("Flag --decode_input is required for task 'decode'.")
        if not FLAGS.decode_checkpoint_dir:
            # If not provided, try to infer from decode_model_task output dir
            if FLAGS.decode_model_task:
                FLAGS.decode_checkpoint_dir = str(
                    Path(FLAGS.output_dir) / FLAGS.decode_model_task
                )
                print(f"Inferring checkpoint directory: {FLAGS.decode_checkpoint_dir}")
            else:
                raise ValueError(
                    "Flags --decode_checkpoint_dir or --decode_model_task required for task 'decode'."
                )
        if not FLAGS.decode_model_task:
            raise ValueError(
                "Flag --decode_model_task (e.g., coco_char) is required for task 'decode'."
            )

        decode_task = FLAGS.decode_model_task
        checkpoint_dir = Path(FLAGS.decode_checkpoint_dir)
        # Look for the latest checkpoint if model.pkl.gz doesn't exist
        checkpoint_file = checkpoint_dir / "model.pkl.gz"
        if not checkpoint_file.exists():
            checkpoints = sorted(checkpoint_dir.glob("model_*.pkl.gz"), reverse=True)
            if checkpoints:
                checkpoint_file = checkpoints[0]
            else:
                raise FileNotFoundError(
                    f"No checkpoint file (model.pkl.gz or model_*.pkl.gz) found in {checkpoint_dir}"
                )

        input_sentence = FLAGS.decode_input
        max_len = FLAGS.max_len  # Use general max_len flag

        print(
            f"Decoding using model from task '{decode_task}' and checkpoint '{checkpoint_file}'"
        )

        # Determine vocab path, vocab size, model creator based on the decode_model_task
        model_name = None  # Specific model name for creator function
        if decode_task == "coco_char":
            data_dir = config.get_coco_char_data_dir()
            vocab_path = data_dir / config.COCO_CHAR_VOCAB_FILE
            if not vocab_path.is_file():
                raise FileNotFoundError(f"Vocab not found: {vocab_path}")
            vocab_list = coco_data.create_coco_vocab(vocab_path)
            vocab_size = len(vocab_list) + 1
            model_creator = coco_transformer.create_coco_model
            model_name = (
                config.COCO_MODEL_NAME
            )  # Use default or load from checkpoint? Needs state saving. Assume default.
        elif decode_task == "coco_word":
            data_dir = config.get_coco_word_data_dir()
            vocab_path = data_dir / config.COCO_WORD_VOCAB_FILE
            if not vocab_path.is_file():
                raise FileNotFoundError(f"Vocab not found: {vocab_path}")
            vocab_list = coco_data.create_coco_vocab(vocab_path)
            vocab_size = len(vocab_list) + 1
            model_creator = coco_transformer.create_coco_model
            model_name = config.COCO_MODEL_NAME
        elif decode_task == "multicore_parabank":
            data_dir = config.get_parabank_data_dir()
            vocab_path = data_dir / config.PARABANK_VOCAB_FILE
            if not vocab_path.is_file():
                raise FileNotFoundError(f"Vocab not found: {vocab_path}")
            vocab_size = config.PARABANK_VOCAB_SIZE  # Subword vocab size
            model_creator = coco_transformer.create_coco_model  # It used coco model
            model_name = "transformer"  # Assuming standard transformer was used
            # Note: Subword detokenization is needed here! Current decode_sentence assumes chars.
            print(
                "WARNING: Decoding for multicore_parabank assumes character detokenization, which is likely incorrect for subwords."
            )
        elif decode_task == "phrase_generator":
            # Phrase generator used dynamic char vocab - need to reconstruct or save it.
            # This decode path is problematic without saved vocab mapping.
            # For now, raise error.
            # TODO: Implement saving/loading of phrase_generator vocab (ctoi/itoc) during training.
            raise NotImplementedError(
                "Decoding for 'phrase_generator' requires saved vocab mapping (ctoi/itoc), which is not implemented yet."
            )
            # If vocab was saved:
            # vocab_path = ... # Path to saved vocab/mapping
            # ctoi, itoc = load_phrase_vocab(vocab_path)
            # vocab_size = len(ctoi) + 1
            # model_creator = phrase_transformer.create_phrase_model
            # model_name = None # Not needed if creator doesn't use it
        else:
            raise ValueError(
                f"Unknown task specified for --decode_model_task: {decode_task}"
            )

        # Call the generic decode function
        output_sentence = training_utils.decode_sentence(
            model_class_creator=model_creator,
            checkpoint_path=checkpoint_file,
            input_sentence=input_sentence,
            vocab_path=vocab_path,
            vocab_size=vocab_size,
            model_name=model_name,  # Pass model name if creator needs it
            max_len=max_len,
            # Add other decode params like temperature, beams if needed via flags
        )

        print("-" * 40)
        print(f"Input:  {input_sentence}")
        print(f"Output: {output_sentence}")
        print("-" * 40)

    else:
        print(f"Unknown task: {task}")


if __name__ == "__main__":
    app.run(main)

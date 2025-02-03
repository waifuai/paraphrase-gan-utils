import os
from trax import data
from trax.data import inputs

def download_and_process_data(data_dir, tmp_dir, train_filename, val_filename, vocab_filename, max_len, mode):
    """
    Downloads and processes the MS COCO Captions data if not already present.
    
    For the character-level pipeline, mode should be 'char' and the data_fn will be
    inputs.bidi_characters; for the word-level pipeline, mode should be 'word' and the
    data_fn will be inputs.bidi_inputs.
    """
    train_path = os.path.join(data_dir, train_filename)
    val_path = os.path.join(data_dir, val_filename)
    vocab_path = os.path.join(data_dir, vocab_filename)
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(vocab_path)):
        # Download and prepare the data.
        inputs.MsCocoCaptions('.', data_dir=data_dir, tmp_dir=tmp_dir)
        if mode == 'char':
            data_fn = inputs.bidi_characters
        elif mode == 'word':
            data_fn = inputs.bidi_inputs
        else:
            raise ValueError(f"Unknown mode: {mode}")

        train_data = list(data_fn(data_dir=data_dir, mode='train',
                                  max_source_length=max_len, max_target_length=max_len)())
        val_data = list(data_fn(data_dir=data_dir, mode='eval',
                                max_source_length=max_len, max_target_length=max_len)())

        # Build vocabulary from training data.
        vocab = set()
        for source, target in train_data:
            vocab.update(source)
            vocab.update(target)
        vocab = sorted(list(vocab))

        # Write vocabulary file.
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in vocab:
                # For character-level, token is a code point; for word-level, the token is still represented as a code.
                f.write(chr(token) + "\n")

        # Write train and validation files.
        for fname, dataset in ((train_filename, train_data), (val_filename, val_data)):
            with open(os.path.join(data_dir, fname), "w", encoding="utf-8") as f:
                for source, target in dataset:
                    f.write(" ".join(map(str, source)) + "\t" + " ".join(map(str, target)) + "\n")
    return

def create_vocab(vocab_file):
    """Creates a vocabulary list from a vocab file, appending newline (EOS)."""
    with open(vocab_file, encoding="utf-8") as f:
        vocab = [ord(line.strip()) for line in f]
    # Append newline as EOS
    return vocab + [ord('\n')]

def read_and_batch_data(data_file, vocab_file, batch_size, max_len):
    """
    Reads, tokenizes, and batches data from a file.
    
    Uses a Trax pipeline to tokenize the text (using the provided vocab file),
    filter by length, bucket, and add loss weights.
    """
    def data_generator():
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                source, target = line.strip().split("\t")
                source_ids = [int(c) for c in source.split()]
                target_ids = [int(c) for c in target.split()]
                yield (source_ids, target_ids)
    
    pipeline = [
        data.Tokenize(vocab_file=vocab_file, keys=[0, 1]),
        data.FilterByLength(max_length=max_len, length_keys=[0, 1]),
        data.BucketByLength(
            boundaries=[32, 64, 128, 256],
            batch_sizes=[batch_size, batch_size, batch_size, batch_size, batch_size // 2],
            length_keys=[0, 1]
        ),
        data.AddLossWeights(id_to_mask=0)
    ]
    data_pipeline = data.Serial(pipeline, input_signature=None)
    return data_pipeline(data_generator())

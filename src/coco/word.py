import os
import functools
from absl import app, flags

import jax
from trax import data

from data_utils import download_and_process_data, create_vocab, read_and_batch_data
from model_utils import create_model, train_task_setup, decode

FLAGS = flags.FLAGS

# Flags for word-level model.
flags.DEFINE_string('data_dir', './word_data', 'Path to the data directory.')
flags.DEFINE_string('output_dir', './model_output', 'Path to the output directory.')
flags.DEFINE_integer('train_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('eval_steps', 10, 'Number of evaluation steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training and evaluation.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for the optimizer.')
flags.DEFINE_string('model_name', 'transformer', 'Name of the model to use (transformer or transformer_encoder).')
flags.DEFINE_integer('max_len', 256, 'Maximum sequence length for training and inference')

# File names for word-level data.
TRAIN_FILE = "train_word.txt"
VAL_FILE = "val_word.txt"
VOCAB_FILE = "vocab_word.txt"

def main(_):
    data_dir = os.path.expanduser(FLAGS.data_dir)
    tmp_dir = os.path.join(data_dir, 'tmp')

    # Create necessary directories.
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Download and process data if needed.
    download_and_process_data(data_dir, tmp_dir, TRAIN_FILE, VAL_FILE, VOCAB_FILE, FLAGS.max_len, mode='word')

    # Create vocabulary.
    vocab = create_vocab(os.path.join(data_dir, VOCAB_FILE))
    vocab_size = len(vocab) + 1  # +1 for padding

    # Define data streams.
    train_stream = functools.partial(read_and_batch_data, 
                                       data_file=os.path.join(data_dir, TRAIN_FILE),
                                       vocab_file=os.path.join(data_dir, VOCAB_FILE),
                                       batch_size=FLAGS.batch_size,
                                       max_len=FLAGS.max_len)
    eval_stream = functools.partial(read_and_batch_data, 
                                      data_file=os.path.join(data_dir, VAL_FILE),
                                      vocab_file=os.path.join(data_dir, VOCAB_FILE),
                                      batch_size=FLAGS.batch_size,
                                      max_len=FLAGS.max_len)

    # Create the model.
    model = create_model('train', vocab_size, model_name=FLAGS.model_name)

    # Set up the training loop.
    loop = train_task_setup(model, train_stream, eval_stream,
                            FLAGS.output_dir, FLAGS.train_steps, FLAGS.eval_steps, FLAGS.learning_rate)

    # Run training.
    loop.run(FLAGS.train_steps)
    print("Training complete.")

    # Decode (inference) example.
    print("Enter a sentence to paraphrase:")
    input_sentence = input()
    paraphrased = decode(model, input_sentence, vocab, FLAGS.data_dir, VOCAB_FILE, FLAGS.output_dir, max_len=FLAGS.max_len)
    print("Paraphrased sentence:", paraphrased)

if __name__ == '__main__':
    app.run(main)

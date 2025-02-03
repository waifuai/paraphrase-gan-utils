import os
import tempfile
import unittest
import functools
import jax
import numpy as np

# Import functions to test
from data_utils import (
    download_and_process_data,
    create_vocab,
    read_and_batch_data,
)
from model_utils import (
    create_model,
    train_task_setup,
    decode,
)

# ===============================
# Tests for data_utils.py
# ===============================
class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to simulate the data_dir.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.temp_dir.name
        # File paths for fake data
        self.train_file = os.path.join(self.data_dir, "train.txt")
        self.val_file = os.path.join(self.data_dir, "val.txt")
        self.vocab_file = os.path.join(self.data_dir, "vocab.txt")
        # Write a fake vocab file with two characters "A" and "B".
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            f.write("A\nB\n")
        # Write fake training and validation files.
        # The format is: source_ids <tab> target_ids.
        # Here we use numbers corresponding to ord("A") and ord("B").
        with open(self.train_file, "w", encoding="utf-8") as f:
            f.write("65 66\t66 65\n")
        with open(self.val_file, "w", encoding="utf-8") as f:
            f.write("65 66\t66 65\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_vocab(self):
        # create_vocab reads each line, converts via ord() and then appends newline as EOS.
        vocab = create_vocab(self.vocab_file)
        expected = [65, 66, ord('\n')]
        self.assertEqual(vocab, expected)

    def test_read_and_batch_data(self):
        batch_size = 1
        max_len = 10
        # read_and_batch_data returns a Trax pipeline iterator.
        data_iter = read_and_batch_data(
            data_file=self.train_file,
            vocab_file=self.vocab_file,
            batch_size=batch_size,
            max_len=max_len,
        )
        # Try to get one batch from the iterator.
        try:
            batch = next(iter(data_iter))
        except StopIteration:
            batch = None
        self.assertIsNotNone(batch)
        # Check that the batch is a tuple (e.g. containing source and target tensors)
        self.assertIsInstance(batch, tuple)
        self.assertGreaterEqual(len(batch), 1)


# ===============================
# Tests for model_utils.py
# ===============================
class TestModelUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to serve as the output directory.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        # Create a dummy vocabulary (same as in TestDataUtils).
        self.vocab = [65, 66, ord('\n')]
        # Note: your code computes vocab_size as len(vocab) + 1 (for padding).
        self.vocab_size = len(self.vocab) + 1

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_model_transformer(self):
        # Test that create_model returns a Transformer model.
        model = create_model('train', self.vocab_size, model_name='transformer')
        self.assertTrue(hasattr(model, "name"))
        # For a transformer model, we expect the name to be "Transformer"
        self.assertEqual(model.name, "Transformer")

    def test_create_model_transformer_encoder(self):
        # Test that create_model returns a TransformerEncoder model.
        model = create_model('train', self.vocab_size, model_name='transformer_encoder')
        self.assertTrue(hasattr(model, "name"))
        self.assertEqual(model.name, "TransformerEncoder")

    def test_train_task_setup(self):
        # Create dummy data streams that yield one sample.
        def dummy_stream():
            yield ([65, 66], [66, 65])
        train_stream = lambda: dummy_stream()
        eval_stream = lambda: dummy_stream()
        model = create_model('train', self.vocab_size, model_name='transformer')
        loop = train_task_setup(
            model, train_stream, eval_stream,
            self.output_dir, train_steps=10, eval_steps=5,
            learning_rate=0.001
        )
        # Check that the returned loop object has a run method.
        self.assertTrue(hasattr(loop, "run"))

    def test_decode(self):
        # To test decode (which normally loads checkpoint weights and does inference),
        # we monkey-patch parts of the implementation so that it returns a fixed output.
        # Create a dummy model class.
        class DummyModel:
            name = "Transformer"
            def init_from_file(self, path, weights_only):
                # Simulate successful weight loading.
                pass
        dummy_model = DummyModel()

        # Monkey-patch model_utils.create_model so that it returns our dummy_model.
        import model_utils
        original_create_model = model_utils.create_model
        model_utils.create_model = lambda mode, vocab_size, model_name="transformer": dummy_model

        # Monkey-patch the fast_decode function.
        from trax import models
        original_fast_decode = models.transformer.fast_decode

        def dummy_fast_decode(model, inputs, start_id, eos_id, max_len, temperature, n_beams):
            # Return a numpy array with a fixed sequence:
            # [start_id, 65, 66, eos_id, 0, 0, ...] padded to max_len.
            seq = [start_id, 65, 66, eos_id] + [0] * (max_len - 4)
            return jax.numpy.array([seq])
        models.transformer.fast_decode = dummy_fast_decode

        # Prepare a temporary directory with a fake vocab file.
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "vocab.txt")
            with open(vocab_path, "w", encoding="utf-8") as f:
                f.write("A\nB\n")
            # Call decode with an arbitrary input sentence.
            input_sentence = "Any input"
            # decode converts tokens to characters using chr(), and it strips the EOS token (newline).
            # Our dummy_fast_decode returns: [start_id, 65, 66, eos_id, 0, ...]
            # With start_id = eos_id = ord('\n'), the remaining tokens 65 and 66 become "A" and "B".
            result = decode(
                dummy_model, input_sentence, self.vocab,
                data_dir=tmpdir, vocab_filename="vocab.txt",
                output_dir=self.output_dir, max_len=10
            )
            self.assertEqual(result, "AB")

        # Restore the original functions.
        model_utils.create_model = original_create_model
        models.transformer.fast_decode = original_fast_decode


# ===============================
# Main entry point for the test suite.
# ===============================
if __name__ == '__main__':
    unittest.main()

import os
import tempfile
import unittest
import itertools
import numpy as np
import random

import trax
from trax import layers as tl
from trax.supervised import training

# Import functions from your project.
from phrase_generator.data import load_parabank_data, data_generator
from phrase_generator.model import create_transformer_model
from phrase_generator.trainer.train import create_tasks, train_model

class TestDataModule(unittest.TestCase):
    def setUp(self):
        # Create a temporary parabank.tsv file with known content.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "parabank.tsv")
        # Each line contains two phrases separated by a tab.
        with open(self.temp_file, "w", encoding="utf-8") as f:
            # For a single line, permutations of two phrases yield two pairs.
            f.write("hello\tworld\n")
            f.write("foo\tbar\n")
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_load_parabank_data(self):
        data = load_parabank_data(filepath=self.temp_file)
        # For each line with two phrases, we expect 2 permutations.
        expected_line1 = set(itertools.permutations(["hello", "world"], 2))
        expected_line2 = set(itertools.permutations(["foo", "bar"], 2))
        # Total expected pairs = 4.
        self.assertEqual(len(data), 4)
        # Check that each expected pair is in the loaded data.
        self.assertTrue(expected_line1.issubset(data))
        self.assertTrue(expected_line2.issubset(data))
    
    def test_data_generator(self):
        # Create a small dummy data list of phrase pairs.
        dummy_data = [("a", "b"), ("c", "d"), ("e", "f")]
        batch_size = 2
        gen = data_generator(dummy_data, batch_size=batch_size, loop=False)
        # Get one batch from the generator.
        inputs, targets, ones1, ones2 = next(gen)
        self.assertEqual(len(inputs), batch_size)
        self.assertEqual(len(targets), batch_size)
        # Check that ones arrays have the same shape as targets (list of strings).
        # Since np.ones_like on a list converts to an array with the same shape.
        self.assertEqual(np.array(ones1).shape, np.array(targets).shape)
        self.assertEqual(np.array(ones2).shape, np.array(targets).shape)

class TestModelModule(unittest.TestCase):
    def test_create_transformer_model(self):
        # Create a model with small dimensions for fast testing.
        model = create_transformer_model(
            input_vocab_size=100,
            output_vocab_size=100,
            d_model=16,
            d_ff=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            mode="eval"
        )
        # Check that the returned model is a Trax Serial model.
        self.assertTrue(isinstance(model, tl.Serial))
    
    def test_model_forward_pass(self):
        # Create a model.
        model = create_transformer_model(
            input_vocab_size=50,
            output_vocab_size=50,
            d_model=16,
            d_ff=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            mode="eval"
        )
        # Dummy input: an integer array representing token ids.
        # The model expects a single input that will be duplicated internally.
        # We create a batch of 2 sequences, each of length 5.
        dummy_input = np.random.randint(0, 50, size=(2, 5))
        # A forward pass; the model’s output should be of shape (batch_size, d_model)
        # after the final LogSoftmax.
        output = model(dummy_input)
        # The output shape should match (2, 16) (or the last dimension equal to output_vocab_size
        # if the model’s final dense layer dictates that, depending on implementation details).
        # In our model, after the Parallel encoder/decoder and Add, we apply LogSoftmax.
        # We check that the last dimension equals the output vocab size.
        self.assertEqual(output.shape[0], dummy_input.shape[0])
        self.assertEqual(output.shape[1], 50)

class TestTrainerModule(unittest.TestCase):
    def setUp(self):
        # Create dummy generators that yield fixed batches.
        self.batch_size = 2
        self.dummy_data = [("a", "b"), ("c", "d"), ("e", "f")]
        self.train_gen = data_generator(self.dummy_data, batch_size=self.batch_size, loop=False)
        self.eval_gen = data_generator(self.dummy_data, batch_size=self.batch_size, loop=False)
        # Create a small transformer model.
        self.model = create_transformer_model(
            input_vocab_size=20,
            output_vocab_size=20,
            d_model=16,
            d_ff=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            mode="train"
        )
    
    def test_create_tasks(self):
        learning_rate = 0.01
        train_task, eval_task = create_tasks(self.train_gen, self.eval_gen, learning_rate)
        # Check that train_task and eval_task are instances of Trax’s training task classes.
        from trax.supervised.training import TrainTask, EvalTask
        self.assertIsInstance(train_task, TrainTask)
        self.assertIsInstance(eval_task, EvalTask)
    
    def test_train_model_runs(self):
        # Create tasks.
        learning_rate = 0.01
        train_task, eval_task = create_tasks(self.train_gen, self.eval_gen, learning_rate)
        # Use a temporary directory for output.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run the training loop for a very small number of steps.
            n_steps = 1
            # The training loop should return without error.
            training_loop = train_model(self.model, train_task, eval_task, output_dir=tmpdir, n_steps=n_steps)
            # We can check that the training_loop has run at least one step.
            self.assertGreaterEqual(training_loop.step, n_steps)

if __name__ == '__main__':
    # For reproducibility in tests.
    random.seed(42)
    np.random.seed(42)
    unittest.main()

# Paraphrase Generation with Transformer Model

This project implements a paraphrase generation model using a Transformer architecture built with the Trax library. The model is trained on the ParaBank dataset to learn how to generate paraphrases of input phrases.

## Overview

The code performs the following key steps:

1. **Data Loading and Preprocessing:**
    *   Loads the ParaBank dataset from a TSV file.
    *   Creates phrase pairs using permutations of phrases within each line.
    *   Splits the data into training and evaluation sets.
    *   Defines a data generator to feed batches of phrase pairs to the model during training.

2. **Model Definition:**
    *   Creates a Transformer model using Trax's `tl.Serial`, `tl.Embedding`, `tl.Relu`, `tl.Dense`, `tl.LayerNorm`, `tl.Mean`, `tl.Parallel`, `tl.Add`, and `tl.LogSoftmax` layers.
    *   The model consists of an encoder and a decoder, each with multiple layers of self-attention and feed-forward networks.
    *   The encoder processes the input phrase, and the decoder generates the output (paraphrased) phrase.

3. **Training Setup:**
    *   Defines hyperparameters such as model dimensions, number of layers, learning rate, etc.
    *   Creates training and evaluation tasks using `training.TrainTask` and `training.EvalTask`.
    *   Uses `WeightedCategoryCrossEntropy` as the loss function and `WeightedCategoryAccuracy` as an evaluation metric.
    *   Implements a learning rate schedule using `trax.lr.warmup_and_rsqrt_decay`.

4. **Training Loop:**
    *   Defines a training loop using `training.Loop` to train the model.
    *   Runs the training loop for a specified number of steps, periodically evaluating the model on the evaluation set.
    *   Saves model checkpoints to the specified output directory.

## Files

*   **`problem.py`:** The main Python script containing the code for data loading, model creation, training setup, and the training loop.

## Requirements

*   Python 3.x
*   Trax
*   NumPy
*   JAX

## Setup

1. **Install Dependencies:**

    ```bash
    pip install trax numpy jax
    ```

2. **Download ParaBank Dataset (Optional):**

    *   If you want to use a different ParaBank dataset than the default `parabank.tsv` in the current directory, download it and update the `load_parabank_data` function accordingly.

## Usage

1. **Modify Hyperparameters (Optional):**

    *   Adjust the hyperparameters in the `hparams` dictionary within `problem.py` if desired.

2. **Run the Script:**

    ```bash
    python problem.py
    ```

    This will start the training process. Model checkpoints and logs will be saved in the `~/output_dir/` directory.

## Notes

*   The current code uses a simple character-level encoding for the input and output phrases. For improved performance, consider using a more sophisticated vocabulary and subword tokenization.
*   The model architecture and hyperparameters can be further tuned to achieve better paraphrase generation quality.
*   The training process can be time-consuming, especially with a large dataset and complex model.
*   You can monitor the training progress by observing the loss and accuracy values printed during training.

## License

This project is licensed under the MIT-0 License.

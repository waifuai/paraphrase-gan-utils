# Paraphrase Generation Project

This repository contains code for training and evaluating Transformer models on various datasets for paraphrase generation. It leverages the Trax deep learning library and offers implementations for both character-level and word-level models, as well as examples of using multiple CPU cores for parallel processing.

## Project Structure

The repository is organized into the following directories:

-   **`utils/`**: Contains utility scripts used alongside the main program.
-   **`src/`**: Contains the main source code for the project.
    -   **`coco/`**: Implements character-level (`char.py`) and word-level (`word.py`) paraphrase generation models trained on the MS COCO Captions dataset. It includes detailed instructions in its `README.md`.
    -   **`multicore/`**: Demonstrates how to use multiple CPU cores for parallel processing of the training data and includes a specialized implementation for paraphrase generation using the ParaBank dataset in the `phrase_generator` subfolder.
        -   **`phrase_generator/`**: Focuses on paraphrase generation from the ParaBank dataset, including data preprocessing, model definition, training, and evaluation.
    -   **`count_words.sh`**: A shell script to count the number of unique words in a file.
    -   **`multicore.py`**: A Python script that orchestrates data preprocessing, model training, and evaluation using multiple CPU cores.
    -   **`sh.py`**: A utility Python script to run shell commands, used for testing purposes within the project.

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file lists the necessary packages, including `trax`, `jax`, and `absl-py`.

## Usage

### MS COCO Captions Dataset (`src/coco/`)

This section focuses on training Transformer models for paraphrase generation using the MS COCO Captions dataset. It provides implementations for both character-level and word-level models.

#### Character-Level Model (`char.py`)

To train the character-level model, run:

```bash
python src/coco/char.py --data_dir="./char_data" --output_dir="./model_output" --train_steps=1000 --eval_steps=10 --batch_size=128 --learning_rate=0.0005 --model_name="transformer" --max_len=256
```

#### Word-Level Model (`word.py`)

To train the word-level model, run:

```bash
python src/coco/word.py --data_dir="./word_data" --output_dir="./model_output" --train_steps=1000 --eval_steps=10 --batch_size=128 --learning_rate=0.0005 --model_name="transformer" --max_len=256
```

Refer to the `src/coco/README.md` for a detailed explanation of the flags and the training process.

### ParaBank Dataset with Multicore Processing (`src/multicore/`)

This section demonstrates how to use multiple CPU cores for parallel processing of the ParaBank dataset during training.

#### Phrase Generator (`phrase_generator/`)

The `phrase_generator` subfolder contains code specifically designed for paraphrase generation using the ParaBank dataset.

To train the model, first preprocess the ParaBank dataset and then execute:

```bash
python src/multicore/phrase_generator/trainer/problem.py
```
Check `src/multicore/phrase_generator/README.md` for the detailed description.

#### `multicore.py`

The `multicore.py` script orchestrates the entire process of data preprocessing, model training, and evaluation using multiple CPU cores. It splits the data into chunks, processes each chunk in parallel (tokenization and vocabulary generation), and then trains a Transformer model on the combined data.

**Steps:**

1. **Data Preparation:**
    -   Ensure you have the `parabank.tsv` file in the `data` directory.
    -   The script will split the `parabank.tsv` into multiple chunks for parallel processing.

2. **Parallel Processing:**
    -   The script utilizes all available CPU cores to process each chunk in parallel.
    -   Each chunk is tokenized, and a shared vocabulary file is generated.

3. **Training:**
    -   The processed chunks are merged.
    -   The data is split into training and evaluation sets.
    -   A Transformer model is trained using Trax.

**To run the `multicore.py` script:**

```bash
python src/multicore.py
```

**Note:** The script assumes the presence of the `parabank.tsv` file in the `data` directory. Adjust the `DATA_DIR` constant if your data is located elsewhere.

## Example Inference

After training any of the models (char, word, or multicore), you can generate paraphrases using the `decode` function provided in the respective scripts. The scripts will prompt you to enter a sentence, and they will output the paraphrased version.

```
Enter a sentence to paraphrase:
a man riding a wave on top of a surfboard.
Paraphrased sentence: a man is surfing on a wave.
```

## License

This project is licensed under the MIT-0 License. See the LICENSE file for details.

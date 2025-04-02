# Paraphrase Generation with Hugging Face Transformers

This repository contains code for training and evaluating a sequence-to-sequence Transformer model for paraphrase generation using the Hugging Face `transformers` and `datasets` libraries. It uses a small custom dataset for demonstration purposes.

## Project Structure

```
.
├── data/
│   ├── custom_train.tsv      # Custom training data (tab-separated: source<TAB>target)
│   └── custom_eval.tsv       # Custom evaluation data (tab-separated: source<TAB>target)
├── src/
│   ├── config.py             # Configuration (paths, training defaults)
│   ├── data_processing/
│   │   └── custom_dataset.py # Loads and tokenizes the custom dataset
│   ├── training_utils.py     # Utilities for Trainer (Args, metrics, collator)
│   └── main.py               # Main script for training and decoding
├── requirements.txt          # Project dependencies
├── .venv/                    # Virtual environment (created by uv)
├── model_output/             # Default directory for saving models/logs
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create Virtual Environment (using uv):**
    ```bash
    # Ensure uv is installed (e.g., pip install uv)
    python -m uv venv .venv
    ```
    *Note: On some systems, you might need to ensure pip is available in the venv (`.venv/Scripts/python.exe -m ensurepip`) and install uv inside (`.venv/Scripts/python.exe -m pip install uv`) before installing requirements.*

3.  **Install Dependencies:**
    Activate the environment (e.g., `source .venv/Scripts/activate` on Git Bash/Linux, or `.venv\Scripts\activate.bat` on Windows CMD) or use the venv's python directly:
    ```bash
    .venv/Scripts/python.exe -m uv pip install -r requirements.txt
    ```
    The `requirements.txt` file lists necessary packages like `transformers`, `datasets`, `torch`, `evaluate`, etc.

## Custom Dataset Format

The custom dataset files (`data/custom_train.tsv`, `data/custom_eval.tsv`) should contain tab-separated pairs of sentences, where the first column is the source sentence and the second column is the target paraphrase. Example:

```tsv
Original sentence one.<TAB>Paraphrased sentence one.
Original sentence two.<TAB>Paraphrased sentence two.
```

## Usage

The main script `src/main.py` handles training and decoding.

### Training

To train a model (e.g., `t5-small`) on the custom dataset:

```bash
.venv/Scripts/python.exe src/main.py --task train_custom --model_checkpoint t5-small --output_dir ./model_output --num_train_epochs 5 --per_device_train_batch_size 2 --per_device_eval_batch_size 2
```

**Key Flags for Training:**

*   `--task train_custom`: Specifies the training task.
*   `--model_checkpoint`: (Required) The Hugging Face model identifier (e.g., `t5-small`, `google/bart-base`).
*   `--output_dir`: Base directory to save checkpoints and logs. A subdirectory like `train_custom_t5-small` will be created inside.
*   `--custom_train_file`: Path to the training TSV file (defaults to `data/custom_train.tsv`).
*   `--custom_eval_file`: Path to the evaluation TSV file (defaults to `data/custom_eval.tsv`).
*   `--num_train_epochs`: Number of training epochs (default: 3).
*   `--per_device_train_batch_size`: Training batch size per device (default: 8).
*   `--per_device_eval_batch_size`: Evaluation batch size per device (default: 8).
*   `--learning_rate`: Set a specific learning rate.
*   `--max_source_length`, `--max_target_length`: Max sequence lengths for tokenizer.
*   `--seed`: Set a random seed.

### Decoding (Inference)

To generate a paraphrase for a given sentence using a trained model:

```bash
.venv/Scripts/python.exe src/main.py --task decode_custom --decode_input "This is the sentence to paraphrase." --decode_checkpoint_dir ./model_output/train_custom_t5-small
```

**Key Flags for Decoding:**

*   `--task decode_custom`: Specifies the decoding task.
*   `--decode_input`: (Required) The input sentence to paraphrase.
*   `--decode_checkpoint_dir`: (Required) Path to the directory containing the saved trained model checkpoint (e.g., the output directory from the training run).
*   `--max_target_length`: Max length for the generated paraphrase.

## License

This project is licensed under the MIT-0 License. See the LICENSE file for details.

# Paraphrase Generation with Transformer Models

This repository contains code for training and evaluating Transformer models on the MS COCO Captions dataset for paraphrase generation. The code has been refactored to separate data handling and model training into reusable modules.

## Repository Structure

```
paraphrase/
├── README.md
├── requirements.txt
├── char.py         # Character-level model training and inference.
├── word.py         # Word-level model training and inference.
├── data_utils.py   # Common data download and processing functions.
└── model_utils.py  # Common model, training, and inference functions.
```

## Usage

### Character-Level Model

```bash
python char.py --data_dir="./char_data" --output_dir="./model_output" --train_steps=1000 --eval_steps=10 --batch_size=128 --learning_rate=0.0005 --model_name="transformer" --max_len=256
```

### Word-Level Model

```bash
python word.py --data_dir="./word_data" --output_dir="./model_output" --train_steps=1000 --eval_steps=10 --batch_size=128 --learning_rate=0.0005 --model_name="transformer" --max_len=256
```

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

*Note: Ensure `requirements.txt` contains dependencies such as `trax`, `jax`, and `absl-py`.*

## Data

The scripts automatically download and preprocess the MS COCO Captions dataset if it is not already available in the specified `data_dir`. The dataset is split into training and validation sets along with a vocabulary file.

## Training and Inference

After training, the scripts will prompt you to enter a sentence for paraphrasing. The trained model will generate and display the paraphrased sentence.

## License

This project is licensed under the MIT-0 License - see the LICENSE file for details.
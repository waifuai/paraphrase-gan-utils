# Paraphrase Generation with Google Gemini API

This repository contains code for generating paraphrases using the `gemini-2.5-pro-preview-05-06` model via the Google Generative AI API.

## Project Structure

```
.
├── data/
│   ├── custom_train.tsv      # Custom training data (tab-separated: source<TAB>target) - Used for structure reference
│   └── custom_eval.tsv       # Custom evaluation data (tab-separated: source<TAB>target) - Used for structure reference
├── src/
│   ├── config.py             # Configuration (paths, Gemini API key loading, model name)
│   ├── gemini_api.py         # Module for interacting with the Gemini API
│   └── main.py               # Main script for validating API and decoding
├── requirements.txt          # Project dependencies (includes google-generativeai)
├── .venv/                    # Virtual environment (created by uv)
├── model_output/             # Default directory for output (less critical now)
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Obtain Gemini API Key:**
    *   Get an API key from the Google AI Studio (https://aistudio.google.com/app/apikey).
    *   Save your API key in a file named `.api-gemini` in your home directory (`~/.api-gemini`).

3.  **Create Virtual Environment (using uv):**
    ```bash
    # Ensure uv is installed (e.g., pip install uv)
    python -m uv venv .venv
    ```
    *Note: On some systems, you might need to ensure pip is available in the venv (`.venv/Scripts/python.exe -m ensurepip`) and install uv inside (`.venv/Scripts/python.exe -m pip install uv`) before installing requirements.*

4.  **Install Dependencies:**
    Activate the environment (e.g., `source .venv/bin/activate` on Git Bash/Linux, or `.venv\Scripts\activate.bat` on Windows CMD) or use the venv's python directly:
    ```bash
    .venv/Scripts/python.exe -m uv pip install -r requirements.txt
    ```
    The `requirements.txt` file lists necessary packages like `google-generativeai`, `absl-py`, and `numpy`.

## Custom Dataset Format

The custom dataset files (`data/custom_train.tsv`, `data/custom_eval.tsv`) are included for historical context and potential future use, but are not directly used by the current Gemini API-based paraphrase generation logic. They contain tab-separated pairs of sentences, where the first column is the source sentence and the second column is the target paraphrase. Example:

```tsv
Original sentence one.<TAB>Paraphrased sentence one.
Original sentence two.<TAB>Paraphrased sentence two.
```

## Usage

The main script `src/main.py` handles validating the API connection and performing paraphrase generation.

### Validate API Connection

To validate that your Gemini API key is correctly set up and the API can be reached:

```bash
.venv/Scripts/python.exe src/main.py --task train_custom
```
*(Note: The task name `train_custom` is kept for compatibility but now performs API validation instead of model training.)*

### Generate Paraphrase (Decoding)

To generate a paraphrase for a given sentence using the Gemini API:

```bash
.venv/Scripts/python.exe src/main.py --task decode_custom --decode_input "This is the sentence to paraphrase."
```

**Key Flags:**

*   `--task decode_custom`: Specifies the paraphrase generation task.
*   `--decode_input`: (Required) The input sentence to paraphrase.

## License

This project is licensed under the MIT-0 License. See the LICENSE file for details.

# Paraphrase Generation with Google Gemini API

This repository contains code for generating paraphrases using the `gemini-2.5-pro` model via the Google GenAI SDK.

Key migration notes:
- Uses the centralized Client API: [`python.from google import genai`](src/gemini_api.py:3) and [`python.genai.Client()`](src/gemini_api.py:20)
- Auth precedence is env var first, then key file: see [`python.load_gemini_api_key()`](src/config.py:24) and client init in [`python.initialize_gemini_api()`](src/gemini_api.py:12)
- Runtime requirements now use `google-genai~=1.28`; dev/test pins moved to `requirements-dev.txt`

Key migration notes:
- Uses the centralized Client API: [`python.from google import genai`](src/gemini_api.py:3) and [`python.genai.Client()`](src/gemini_api.py:12)
- Auth precedence is env var first, then key file: see [`python.load_gemini_api_key()`](src/config.py:24) and client init in [`python.initialize_gemini_api()`](src/gemini_api.py:12)
- Runtime requirements now use `google-genai~=1.28`; dev/test pins moved to `requirements-dev.txt`

## Project Structure

```
.
├── data/
│   ├── custom_train.tsv      # Custom training data (tab-separated: source<TAB>target) - Used for structure reference
│   └── custom_eval.tsv       # Custom evaluation data (tab-separated: source<TAB>target) - Used for structure reference
├── src/
│   ├── config.py             # Configuration (paths, API key loading, model name)
│   ├── gemini_api.py         # Wrapper around Google GenAI SDK Client
│   └── main.py               # CLI tasks: validate API and paraphrase decoding
├── requirements.txt          # Runtime dependencies (flexible; includes google-genai)
├── requirements-dev.txt      # Dev/test pinned dependencies
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
    *   Auth precedence:
        1) Environment variable GEMINI_API_KEY (preferred) or GOOGLE_API_KEY
        2) Fallback single line text file at `~/.api-gemini`

3.  **Create Virtual Environment (using uv):**
    ```bash
    python -m uv venv .venv
    .venv/Scripts/python.exe -m ensurepip
    .venv/Scripts/python.exe -m pip install uv
    ```
    Note: We standardize on calling uv via the venv Python shim on Windows paths.

4.  **Install Dependencies:**
    Use the venv's python directly (recommended):
    ```bash
    .venv/Scripts/python.exe -m uv pip install -r requirements.txt
    .venv/Scripts/python.exe -m uv pip install -r requirements-dev.txt
    ```
    Runtime `requirements.txt` contains flexible pins including `google-genai~=1.28`.
    Development `requirements-dev.txt` pins test tooling for reproducibility.

## Custom Dataset Format

The custom dataset files (`data/custom_train.tsv`, `data/custom_eval.tsv`) are included for historical context and potential future use, but are not directly used by the current Gemini API-based paraphrase generation logic. They contain tab-separated pairs of sentences, where the first column is the source sentence and the second column is the target paraphrase. Example:

```tsv
Original sentence one.<TAB>Paraphrased sentence one.
Original sentence two.<TAB>Paraphrased sentence two.
```

## Usage

The main script `src/main.py` handles validating the API connection and performing paraphrase generation.

### Validate API Connection

Auth precedence:
1) Set env variable `GEMINI_API_KEY` (preferred) or `GOOGLE_API_KEY`
2) Or place a single-line key file at `~/.api-gemini` for fallback

Validate that your Gemini API key is correctly set up and the API can be reached:

Option A: Run as a package (recommended for relative imports)
```bash
.venv/Scripts/python.exe -m src.main --task train_custom
```

Option B: If your shell does not recognize the package path, inject src to PYTHONPATH
```bash
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); import main; from absl import app; import sys as _s; _s.argv=['prog','--task','train_custom']; app.run(main.main)"
```

The command will initialize the GenAI Client and perform a small test request.

### Generate Paraphrase (Decoding)

To generate a paraphrase for a given sentence using the Google GenAI SDK:

Option A:
```bash
.venv/Scripts/python.exe -m src.main --task decode_custom --decode_input "This is the sentence to paraphrase."
```

Option B:
```bash
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); import main; from absl import app; import sys as _s; _s.argv=['prog','--task','decode_custom','--decode_input','This is the sentence to paraphrase.']; app.run(main.main)"
```

**Key Flags:**

*   `--task decode_custom`: Specifies the paraphrase generation task.
*   `--decode_input`: (Required) The input sentence to paraphrase.

## License

This project is licensed under the MIT-0 License. See the LICENSE file for details.

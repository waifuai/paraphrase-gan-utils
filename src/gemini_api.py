# src/gemini_api.py
from google import genai
import os
from pathlib import Path
import config
from typing import Optional

_client: Optional[genai.Client] = None

def _read_env_api_key() -> Optional[str]:
    # Prefer GEMINI_API_KEY then GOOGLE_API_KEY
    k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return k.strip() if k else None

def initialize_gemini_api() -> genai.Client:
    """Initializes and returns a singleton Google GenAI client."""
    global _client
    if _client is not None:
        return _client

    api_key = _read_env_api_key()
    if not api_key:
        api_key = config.load_gemini_api_key()
    _client = genai.Client(api_key=api_key)
    print("Gemini GenAI client initialized.")
    return _client

def _extract_text(response) -> str:
    """Extracts the first text segment from SDK response robustly."""
    try:
        # New SDK returns a typed object; access the first candidate text
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            # candidate.content.parts iterable with .text fields
            if hasattr(cand, "content") and getattr(cand, "content"):
                parts = getattr(cand.content, "parts", None)
                if parts:
                    for p in parts:
                        txt = getattr(p, "text", None)
                        if txt:
                            return txt
        # Fallback: attempt common attributes
        if hasattr(response, "text"):
            return response.text
    except Exception:
        pass
    return ""

def generate_paraphrase(input_sentence: str) -> str:
    """
    Generates a paraphrase for the input sentence using the Google GenAI SDK.
    """
    try:
        client = initialize_gemini_api()
        prompt = f"Paraphrase the following sentence: {input_sentence}"
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=prompt,
        )
        text = _extract_text(resp).strip()
        if text:
            return text
        print("Warning: GenAI returned empty or unexpected response.")
        return "Could not generate paraphrase."
    except Exception as e:
        print(f"Error calling GenAI SDK: {e}")
        return "Error generating paraphrase."

# Helper used by facade to state current model for docs/logging
def get_current_model_name() -> str:
    return config.GEMINI_MODEL_NAME
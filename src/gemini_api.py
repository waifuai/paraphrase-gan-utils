# src/gemini_api.py
import google.generativeai as genai
import os
from pathlib import Path
import config

def initialize_gemini_api():
    """Initializes the Gemini API client."""
    api_key = config.load_gemini_api_key()
    genai.configure(api_key=api_key)
    print("Gemini API initialized.")

def generate_paraphrase(input_sentence: str) -> str:
    """
    Generates a paraphrase for the input sentence using the Gemini API.

    Args:
        input_sentence: The sentence to paraphrase.

    Returns:
        The generated paraphrase.
    """
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        prompt = f"Paraphrase the following sentence: {input_sentence}"
        response = model.generate_content(prompt)
        # Assuming the response structure contains the generated text directly
        # This might need adjustment based on actual API response format
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text
        else:
             print("Warning: Gemini API returned an empty or unexpected response.")
             return "Could not generate paraphrase."
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Error generating paraphrase."

if __name__ == "__main__":
    # Example usage (for testing the module directly)
    initialize_gemini_api()
    test_sentence = "The quick brown fox jumps over the lazy dog."
    paraphrase = generate_paraphrase(test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Paraphrase: {paraphrase}")
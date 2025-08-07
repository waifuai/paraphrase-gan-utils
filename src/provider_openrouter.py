# src/provider_openrouter.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_FILE_PATH = Path.home() / ".api-openrouter"


def _resolve_openrouter_api_key() -> Optional[str]:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        if OPENROUTER_API_KEY_FILE_PATH.is_file():
            return OPENROUTER_API_KEY_FILE_PATH.read_text(encoding="utf-8").strip() or None
    except Exception:
        pass
    return None


def post_chat_completion(model_name: str, prompt: str, timeout: int = 60) -> Optional[str]:
    api_key = _resolve_openrouter_api_key()
    if not api_key:
        return None

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        content = (choices[0].get("message", {}).get("content") or "").strip()
        return content or None
    except Exception:
        return None


def classify_with_openrouter(text: str, model_name: str, timeout: int = 60) -> Optional[str]:
    prompt = (
        "Classify the following sentence as human-written or machine-generated.\n\n"
        "Categories:\n"
        "- '1': Human-written\n"
        "- '0': Machine-generated\n\n"
        "Respond with ONLY the digit '0' or '1'.\n\n"
        f"Sentence: \"{text}\"\n\n"
        "Classification:"
    )
    content = post_chat_completion(model_name=model_name, prompt=prompt, timeout=timeout)
    if content is None:
        return None

    if content in ("0", "1"):
        return content
    contains0 = "0" in content
    contains1 = "1" in content
    if contains0 and not contains1:
        return "0"
    if contains1 and not contains0:
        return "1"
    return None


def paraphrase_with_openrouter(text: str, model_name: str, timeout: int = 60) -> Optional[str]:
    prompt = f"Paraphrase the following sentence: {text}"
    return post_chat_completion(model_name=model_name, prompt=prompt, timeout=timeout)
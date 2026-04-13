"""
Provider facade for unified AI model access.

This module provides a unified interface for accessing AI providers
via OpenRouter's chat completions API. It handles model configuration
and routes requests to the OpenRouter backend.

Key Features:
- Automatic model resolution with fallbacks
- Environment variable and file-based configuration
- Clean API for paraphrase generation
- Error handling and fallback mechanisms
"""
# src/provider_facade.py
from typing import Optional
import config
from provider_openrouter import paraphrase_with_openrouter

DEFAULT_PROVIDER = "openrouter"


def resolve_model(provider: str, explicit_model: Optional[str]) -> str:
    """
    Determine the model name given an optional explicit override.
    """
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()
    return config.resolve_openrouter_model_name()


def generate_paraphrase(text: str, provider: str = DEFAULT_PROVIDER, model: Optional[str] = None) -> str:
    """
    Generate a paraphrase using OpenRouter.
    Returns a human-readable string on success or a short error message.
    """
    model_name = resolve_model(provider, model)
    out = paraphrase_with_openrouter(text, model_name=model_name)
    return out if out else "Could not generate paraphrase."

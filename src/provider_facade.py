"""
Provider facade for unified AI model access.

This module provides a unified interface for accessing different AI providers
(Gemini, OpenRouter) with automatic model resolution and provider abstraction.
It handles provider selection, model configuration, and routing requests to
the appropriate backend.

Key Features:
- Provider-agnostic interface for paraphrase generation
- Automatic model resolution with fallbacks
- Environment variable and file-based configuration
- Provider-specific model selection logic
- Clean separation between provider implementations
- Error handling and fallback mechanisms
- Support for multiple providers with unified API
"""
# src/provider_facade.py
from typing import Optional
import config
import gemini_api
from provider_openrouter import paraphrase_with_openrouter

DEFAULT_PROVIDER = "openrouter"


def resolve_model(provider: str, explicit_model: Optional[str]) -> str:
    """
    Determine the model name given a provider and an optional explicit override.
    """
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()
    prov = (provider or DEFAULT_PROVIDER).lower()
    if prov == "openrouter":
        return config.resolve_openrouter_model_name()
    if prov == "gemini":
        return config.resolve_gemini_model_name()
    # Fallback to openrouter default resolution
    return config.resolve_openrouter_model_name()


def generate_paraphrase(text: str, provider: str = DEFAULT_PROVIDER, model: Optional[str] = None) -> str:
    """
    Provider-agnostic paraphrase function.
    Returns a human-readable string on success or a short error message.
    """
    prov = (provider or DEFAULT_PROVIDER).lower()
    model_name = resolve_model(prov, model)

    if prov == "openrouter":
        out = paraphrase_with_openrouter(text, model_name=model_name)
        return out if out else "Could not generate paraphrase."
    elif prov == "gemini":
        # gemini_api uses config.GEMINI_MODEL_NAME internally which we keep resolved at import time.
        # For simplicity, we call through to the existing implementation.
        return gemini_api.generate_paraphrase(text)
    else:
        return "Unknown provider."
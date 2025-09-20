"""
Tests for model resolution and OpenRouter integration.

This module contains comprehensive tests for model configuration resolution,
OpenRouter API integration, and provider-specific functionality. It includes
tests for environment variable resolution, file-based configuration, API
connectivity, and response parsing.

Key Features:
- Model resolution testing with environment and file fallbacks
- OpenRouter API integration tests with mocked responses
- Configuration validation and error handling
- Provider-specific API behavior testing
- Response parsing and normalization validation
- Error condition handling and edge cases
"""
# tests/test_model_resolution_and_openrouter.py
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import config
from provider_openrouter import post_chat_completion, classify_with_openrouter


def test_resolve_openrouter_model_env_overrides_file(tmp_path, monkeypatch):
    # Create fake ~/.model-openrouter file
    model_path = tmp_path / ".model-openrouter"
    model_path.write_text("file-model\n", encoding="utf-8")

    # Point config's path to our temp file
    with patch.object(config, "OPENROUTER_MODEL_FILE_PATH", model_path):
        # No env -> use file
        monkeypatch.delenv("MODEL_OPENROUTER", raising=False)
        assert config.resolve_openrouter_model_name() == "file-model"

        # Env present -> env wins
        monkeypatch.setenv("MODEL_OPENROUTER", "env-model")
        assert config.resolve_openrouter_model_name() == "env-model"


def test_resolve_gemini_model_env_overrides_file(tmp_path, monkeypatch):
    model_path = tmp_path / ".model-gemini"
    model_path.write_text("file-gemini\n", encoding="utf-8")

    with patch.object(config, "GEMINI_MODEL_FILE_PATH", model_path):
        monkeypatch.delenv("MODEL_GEMINI", raising=False)
        assert config.resolve_gemini_model_name() == "file-gemini"

        monkeypatch.setenv("MODEL_GEMINI", "env-gemini")
        assert config.resolve_gemini_model_name() == "env-gemini"


@patch("provider_openrouter.requests.post")
def test_openrouter_post_success(mock_post, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Paraphrase output"}}]}
    mock_post.return_value = mock_resp

    out = post_chat_completion("deepseek/deepseek-chat-v3-0324:free", "Paraphrase the following sentence: x")
    assert out == "Paraphrase output"
    mock_post.assert_called_once()


@patch("provider_openrouter.requests.post")
def test_openrouter_post_non_200(mock_post, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {"error": "forbidden"}
    mock_post.return_value = mock_resp

    out = post_chat_completion("deepseek/deepseek-chat-v3-0324:free", "hello")
    assert out is None


@patch("provider_openrouter.requests.post")
def test_openrouter_classify_normalization(mock_post, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "The answer is 1"}}]}
    mock_post.return_value = mock_resp

    out = classify_with_openrouter("example", "deepseek/deepseek-chat-v3-0324:free")
    assert out == "1"
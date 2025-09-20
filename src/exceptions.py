"""
Custom exception hierarchy for the paraphrase generation system.

This module defines a comprehensive exception hierarchy for handling various
error conditions in the paraphrase system, including API errors, validation
errors, rate limiting, and configuration issues. All exceptions provide
detailed error information and support conversion to dictionaries for
API responses and logging.

Key Features:
- Hierarchical exception structure with base ParaphraseError class
- Provider-specific error types (APIError, AuthenticationError, RateLimitError)
- Validation and configuration error handling
- Batch processing error management
- Error conversion utilities for API responses
- Detailed error context and metadata
- Original exception chaining support
"""
# src/exceptions.py
from typing import Optional, Dict, Any
import logging

class ParaphraseError(Exception):
    """Base exception for paraphrase generation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "GENERIC_ERROR",
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_error = original_error

    def __str__(self):
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None
        }

class APIError(ParaphraseError):
    """Exception for API-related errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=f"API_ERROR_{provider.upper()}",
            details={
                "provider": provider,
                "status_code": status_code,
                "response_body": response_body
            },
            **kwargs
        )
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body

class AuthenticationError(ParaphraseError):
    """Exception for authentication/authorization errors."""

    def __init__(self, provider: str, **kwargs):
        super().__init__(
            message=f"Authentication failed for provider: {provider}",
            error_code=f"AUTH_ERROR_{provider.upper()}",
            details={"provider": provider},
            **kwargs
        )
        self.provider = provider

class RateLimitError(ParaphraseError):
    """Exception for rate limiting errors."""

    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message=f"Rate limit exceeded for provider: {provider}",
            error_code=f"RATE_LIMIT_{provider.upper()}",
            details={
                "provider": provider,
                "retry_after": retry_after
            },
            **kwargs
        )
        self.provider = provider
        self.retry_after = retry_after

class ConfigurationError(ParaphraseError):
    """Exception for configuration-related errors."""

    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            message=f"Configuration error for key: {config_key}",
            error_code="CONFIG_ERROR",
            details={"config_key": config_key},
            **kwargs
        )
        self.config_key = config_key

class ValidationError(ParaphraseError):
    """Exception for input validation errors."""

    def __init__(self, field: str, value: Any, reason: str, **kwargs):
        super().__init__(
            message=f"Validation error for field '{field}': {reason}",
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": value,
                "reason": reason
            },
            **kwargs
        )
        self.field = field
        self.value = value
        self.reason = reason

class BatchProcessingError(ParaphraseError):
    """Exception for batch processing errors."""

    def __init__(
        self,
        failed_items: list,
        successful_items: list,
        **kwargs
    ):
        super().__init__(
            message=f"Batch processing failed for {len(failed_items)} items",
            error_code="BATCH_ERROR",
            details={
                "failed_count": len(failed_items),
                "successful_count": len(successful_items),
                "failed_items": failed_items
            },
            **kwargs
        )
        self.failed_items = failed_items
        self.successful_items = successful_items

def handle_api_error(
    error: Exception,
    provider: str,
    logger: Optional[logging.Logger] = None
) -> ParaphraseError:
    """Convert various API errors to standardized ParaphraseError instances."""

    if logger:
        logger.error(f"API error from {provider}: {error}")

    # Handle specific error types
    if hasattr(error, 'status_code'):
        if error.status_code == 401:
            return AuthenticationError(provider=provider, original_error=error)
        elif error.status_code == 429:
            retry_after = getattr(error, 'retry_after', None)
            return RateLimitError(
                provider=provider,
                retry_after=retry_after,
                original_error=error
            )
        else:
            return APIError(
                message=str(error),
                provider=provider,
                status_code=error.status_code,
                original_error=error
            )
    elif isinstance(error, ConnectionError):
        return APIError(
            message="Connection error",
            provider=provider,
            original_error=error
        )
    elif isinstance(error, TimeoutError):
        return APIError(
            message="Request timeout",
            provider=provider,
            original_error=error
        )
    else:
        return APIError(
            message=str(error),
            provider=provider,
            original_error=error
        )
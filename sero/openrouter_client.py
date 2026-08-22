"""
OpenRouter API client — drop-in replacement for OpenAI client.
Uses the OpenAI-compatible endpoint at openrouter.ai/api/v1.
"""

import os
import time
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_API_MAX_RETRIES = int(os.getenv("SERO_API_MAX_RETRIES", "5"))
DEFAULT_API_RETRY_BASE_DELAY = float(os.getenv("SERO_API_RETRY_BASE_DELAY", "8.0"))
DEFAULT_API_TIMEOUT = float(os.getenv("SERO_API_TIMEOUT", "90.0"))


class OpenRouterSampleSkipError(RuntimeError):
    """Raised when a model response should skip only the current sample."""


def _is_gemini_25_flash_model(model: str) -> bool:
    normalized = model.lower()
    return "gemini-2.5-flash" in normalized


def _is_qwen3_model(model: str) -> bool:
    normalized = model.lower()
    return "qwen3-" in normalized


def _model_request_option_candidates(model: str) -> list[dict[str, Any]]:
    if _is_qwen3_model(model):
        return [
            {"extra_body": {"enable_thinking": False}},
            {"enable_thinking": False},
        ]

    if not _is_gemini_25_flash_model(model):
        return [{}]

    thinking_budget_disabled = {
        "google": {
            "thinking_config": {
                "thinking_budget": 0,
            }
        }
    }

    return [
        {"reasoning_effort": "none"},
        {"extra_body": {"extra_body": thinking_budget_disabled}},
        {"extra_body": thinking_budget_disabled},
    ]


def _looks_like_model_option_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "reasoning_effort",
            "thinking_budget",
            "thinking_config",
            "enable_thinking",
            "extra_body",
            "unexpected keyword argument",
            "unknown parameter",
            "unknown field",
            "unrecognized field",
        )
    )


def _looks_like_content_inspection_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "data_inspection_failed",
            "inappropriate content",
            "content policy",
            "content filter",
            "moderation",
            "unsafe content",
        )
    )


class OpenRouterClient:
    """
    Thin wrapper around the OpenAI client pointed at OpenRouter.
    Handles retries and rate-limit backoff automatically.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=DEFAULT_API_TIMEOUT)

    def chat(
        self,
        model: str,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = DEFAULT_API_MAX_RETRIES,
        retry_delay: float = DEFAULT_API_RETRY_BASE_DELAY,
    ) -> str:
        """
        Call a model via OpenRouter. Returns the assistant message content.
        temperature=0 → deterministic (for Phase A inference).
        """
        option_candidates = _model_request_option_candidates(model)

        for candidate_index, request_options in enumerate(option_candidates, start=1):
            for attempt in range(max_retries):
                try:
                    request_kwargs = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    request_kwargs.update(request_options)
                    response = self._client.chat.completions.create(**request_kwargs)
                    choice = response.choices[0]
                    if choice.finish_reason == "length":
                        logger.warning("Response truncated (finish_reason=length) for model=%s", model)
                    return choice.message.content or ""
                except Exception as e:
                    should_try_next_candidate = (
                        candidate_index < len(option_candidates)
                        and _looks_like_model_option_error(e)
                    )
                    if should_try_next_candidate:
                        logger.warning(
                            "Model-specific request options rejected for model=%s; trying fallback %d/%d: %s",
                            model,
                            candidate_index + 1,
                            len(option_candidates),
                            e,
                        )
                        break
                    if attempt == max_retries - 1:
                        if _looks_like_content_inspection_error(e):
                            logger.error(
                                "OpenRouter content inspection blocked current sample after %d retries: %s",
                                max_retries,
                                e,
                            )
                            raise OpenRouterSampleSkipError(str(e)) from e
                        logger.error("OpenRouter call failed after %d retries: %s", max_retries, e)
                        raise
                    wait = retry_delay * (2 ** attempt)
                    logger.warning("OpenRouter error (attempt %d/%d), retrying in %.1fs: %s",
                                   attempt + 1, max_retries, wait, e)
                    time.sleep(wait)
        return ""  # unreachable

    def system_user(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Convenience wrapper: system + user message."""
        return self.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

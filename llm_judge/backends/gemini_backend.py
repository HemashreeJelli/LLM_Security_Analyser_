"""
GeminiBackend — calls Google's Gemini API via its OpenAI-compatible endpoint.

Requires:  pip install openai   (no separate google SDK needed)

Get your API key at: https://aistudio.google.com/apikey

Usage:
    from llm_judge.backends.gemini_backend import GeminiBackend
    from llm_judge import get_judge

    judge = get_judge(backend=GeminiBackend(
        api_key="AIza...",          # or set GEMINI_API_KEY env var
        model="gemini-2.0-flash",   # fast + cheap for classification
    ))
"""

from __future__ import annotations

import os


class GeminiBackend:
    """
    LLMBackend implementation for Google Gemini (OpenAI-compatible endpoint).
    Satisfies the LLMBackend Protocol — no inheritance needed.

    Args:
        api_key: Google AI Studio API key. Falls back to GEMINI_API_KEY env var.
        model:   Gemini model ID.
                 "gemini-2.0-flash"       — fast, cheap, great for classification
                 "gemini-2.0-flash-lite"  — even cheaper
                 "gemini-1.5-pro"         — more capable, higher cost
        timeout: Request timeout in seconds.
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        timeout: int = 30,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            base_url=self._BASE_URL,
            timeout=timeout,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content

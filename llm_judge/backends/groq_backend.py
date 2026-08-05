"""
GroqBackend — calls Groq's LPUs for ultra-fast Llama-3 / Mixtral inference (OpenAI-compatible).

Requires:  pip install openai

Get your free API key at: https://console.groq.com/keys

Usage:
    from llm_judge.backends.groq_backend import GroqBackend
    from llm_judge import get_judge

    judge = get_judge(backend=GroqBackend(
        api_key="gsk_...",              # or set GROQ_API_KEY env var
        model="llama-3.3-70b-versatile", # fast + highly accurate judge
    ))
"""

from __future__ import annotations

import os


class GroqBackend:
    """
    LLMBackend implementation for Groq (OpenAI-compatible endpoint).
    Satisfies the LLMBackend Protocol — no inheritance needed.

    Args:
        api_key: Groq API key. Falls back to GROQ_API_KEY env var.
        model:   Groq model ID.
                 "llama-3.3-70b-versatile" — fast & smart (recommended)
                 "llama-3.1-8b-instant"    — ultra fast (~500+ tokens/sec)
                 "mixtral-8x7b-32768"     — Mixtral model
        timeout: Request timeout in seconds.
    """

    _BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
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
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
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

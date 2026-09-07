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
            import openai
        except ImportError:
            raise ImportError(
                "The `openai` package is required to use GroqBackend. "
                "Install it with: pip install openai"
            )

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        # The OpenAI client strictly checks for OPENAI_API_KEY in the env if you don't
        # pass api_key explicitly to the client (even though we do). We patch the env temporarily.
        original_env_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "dummy_key_to_bypass_check"

        try:
            self.model = model
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self._BASE_URL,
                timeout=timeout,
            )
        finally:
            if original_env_key is not None:
                os.environ["OPENAI_API_KEY"] = original_env_key
            else:
                del os.environ["OPENAI_API_KEY"]

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content

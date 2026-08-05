"""
OpenAIBackend — calls OpenAI's chat completions API.

Requires:  pip install openai

Usage:
    from llm_judge.backends.openai_backend import OpenAIBackend
    from llm_judge import get_judge

    judge = get_judge(backend=OpenAIBackend(
        api_key="sk-...",       # or set OPENAI_API_KEY env var
        model="gpt-4o-mini",    # cheapest + fastest option
    ))
"""

from __future__ import annotations

import os


class OpenAIBackend:
    """
    LLMBackend implementation for OpenAI chat completions.
    Satisfies the LLMBackend Protocol — no inheritance needed.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model:   Model ID. Recommended: "gpt-4o-mini" (fast + cheap).
                 Others: "gpt-4o", "gpt-3.5-turbo"
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
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
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.0,    # deterministic
        )
        return response.choices[0].message.content

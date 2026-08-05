"""
LLMBackend — the Protocol every backend must satisfy.

Using typing.Protocol (structural subtyping) instead of ABC:
  - No import required by the implementor — duck typing, zero coupling.
  - Any object with a `complete(messages)` method is a valid backend.
  - Swap Ollama → Anthropic → OpenAI by passing a different backend
    instance to LLMJudge — no other changes anywhere.

Message format (OpenAI-compatible, supported by Ollama, LiteLLM, etc.):
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """
    Structural protocol for LLM backends.

    Any object implementing `complete(messages) -> str` satisfies this
    protocol — no inheritance required.

    Implementing a new backend (e.g. Anthropic):
        class AnthropicBackend:
            def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.model = model

            def complete(self, messages: list[dict[str, str]]) -> str:
                # Separate system message from user messages
                system = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                user_msgs = [m for m in messages if m["role"] != "system"]
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=system,
                    messages=user_msgs,
                )
                return response.content[0].text

    Implementing an OpenAI backend:
        class OpenAIBackend:
            def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.model = model

            def complete(self, messages: list[dict[str, str]]) -> str:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                return response.choices[0].message.content
    """

    model: str  # the model name/identifier this backend uses

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Send a list of messages to the LLM and return the text response.

        Args:
            messages: OpenAI-compatible message list, e.g.:
                [
                    {"role": "system", "content": "You are a ..."},
                    {"role": "user",   "content": "Classify this: ..."},
                ]

        Returns:
            The model's response text as a plain string.

        Raises:
            Any network/API exception — the caller (LLMJudge) handles these.
        """
        ...

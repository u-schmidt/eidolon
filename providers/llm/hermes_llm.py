"""Hermes LLM provider for Eidolon.

Wraps Hermes' existing LLM provider as an ``LLMProvider``.
The adapter passes a callable that matches Hermes' completion interface.
"""

from __future__ import annotations

from collections.abc import Callable

from core.interfaces import LLMProvider


class HermesLLM:
    """Wraps a Hermes-compatible completion function as an ``LLMProvider``.

    Parameters
    ----------
    complete_fn : Callable
        A callable with signature ``(prompt, system, max_tokens) -> str``
        that matches Hermes' LLM completion interface.
    """

    def __init__(self, complete_fn: Callable[..., str]) -> None:
        self._complete_fn = complete_fn

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 100,
    ) -> str:
        """Send a completion request via the wrapped Hermes function.

        Parameters
        ----------
        prompt : str
            The user/assistant prompt.
        system_prompt : str | None
            Optional system prompt.
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str
            The completion text.
        """
        return self._complete_fn(
            prompt,
            system=system_prompt,
            max_tokens=max_tokens,
        )

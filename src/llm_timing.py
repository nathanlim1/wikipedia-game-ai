from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Session dict for the current step; set by runner before agent.step()
current_session_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "current_session", default=None
)


class TimingLLMWrapper:
    """Wraps an LLM client and records first_llm_at on the current session."""

    def __init__(self, real_llm: Any) -> None:
        self._real = real_llm

    def _record_first_llm_if_needed(self) -> None:
        session = current_session_var.get()
        if session is not None and session.get("first_llm_at") is None:
            session["first_llm_at"] = time.time()

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        self._record_first_llm_if_needed()
        return self._real.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            timeout_s=timeout_s,
        )

    def completion(
        self,
        prompt_text: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_s: float | None = None,
    ) -> str:
        self._record_first_llm_if_needed()
        return self._real.completion(
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

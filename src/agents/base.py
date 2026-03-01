from __future__ import annotations

from typing import Any, Dict, Optional, Protocol


SessionState = Dict[str, Any]
Event = Dict[str, Any]


class GameAgent(Protocol):
    agent_id: str

    def initialize_session(self, start_title: str, target_title: str) -> SessionState:
        """Build a new session state for this agent."""

    def step(self, session: SessionState) -> Optional[Event]:
        """Advance one step and return an optional UI event payload."""

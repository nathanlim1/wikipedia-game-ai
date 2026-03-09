from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union


SessionState = Dict[str, Any]
Event = Dict[str, Any]


class GameAgent(Protocol):
    agent_id: str

    def initialize_session(self, start_title: str, target_title: str) -> SessionState:
        """Build a new session state for this agent."""

    def step(self, session: SessionState) -> Union[None, Event, List[Event]]:
        """Advance one step. May return a single event, a list of events, or None."""

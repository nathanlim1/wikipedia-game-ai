from __future__ import annotations

import secrets
from typing import Dict, Optional

from src.agents.base import SessionState


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def create(self, session: SessionState) -> str:
        session_id = secrets.token_urlsafe(12)
        self._sessions[session_id] = session
        return session_id

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def set(self, session_id: str, session: SessionState) -> None:
        self._sessions[session_id] = session

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from src.agents.default_agent import CANDIDATE_POOL
from src.agents.registry import AgentRegistry
from src.game.session_store import InMemorySessionStore


class GameRunner:
    def __init__(
        self,
        session_store: InMemorySessionStore,
        agent_registry: AgentRegistry,
        safety_max_moves: int = 800,
        safety_max_seconds: int = 7 * 60,
    ) -> None:
        self._session_store = session_store
        self._agent_registry = agent_registry
        self._safety_max_moves = safety_max_moves
        self._safety_max_seconds = safety_max_seconds

    def start(
        self,
        start_title: str,
        target_title: str,
        agent_id: Optional[str] = None,
        llm_choices: Optional[int] = None,
    ) -> Dict[str, Any]:
        agent = self._agent_registry.get(agent_id)
        session = agent.initialize_session(start_title=start_title, target_title=target_title)
        session["agent_id"] = agent.agent_id
        session.setdefault("started_at", time.time())
        if agent.agent_id == "default" and llm_choices is not None:
            n = max(1, llm_choices)
            session["llm_choices"] = n
            session["candidate_pool"] = max(n, CANDIDATE_POOL)
        session_id = self._session_store.create(session)
        return {
            "session_id": session_id,
            "resolved_start": session["resolved_start"],
            "resolved_target": session["resolved_target"],
        }

    def step(self, session_id: str) -> Tuple[Dict[str, Any], int]:
        session = self._session_store.get(session_id)
        if not session:
            return {"failure_reason": "Invalid session_id. Click Run again."}, 400

        if session["done"]:
            return self._payload(session), 200

        if (time.time() - session["started_at"]) > self._safety_max_seconds:
            session["done"] = True
            session["success"] = False
            session["failure_reason"] = f"Stopped by safety timer ({self._safety_max_seconds}s)."
            return self._payload(session), 200

        if len(session["moves"]) >= self._safety_max_moves:
            session["done"] = True
            session["success"] = False
            session["failure_reason"] = f"Stopped by safety move cap ({self._safety_max_moves})."
            return self._payload(session), 200

        try:
            agent = self._agent_registry.get(session["agent_id"])
            event = agent.step(session)
            return self._payload(session, event), 200
        except Exception as exc:
            session["done"] = True
            session["success"] = False
            session["failure_reason"] = f"Agent step failed: {exc}"
            return self._payload(session), 200

    @staticmethod
    def _payload(session: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "done": session["done"],
            "success": session["success"],
            "failure_reason": session["failure_reason"],
            "resolved_start": session["resolved_start"],
            "resolved_target": session["resolved_target"],
            "hops": len(session["moves"]),
            "chain": session["chain_builder"](session["resolved_start"], session["moves"]),
            "steps_text": session["steps_builder"](
                session["resolved_start"], session["resolved_target"], session["moves"]
            ),
            "event": event,
        }

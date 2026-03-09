from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.agents.default_agent import CANDIDATE_POOL
from src.agents.registry import AgentRegistry
from src.llm_timing import current_session_var
from src.game.session_store import InMemorySessionStore
from src.wikipedia import WikipediaClient
from src.wikipath_client import WikipathClient

log = logging.getLogger(__name__)


class GameRunner:
    def __init__(
        self,
        session_store: InMemorySessionStore,
        agent_registry: AgentRegistry,
        wiki_client: Optional[WikipediaClient] = None,
        wikipath_client: Optional[WikipathClient] = None,
        safety_max_moves: int = 800,
        safety_max_seconds: int = 7 * 60,
    ) -> None:
        self._session_store = session_store
        self._agent_registry = agent_registry
        self._wiki_client = wiki_client
        self._wikipath_client = wikipath_client
        self._safety_max_moves = safety_max_moves
        self._safety_max_seconds = safety_max_seconds

    def start(
        self,
        start_title: str,
        target_title: str,
        agent_id: Optional[str] = None,
        llm_choices: Optional[int] = None,
        retrieval_top_k: Optional[int] = None,
        tot_k: Optional[int] = None,
        tot_llm_candidates: Optional[int] = None,
        tot_expansions_per_step: Optional[int] = None,
        tot_score_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        agent = self._agent_registry.get(agent_id)
        session = agent.initialize_session(start_title=start_title, target_title=target_title)
        session["agent_id"] = agent.agent_id
        session.setdefault("started_at", time.time())
        if agent.agent_id == "default" and llm_choices is not None:
            n = max(1, llm_choices)
            session["llm_choices"] = n
            session["candidate_pool"] = max(n, CANDIDATE_POOL)
        if agent.agent_id == "planning" and retrieval_top_k is not None:
            session["retrieval_top_k"] = max(1, retrieval_top_k)
        if agent.agent_id == "tot":
            if tot_k is not None:
                session["tot_k"] = max(1, tot_k)
            if tot_llm_candidates is not None:
                session["tot_llm_candidates"] = max(1, tot_llm_candidates)
            if tot_expansions_per_step is not None:
                session["tot_expansions_per_step"] = max(1, tot_expansions_per_step)
            if tot_score_samples is not None:
                session["tot_score_samples"] = max(1, tot_score_samples)

        session["optimal_path"] = None
        session["optimal_length"] = None
        session["optimal_count"] = None
        session["wikipath_error"] = None

        if self._wikipath_client is not None and self._wiki_client is not None:
            try:
                source_id = self._wiki_client.get_page_id(session["resolved_start"])
                target_id = self._wiki_client.get_page_id(session["resolved_target"])
                result = self._wikipath_client.get_shortest_path(source_id, target_id)
                if result is not None:
                    session["optimal_length"] = result["length"]
                    session["optimal_count"] = result["count"]
                    # Resolve page IDs in the path back to titles
                    path_ids = result.get("path_ids", [])
                    if path_ids:
                        id_to_title = self._wiki_client.get_titles_from_ids(path_ids)
                        session["optimal_path"] = [
                            id_to_title.get(pid, f"(id:{pid})") for pid in path_ids
                        ]
                    else:
                        session["optimal_path"] = []
                else:
                    session["wikipath_error"] = "Wikipath returned no result."
            except Exception as exc:
                log.warning("Wikipath lookup failed: %s", exc)
                session["wikipath_error"] = str(exc)

        session_id = self._session_store.create(session)

        resp: Dict[str, Any] = {
            "session_id": session_id,
            "resolved_start": session["resolved_start"],
            "resolved_target": session["resolved_target"],
        }

        resp["optimal_path"] = session["optimal_path"]
        resp["optimal_length"] = session["optimal_length"]
        resp["optimal_count"] = session["optimal_count"]
        resp["wikipath_error"] = session["wikipath_error"]
        return resp

    def step(self, session_id: str) -> Tuple[Dict[str, Any], int]:
        session = self._session_store.get(session_id)
        if not session:
            return {"failure_reason": "Invalid session_id. Click Run again."}, 400

        if session["done"]:
            return self._payload(session, None, []), 200

        if (time.time() - session["started_at"]) > self._safety_max_seconds:
            session["done"] = True
            session["success"] = False
            session["failure_reason"] = f"Stopped by safety timer ({self._safety_max_seconds}s)."
            return self._payload(session, None, []), 200

        if len(session["moves"]) >= self._safety_max_moves:
            session["done"] = True
            session["success"] = False
            session["failure_reason"] = f"Stopped by safety move cap ({self._safety_max_moves})."
            return self._payload(session, None, []), 200

        try:
            token = current_session_var.set(session)
            try:
                agent = self._agent_registry.get(session["agent_id"])
                result = agent.step(session)
                if session.get("done") and session.get("success"):
                    session["reached_target_at"] = time.time()
                events, event = self._normalize_events(result)
                return self._payload(session, event, events), 200
            finally:
                current_session_var.reset(token)
        except Exception as exc:
            session["done"] = True
            session["success"] = False
            session["failure_reason"] = f"Agent step failed: {exc}"
            return self._payload(session, None, []), 200

    @staticmethod
    def _normalize_events(
        result: Optional[Any],
    ) -> Tuple[list, Optional[Dict[str, Any]]]:
        """Convert step result to (events list, primary event)."""
        if result is None:
            return [], None
        if isinstance(result, list):
            events = result
            # Primary event: last move/backtrack, or last event
            for ev in reversed(events):
                if isinstance(ev, dict) and ev.get("type") in ("move", "backtrack"):
                    return events, ev
            return events, events[-1] if events else None
        # Single event
        return [result], result

    @staticmethod
    def _payload(
        session: Dict[str, Any],
        event: Optional[Dict[str, Any]] = None,
        events: Optional[list] = None,
    ) -> Dict[str, Any]:
        hops = len(session["moves"])
        optimal_length = session.get("optimal_length")

        efficiency = None
        if session["done"] and session["success"] and optimal_length is not None and hops > 0:
            efficiency = round(optimal_length / hops, 4)

        llm_to_target_seconds = None
        first_llm_at = session.get("first_llm_at")
        reached_target_at = session.get("reached_target_at")
        if first_llm_at is not None and reached_target_at is not None:
            llm_to_target_seconds = round(reached_target_at - first_llm_at, 2)

        events_list = events if events is not None else ([event] if event else [])
        return {
            "done": session["done"],
            "success": session["success"],
            "failure_reason": session["failure_reason"],
            "resolved_start": session["resolved_start"],
            "resolved_target": session["resolved_target"],
            "hops": hops,
            "chain": session["chain_builder"](session["resolved_start"], session["moves"]),
            "steps_text": session["steps_builder"](
                session["resolved_start"], session["resolved_target"], session["moves"]
            ),
            "event": event,
            "events": events_list,
            # Wikipath comparison data
            "optimal_path": session.get("optimal_path"),
            "optimal_length": optimal_length,
            "optimal_count": session.get("optimal_count"),
            "wikipath_error": session.get("wikipath_error"),
            "efficiency": efficiency,
            "llm_to_target_seconds": llm_to_target_seconds,
        }

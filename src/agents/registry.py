from __future__ import annotations

from typing import Dict, Optional

from src.agents.base import GameAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, GameAgent] = {}
        self._default_agent_id: Optional[str] = None

    def register(self, agent: GameAgent, is_default: bool = False) -> None:
        self._agents[agent.agent_id] = agent
        if is_default or self._default_agent_id is None:
            self._default_agent_id = agent.agent_id

    def get(self, agent_id: Optional[str] = None) -> GameAgent:
        resolved_id = agent_id or self._default_agent_id
        if not resolved_id or resolved_id not in self._agents:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        return self._agents[resolved_id]

    def default_agent_id(self) -> str:
        if not self._default_agent_id:
            raise ValueError("No default agent registered")
        return self._default_agent_id

    def list_agent_ids(self) -> list[str]:
        return sorted(self._agents.keys())

    def agents(self) -> list[GameAgent]:
        return list(self._agents.values())

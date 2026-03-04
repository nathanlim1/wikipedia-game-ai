from __future__ import annotations

from src.agents.registry import AgentRegistry
from src.tinker_llm import TinkerLLM


class ModelManager:
    """
    Lazily initializes and caches TinkerLLM instances per model.
    Swaps the active model on all registered agents atomically.
    """

    def __init__(self, registry: AgentRegistry, valid_model_ids: set[str]) -> None:
        self._registry = registry
        self._valid_model_ids = valid_model_ids
        self._cache: dict[str, TinkerLLM] = {}
        self.current_id: str = ""

    def seed(self, model_id: str, client: TinkerLLM) -> None:
        """Pre-seed the cache with an already-initialized client."""
        self._cache[model_id] = client
        if not self.current_id:
            self.current_id = model_id

    def _get_or_create(self, model_id: str) -> TinkerLLM:
        if model_id not in self._cache:
            self._cache[model_id] = TinkerLLM(model=model_id)
        return self._cache[model_id]

    def set_model(self, model_id: str) -> None:
        """Switch all agents to use the given model. Blocks until the client is ready."""
        if model_id not in self._valid_model_ids:
            raise ValueError(f"Unknown model_id: {model_id!r}")
        client = self._get_or_create(model_id)
        for agent in self._registry.agents():
            agent.llm = client
        self.current_id = model_id

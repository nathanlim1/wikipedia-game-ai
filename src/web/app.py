from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.agents.default_agent import DefaultAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.tot_agent import ToTAgent
from src.agents.registry import AgentRegistry
from src.game.runner import GameRunner
from src.game.session_store import InMemorySessionStore
from src.llm_timing import TimingLLMWrapper
from src.tinker_llm import TinkerLLM
from src.web.model_manager import ModelManager
from src.web.routes import create_router
from src.web.ui_html import INDEX_HTML
from src.wikipedia import WikipediaClient
from src.wikipath_client import WikipathClient


DEFAULT_MODEL_ID = os.getenv("TINKER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

AVAILABLE_MODELS: list[dict[str, str]] = [
    {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507",   "label": "Qwen3-30B · 30B Instruction (default)"},
    {"id": "Qwen/Qwen3-235B-A22B-Instruct-2507",  "label": "Qwen3-235B · 235B Instruction"},
    {"id": "meta-llama/Llama-3.3-70B-Instruct",   "label": "Llama-3.3-70B · 70B Instruction"},
    {"id": "openai/gpt-oss-120b",                  "label": "GPT-OSS-120B · 120B Reasoning"},
]


def create_app() -> FastAPI:
    wiki_client = WikipediaClient()
    llm_client = TimingLLMWrapper(TinkerLLM(model=DEFAULT_MODEL_ID))

    registry = AgentRegistry()
    registry.register(DefaultAgent(wiki_client=wiki_client, llm_client=llm_client), is_default=True)
    registry.register(PlanningAgent(wiki_client=wiki_client, llm_client=llm_client))
    registry.register(ToTAgent(wiki_client=wiki_client, llm_client=llm_client))

    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    model_manager = ModelManager(registry=registry, valid_model_ids=valid_ids)
    model_manager.seed(DEFAULT_MODEL_ID, llm_client)

    wikipath_client = WikipathClient()

    runner = GameRunner(
        session_store=InMemorySessionStore(),
        agent_registry=registry,
        wiki_client=wiki_client,
        wikipath_client=wikipath_client,
        safety_max_moves=200,
        safety_max_seconds=7 * 60,
    )

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTMLResponse(INDEX_HTML)

    app.include_router(create_router(runner, registry, model_manager, AVAILABLE_MODELS))
    return app


app = create_app()

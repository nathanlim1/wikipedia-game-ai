from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.agents.default_agent import DefaultAgent
from src.agents.registry import AgentRegistry
from src.game.runner import GameRunner
from src.game.session_store import InMemorySessionStore
from src.tinker_llm import TinkerLLM
from src.web.routes import create_router
from src.web.ui_html import INDEX_HTML
from src.wikipedia import WikipediaClient


MODEL_NAME = os.getenv("TINKER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def create_app() -> FastAPI:
    wiki_client = WikipediaClient()
    llm_client = TinkerLLM(model=MODEL_NAME)

    registry = AgentRegistry()
    registry.register(DefaultAgent(wiki_client=wiki_client, llm_client=llm_client), is_default=True)

    runner = GameRunner(
        session_store=InMemorySessionStore(),
        agent_registry=registry,
        safety_max_moves=800,
        safety_max_seconds=7 * 60,
    )

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTMLResponse(INDEX_HTML)

    app.include_router(create_router(runner))
    return app


app = create_app()

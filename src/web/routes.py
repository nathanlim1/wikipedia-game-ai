from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.registry import AgentRegistry
from src.game.runner import GameRunner
from src.web.model_manager import ModelManager


class StartRequest(BaseModel):
    start_title: str
    target_title: str
    agent_id: Optional[str] = None
    llm_choices: Optional[int] = None
    retrieval_top_k: Optional[int] = None


class StepRequest(BaseModel):
    session_id: str


class SetModelRequest(BaseModel):
    model_id: str


def create_router(
    runner: GameRunner,
    registry: AgentRegistry,
    model_manager: ModelManager,
    available_models: list[dict[str, str]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/agents")
    def api_agents():
        return {
            "agents": registry.list_agent_ids(),
            "default": registry.default_agent_id(),
        }

    @router.get("/api/models")
    def api_models():
        return {
            "models": available_models,
            "current": model_manager.current_id,
        }

    @router.post("/api/models")
    def api_set_model(req: SetModelRequest):
        try:
            model_manager.set_model(req.model_id)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"Failed to load model: {exc}"}, status_code=500)
        return {"current": model_manager.current_id}

    @router.post("/api/start")
    def api_start(req: StartRequest):
        try:
            return runner.start(
                start_title=req.start_title,
                target_title=req.target_title,
                agent_id=req.agent_id,
                llm_choices=req.llm_choices,
                retrieval_top_k=req.retrieval_top_k,
            )
        except Exception as exc:
            return JSONResponse({"failure_reason": str(exc)}, status_code=400)

    @router.post("/api/step")
    def api_step(req: StepRequest):
        payload, status_code = runner.step(req.session_id)
        if status_code != 200:
            return JSONResponse(payload, status_code=status_code)
        return payload

    return router

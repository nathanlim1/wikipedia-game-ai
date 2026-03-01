from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.game.runner import GameRunner


class StartRequest(BaseModel):
    start_title: str
    target_title: str
    agent_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str


def create_router(runner: GameRunner) -> APIRouter:
    router = APIRouter()

    @router.post("/api/start")
    def api_start(req: StartRequest):
        try:
            return runner.start(
                start_title=req.start_title,
                target_title=req.target_title,
                agent_id=req.agent_id,
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

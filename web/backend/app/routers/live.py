"""Live endpoints: capability probe, runnable task sets, and the streaming driver.

`POST /api/live/run` is the one that does real work: it resolves a request into a fully
specified run, then streams Server-Sent Events while the AGH process works. The event names
match the replay lane's (`begin`/`step`/`tool_call`/`tool_result`/`final`/`end`) so the
frontend has one renderer for both.

A request may either name a task from a checked-in task set or carry its own prompt. A key
supplied in the body travels only as far as the child process's environment: it is not written
into the run directory, not echoed in the response, and not logged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import config, live

router = APIRouter(prefix="/api/live", tags=["live"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class LiveRequest(BaseModel):
    task_id: str | None = None
    prompt: str | None = None
    route: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    preset: str | None = None
    timeout: int = Field(default=config.LIVE_TIMEOUT, ge=10, le=1800)
    run_check: bool = True


def _resolve_key(route: str, supplied: str | None) -> str:
    """The key to hand the child: the request's, else the environment's, else the key file."""
    if supplied and supplied.strip():
        return supplied.strip()
    env_name = f"AGNES_SECRET_{route.upper().replace('-', '_')}_DEFAULT"
    from_env = os.environ.get(env_name, "").strip()
    if from_env:
        return from_env
    key_file = Path(config.LIVE_KEY_FILE)
    try:
        if key_file.is_file():
            return key_file.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


@router.get("/capabilities")
def get_capabilities(fresh: bool = False) -> dict:
    """Whether Live can run here, and what each route would use.

    `?fresh=true` skips the probe cache. The page needs it immediately after an operator starts
    an endpoint: a failed probe is remembered for 45 s so that a five-second poll does not spend
    its whole interval waiting on a closed port, and without this the page would keep showing a
    red light for most of a minute after the server came up.
    """
    caps = live.capabilities(fresh=fresh)
    tasks = live.list_tasks()
    caps["tasks"] = len(tasks)
    caps["tasks_with_fixtures"] = sum(1 for t in tasks if t["fixture_ready"])
    return caps


@router.get("/tasks")
def get_tasks() -> dict:
    tasks = live.list_tasks()
    return {"tasks": tasks, "count": len(tasks),
            "runnable": sum(1 for t in tasks if t["fixture_ready"])}


@router.post("/run")
async def run_live(req: LiveRequest) -> StreamingResponse:
    route = (req.route or config.LIVE_ROUTE).strip()
    model = (req.model or config.LIVE_MODEL).strip()
    base_url = (req.base_url or config.LIVE_BASE_URL).strip()
    preset = (req.preset or config.LIVE_PRESET).strip()

    task = live.find_task(req.task_id) if req.task_id else None
    prompt = (req.prompt or (task or {}).get("prompt") or "").strip()
    check = (task or {}).get("check") if (task and req.run_check) else None
    fixture = Path(task["fixture_path"]) if (task or {}).get("fixture_path") else None

    async def gen():
        if not prompt:
            yield _sse("live_error", {
                "message": "no prompt: pass task_id or prompt",
                "hint": "GET /api/live/tasks lists runnable tasks"})
            return
        spec = live.LiveSpec(
            task_id=(task or {}).get("task_id") or "ad-hoc",
            prompt=prompt, route=route, model=model, base_url=base_url, preset=preset,
            timeout=req.timeout, api_key=_resolve_key(route, req.api_key),
            expect=list((task or {}).get("expect") or []),
            forbid=list((task or {}).get("forbid") or []),
            check=check, fixture=fixture)
        try:
            async for event, payload in live.stream_run(spec):
                yield _sse(event, payload)
        except Exception as error:  # a driver failure is reported, not swallowed
            yield _sse("live_error", {"message": f"{type(error).__name__}: {error}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

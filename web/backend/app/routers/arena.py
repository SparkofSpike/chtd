from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import live, state

router = APIRouter(prefix="/api/arena", tags=["arena"])

MIN_FRAME = 0.12  # seconds, floor between replay frames


class ArenaRequest(BaseModel):
    mode: str = "replay"  # replay|live; auto-detected if omitted
    task_id: str | None = None
    arms: list[str] = ["base", "chtd"]
    collection: str = "holdout-4-20261014-v3"
    speed: float = 1.0


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_lanes(task_id: str, collection: str, wants: list[str]) -> list[dict]:
    """Pick up to two record index rows for the requested task, base-like then chtd-like."""
    cands = [r for r in state.SESSIONS
             if r.get("task_id") == task_id and r.get("batch") == collection]
    if not cands:
        # any batch
        cands = [r for r in state.SESSIONS if r.get("task_id") == task_id]
    if not cands:
        return []

    def score(r: dict) -> int:
        arm = (r.get("arm") or "").lower()
        kind = (r.get("arm_kind") or "").lower()
        s = 0
        if "nocheck" in arm:
            s += 2  # prefer the main rollouts
        if "base" in arm and "control" not in arm and "smoke" not in arm:
            s += 10
        if "chtd" in arm and "seed0" in arm:
            s += 6
        if "smoke" in arm:
            s -= 5
        return s

    cands.sort(key=score, reverse=True)
    lanes = []
    seen_kind = set()
    for r in cands:
        kind = r.get("arm_kind")
        # prefer one base-like, one chtd-like
        if any((r.get("arm") or "").lower() in l for l in lanes):
            continue
        lanes.append(r)
        if len(lanes) >= 2:
            break
    if not lanes:
        lanes = cands[:2]
    return lanes


@router.post("/run")
async def run_arena(req: ArenaRequest) -> StreamingResponse:
    health = live.health()
    effective = health["mode"]
    want_live = req.mode == "live"

    async def gen():
        fallback_sent = False
        if want_live and effective != "live":
            yield _sse("mode_fallback", {
                "from": "live", "to": "replay", "reason": live.FALLBACK_MESSAGE})
            fallback_sent = True

        lanes = _resolve_lanes(req.task_id, req.collection, req.arms)
        if not lanes:
            yield _sse("error", {"message": f"no recorded session for task_id='{req.task_id}'"})
            return

        lane_names = [l.get("arm") for l in lanes]
        yield _sse("begin", {
            "mode": "live" if (want_live and effective == "live") else "replay",
            "effective_mode": effective,
            "lanes": lane_names,
            "task_id": req.task_id,
            "track_sha256": lanes[0].get("track_sha256"),
            "replay_note": "replaying recorded events" if not fallback_sent else live.FALLBACK_MESSAGE,
        })

        delay = max(MIN_FRAME, (0.4 / max(req.speed, 0.1)))
        for rec in lanes:
            lane = rec.get("arm")
            sess = state.get_session(rec.get("session_key")) or rec
            steps = sess.get("steps", [])
            for step in steps:
                yield _sse("step", {"lane": lane, "step_index": step.get("index"),
                                    "plan": step.get("plan")})
                if step.get("recovery"):
                    yield _sse("recovery", {"lane": lane, "step_index": step.get("index"),
                                            "after_error": True})
                for call in step.get("calls", []):
                    yield _sse("tool_call", {"lane": lane, "step_index": step.get("index"),
                                             "name": call.get("name"), "args": call.get("args"),
                                             "ordinal": call.get("ordinal")})
                    res = call.get("result", {})
                    yield _sse("tool_result", {"lane": lane, "step_index": step.get("index"),
                                               "name": call.get("name"),
                                               "content": res.get("content", ""),
                                               "isError": res.get("isError", False),
                                               "duration_ms": call.get("duration_ms")})
                    await asyncio.sleep(delay)
            yield _sse("final", {"lane": lane, "answer": sess.get("final_answer"),
                                 "verdict": sess.get("verdict")})
            yield _sse("end", {"lane": lane,
                               "session": {"passed": sess.get("passed"),
                                           "check_output": sess.get("check_output"),
                                           "seconds": sess.get("seconds")},
                               "track_sha256": rec.get("track_sha256")})
        yield _sse("done", {"lanes": lane_names})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

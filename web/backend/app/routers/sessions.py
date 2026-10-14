from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import state
from ..loaders.sessions import canonical_sort_key

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions(
    q: str | None = None,
    arm: str | None = None,
    result: str | None = Query(None, pattern="^(pass|fail)$"),
    batch: str | None = None,
) -> dict:
    rows = state.SESSIONS
    out = []
    for r in rows:
        if arm and r.get("arm") != arm:
            continue
        if batch and r.get("batch") != batch:
            continue
        if result == "pass" and not r.get("passed"):
            continue
        if result == "fail" and r.get("passed"):
            continue
        if q:
            needle = q.lower()
            hay = " ".join(str(r.get(k, "")) for k in ("task_id", "arm", "answer", "session_key")).lower()
            if needle not in hay:
                continue
        out.append({
            "batch": r.get("batch"),
            "arm": r.get("arm"),
            "arm_group": r.get("arm_group"),
            "shard": r.get("shard"),
            "task_id": r.get("task_id"),
            "passed": r.get("passed"),
            "seconds": r.get("seconds"),
            "exit": r.get("exit"),
            "timed_out": r.get("timed_out"),
            "session_key": r.get("session_key"),
            "track_sha256": r.get("track_sha256"),
            "preset": r.get("preset"),
            "paired": state.PAIRS.get(r.get("session_key"), {}),
        })

    out.sort(key=canonical_sort_key)

    arms = sorted({r.get("arm") for r in rows if r.get("arm")})
    groups = sorted({r.get("arm_group") for r in rows if r.get("arm_group")})
    tasks = sorted({r.get("task_id") for r in rows if r.get("task_id")})
    return {
        "count": len(out),
        "sessions": out,
        "filters": {"arms": arms, "groups": groups, "results": ["pass", "fail"],
                    "tasks": tasks},
    }


@router.get("/{session_key}")
def get_session(session_key: str) -> dict:
    sess = state.get_session(session_key)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"unknown session_key '{session_key}'")
    return sess

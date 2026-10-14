from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from .. import live

router = APIRouter(tags=["health"])


@router.get("/api/health")
def get_health() -> dict:
    h = live.health()
    h["server_time"] = datetime.now(timezone.utc).isoformat()
    return h

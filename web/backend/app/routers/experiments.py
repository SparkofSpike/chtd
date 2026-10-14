from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

NAMES = ["summary", "failure_rate", "dose_ladder", "pooled", "determinism",
         "tool_health", "scorer_power", "early_eval"]


@router.get("")
def index() -> dict:
    items = []
    for name in NAMES:
        block = state.EXPERIMENTS.get(name) or {}
        meta = block.get("_meta", {}) if isinstance(block, dict) else {}
        items.append({"name": name, "has_data": bool(block),
                      "status": meta.get("status", "ok") if block else "暂无数据"})
    return {"experiments": items}


@router.get("/{name}")
def get_experiment(name: str) -> dict:
    if name not in NAMES:
        raise HTTPException(status_code=404, detail=f"unknown experiment '{name}'")
    block = state.EXPERIMENTS.get(name)
    if not block:
        return {"name": name, "status": "暂无数据", "_meta": {"sources": []}}
    return block

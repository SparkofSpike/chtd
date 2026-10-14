from __future__ import annotations

from fastapi import APIRouter

from ..loaders.paths import PATHS

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _parse_set(name: str) -> dict:
    path = PATHS.task_file(name)
    tasks = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            import json
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            tasks.append({
                "id": row.get("id"),
                "prompt": row.get("prompt"),
                "why": row.get("why"),
            })
    return {"name": name, "count": len(tasks), "tasks": tasks}


@router.get("")
def list_task_sets() -> dict:
    return {"sets": [_parse_set(n) for n in PATHS.TASK_SETS]}

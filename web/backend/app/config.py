"""Runtime configuration for the CHTD web backend.

Everything here is read-only safe: no training, no model imports. The backend
only parses checked-in JSON / JSONL / SQLite ledgers and forwards HTTP health
checks to locally-served model endpoints.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# backend/  (this file lives in backend/app/config.py)
BACKEND_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BACKEND_DIR.parent


def _find_chtd_root() -> Path:
    """Locate the chtd repo root by walking up until runs/, data/ and harness/ coexist.

    Overridable with the CHTD_ROOT env var (used by tests and non-default layouts).
    """
    env = os.environ.get("CHTD_ROOT")
    if env:
        return Path(env).resolve()
    here = BACKEND_DIR
    for candidate in [here, *here.parents]:
        if (candidate / "runs").is_dir() and (candidate / "data").is_dir() and (candidate / "harness").is_dir():
            return candidate
    # Fallback: web/backend is two levels below chtd root in the shipped layout.
    return WEB_DIR.parent


CHTD_ROOT = _find_chtd_root()

# Model endpoints (served by serve_openai.py on the box; absent locally).
BASE_URL = os.environ.get("CHTD_BASE_URL", "http://127.0.0.1:8000")
ADAPTER_URL = os.environ.get("CHTD_ADAPTER_URL", "http://127.0.0.1:8001")
BASE_PORT = int(os.environ.get("CHTD_BASE_PORT", "8000"))
ADAPTER_PORT = int(os.environ.get("CHTD_ADAPTER_PORT", "8001"))

# Agnes CLI (absent on this machine; live.py degrades to replay when missing).
AGNES_CLI = os.environ.get(
    "CHTD_AGNES_CLI",
    r"E:\Coding\agnes-harness\packages\cli\dist\local\agnes.mjs",
)
NODE_BIN = os.environ.get("CHTD_NODE", "node")
PROFILE_TEMPLATE = os.environ.get(
    "CHTD_PROFILE_TEMPLATE",
    str(CHTD_ROOT / "data" / "holdout-4-20261014-v3" / "profile-source.yaml"),
)

# HTTP (8080 is occupied on this box by an unrelated app; default 8088)
SERVER_HOST = os.environ.get("CHTD_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("CHTD_BACKEND_PORT", "8088"))

# Frontend static dist (may be absent; backend must still start).
FRONTEND_DIST = WEB_DIR / "frontend" / "dist"

# Live scratch dir (only place the backend writes at runtime).
LIVE_DIR = BACKEND_DIR / ".live"
LIVE_RUNS_DIR = LIVE_DIR / "runs"

# Live driving. Defaults point at this project's recorded pair; every one of them is either an
# env var here or a field on the run request, so a deployment with a different model pair edits
# the page rather than this file. The teacher is a hosted route (no GPU needed); the student is
# whatever OpenAI-compatible endpoint the operator serves.
LIVE_ROUTE = os.environ.get("CHTD_LIVE_ROUTE", "teacher")
LIVE_MODEL = os.environ.get("CHTD_LIVE_MODEL", "agnes-3.0-flash")
LIVE_BASE_URL = os.environ.get("CHTD_LIVE_BASE_URL", "https://api.agnes-ai.cn/v1")
LIVE_PRESET = os.environ.get("CHTD_LIVE_PRESET", "minimal-rl")
LIVE_TIMEOUT = int(os.environ.get("CHTD_LIVE_TIMEOUT", "240"))
#: Where a route's credential comes from when the request does not carry one. The file is read at
#: run time and never copied into a run directory, so a key stays in one place on disk.
LIVE_KEY_FILE = os.environ.get(
    "CHTD_LIVE_KEY_FILE", str(CHTD_ROOT / "data" / "chtd-round-20261011" / "teacher.key"))
#: Task sets the Live panel can run, newest first. The fixture directory holds one subdirectory
#: per task id; a second directory may be listed as a fallback root.
LIVE_TASK_SETS = [
    (CHTD_ROOT / "harness" / "tasks" / "holdout-3.jsonl",
     CHTD_ROOT / "data" / "holdout-3-20261014" / "workspaces",
     CHTD_ROOT / "harness" / "tasks" / "holdout-3"),
    (CHTD_ROOT / "harness" / "tasks" / "final-holdout.jsonl",
     CHTD_ROOT / "data" / "holdout-4-20261014-v3" / "workspaces",
     None),
]
#: Substrings that must not appear in a task id used to build a path component.
LIVE_TASK_ID_OK = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

# Exported read-only JSON cache.
DATA_DIR = BACKEND_DIR / "data"

#: Per-probe budget. A local endpoint answers in milliseconds; one that is not listening either
#: refuses at once or never answers, and this machine takes the slow path (~2 s to a closed
#: 127.0.0.1 port), so the budget is what keeps a five-second health poll from stalling on it.
ENDPOINT_TIMEOUT = 0.6

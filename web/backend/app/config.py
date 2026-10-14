"""Runtime configuration for the CHTD web backend.

Everything here is read-only safe: no training, no model imports. The backend
only parses checked-in JSON / JSONL / SQLite ledgers and forwards HTTP health
checks to locally-served model endpoints.
"""
from __future__ import annotations

import os
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

# Exported read-only JSON cache.
DATA_DIR = BACKEND_DIR / "data"

ENDPOINT_TIMEOUT = 1.5  # seconds, per health probe

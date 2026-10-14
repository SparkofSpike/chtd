"""In-memory application state, loaded once at startup."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config
from .loaders import sessions as sess_loader

# Experiment exports keyed by name (summary, failure_rate, ...).
EXPERIMENTS: dict[str, dict[str, Any]] = {}
# Flat record index + pairing.
SESSIONS: list[dict[str, Any]] = []
PAIRS: dict[str, dict[str, str]] = {}
# Lookup: session_key -> record index entry.
SESSION_BY_KEY: dict[str, dict[str, Any]] = {}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_all() -> None:
    """Load exported experiment JSONs and build the session index.

    Missing export files degrade gracefully (empty experiments) rather than
    crashing the server.
    """
    EXPERIMENTS.clear()
    if config.DATA_DIR.is_dir():
        for f in config.DATA_DIR.glob("*.json"):
            if f.name in ("batches.json", "sessions_index.json", "provenance.json"):
                continue
            EXPERIMENTS[f.stem] = _load_json(f) or {}

    global SESSIONS, PAIRS, SESSION_BY_KEY
    SESSIONS = sess_loader.discover_records()
    PAIRS = sess_loader.build_pairs(SESSIONS)
    for r in SESSIONS:
        sk = r.get("session_key")
        if sk:
            SESSION_BY_KEY[sk] = r


def get_session(session_key: str) -> dict[str, Any] | None:
    rec = SESSION_BY_KEY.get(session_key)
    if rec is None:
        return None
    return sess_loader.build_full_session(rec)

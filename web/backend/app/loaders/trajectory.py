"""Reconstruct a standardized Session trajectory from raw AGH ledger events.

Pure stdlib. Input: events of one session (ordered by seq). Output: the Session
shape described in PLAN §5.1. Read-only; no torch / transformers.
"""
from __future__ import annotations

import re
from typing import Any

# Absolute paths belonging to whoever ran the sweep leak into tool results (an ENOENT message
# carries the full path it tried). The demo ships those traces publicly, so the machine-specific
# prefix is collapsed to <workspace> / <home> while the meaningful tail is kept: a viewer still
# sees which relative path failed, just not whose disk it was.
_PATH_RULES = [
    re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\\\s'\"]+\\\\?(?:[^\\\\\s'\"]+\\\\?)*?ML_Project\\\\?chtd", re.I),
    re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\\\s'\"]+", re.I),
    re.compile(r"/home/[a-z][\w.\-]*", re.I),
    re.compile(r"/Users/[A-Za-z][\w.\-]*"),
]


def sanitize_text(s: str) -> str:
    """Collapse third-party absolute paths to <workspace> / <home>."""
    if not s:
        return s
    out = s
    for rx in _PATH_RULES[:1]:
        out = rx.sub("<workspace>", out)
    out = _PATH_RULES[1].sub("<workspace>", out)
    out = _PATH_RULES[2].sub("/home/<redacted>", out)
    out = _PATH_RULES[3].sub("/Users/<redacted>", out)
    return out


def _text_of(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            parts.append(item["text"])
    return sanitize_text("\n".join(parts))


def build_session(meta: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a Session dict from ledger events.

    meta carries the records.jsonl row (arm, task_id, passed, seconds, ...) plus
    session/start-derived fields if available. events are already parsed dicts
    in ascending seq order.
    """
    session: dict[str, Any] = {
        "session_key": meta.get("session_key"),
        "arm": meta.get("arm"),
        "task_id": meta.get("task_id"),
        "preset": meta.get("preset"),
        "track_sha256": meta.get("track_sha256"),
        "profile_hash": meta.get("profile_hash"),
        "model_settings": meta.get("model_settings", []),
        "user_prompt": sanitize_text(meta.get("user_prompt") or ""),
        "steps": [],
        "final_answer": sanitize_text(meta.get("answer") or ""),
        "verdict": {"passed": meta.get("passed")},
        "passed": meta.get("passed"),
        "check_output": sanitize_text(meta.get("check_output") or ""),
        "seconds": meta.get("seconds"),
        "exit": meta.get("exit"),
        "timed_out": meta.get("timed_out"),
    }

    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    calls_by_id: dict[str, dict[str, Any]] = {}
    final_answer: str | None = None
    pending_error = False  # set when a tool result isError; next step -> recovery

    def close_step() -> None:
        if current is not None:
            durs = current.pop("_durs", [])
            for i, call in enumerate(current["calls"]):
                if i < len(durs):
                    call["duration_ms"] = durs[i]
            current.pop("_by_id", None)
            steps.append(current)

    for ev in events:
        etype = ev.get("type")
        data = ev.get("data") or {}
        if not isinstance(data, dict):
            continue

        if etype == "session/start":
            session["profile_hash"] = data.get("resolvedProfileHash") or session.get("profile_hash")
            session["preset"] = data.get("preset") or session.get("preset")
            ms = data.get("modelSettings")
            if isinstance(ms, list):
                session["model_settings"] = [
                    {
                        "slot": m.get("slot"),
                        "route": m.get("route"),
                        "model": m.get("model"),
                        "contextWindow": (m.get("settings") or {}).get("contextWindow"),
                    }
                    for m in ms if isinstance(m, dict)
                ]
                session["declared_model_settings"] = [dict(m) for m in session["model_settings"]]
        elif etype == "x/core/model-switch":
            # The declared settings at session/start name the profile's *default* route, which is not
            # necessarily the model that executes: the harness switches slots right after startup.
            # Track the switch so the replay reports the model that actually answered.
            to = data.get("to") or {}
            if not to:
                continue
            slot = data.get("slot") or "primary"
            updated = False
            for m in session["model_settings"]:
                if m.get("slot") == slot:
                    m["route"] = to.get("route")
                    m["model"] = to.get("model")
                    m["contextWindow"] = to.get("contextWindow", m.get("contextWindow"))
                    m["switched"] = True
                    updated = True
                    break
            if not updated:
                session["model_settings"].append({
                    "slot": slot,
                    "route": to.get("route"),
                    "model": to.get("model"),
                    "contextWindow": to.get("contextWindow"),
                    "switched": True,
                })
            session.setdefault("model_switches", []).append({
                "slot": slot,
                "from": data.get("from") or {},
                "to": to,
            })
        elif etype == "user/message":
            txt = _text_of(data.get("content"))
            if txt:
                session["user_prompt"] = txt
        elif etype == "step/start":
            close_step()
            current = {
                "index": len(steps),
                "turn": data.get("turn"),
                "step": data.get("step"),
                "plan": "",
                "calls": [],
                "verifier": None,
                "recovery": pending_error,
                "_durs": [],
                "_by_id": {},
            }
            pending_error = False
        elif etype == "assistant/message":
            txt = _text_of(data.get("content"))
            if not txt:
                continue
            if current is not None and not current["calls"]:
                current["plan"] = (current["plan"] + "\n" + txt).strip() if current["plan"] else txt
            if data.get("stopReason") != "tool_use":
                final_answer = txt
        elif etype == "tool/call":
            tid = data.get("toolUseId")
            policy = data.get("resolvedPolicy") or {}
            call = {
                "name": data.get("name"),
                "args": data.get("args") or {},
                "ordinal": data.get("ordinal"),
                "result": {"content": "", "isError": False},
                "duration_ms": None,
                "policy": {
                    "isReadOnly": policy.get("isReadOnly"),
                    "isDestructive": policy.get("isDestructive"),
                    "isOpenWorld": policy.get("isOpenWorld"),
                    "requiresApproval": policy.get("requiresApproval"),
                },
                "_id": tid,
            }
            if current is not None:
                current["calls"].append(call)
                current["_by_id"][tid] = call
        elif etype == "tool/result":
            tid = data.get("toolUseId")
            content = _text_of(data.get("content"))
            is_err = bool(data.get("isError"))
            call = None
            if current is not None:
                call = current["_by_id"].get(tid)
            if call is None:
                call = calls_by_id.get(tid)
            if call is not None:
                call["result"] = {"content": content, "isError": is_err}
            if is_err:
                pending_error = True
        elif etype == "effect/settled":
            if current is not None:
                dm = data.get("durationMs")
                if isinstance(dm, (int, float)):
                    current["_durs"].append(dm)
        elif etype == "verifier/signal":
            scope = data.get("scope")
            if scope in ("step", "turn") and current is not None:
                current["verifier"] = {
                    "scope": scope,
                    "verdict": data.get("verdict"),
                    "reasons": data.get("reasons", []),
                }

    close_step()
    if final_answer:
        session["final_answer"] = final_answer
    session["steps"] = steps
    return session

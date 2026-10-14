"""A refusal is not a malfunction, and the reason the page has to know.

Measured across the runs in this repo: of 92 error tool-results, 64 are the boundary working as
designed — 56 approval refusals and 8 workspace refusals — and the model adapted after every one.
Rendering those in the same red chip as a genuine failure reads as "the system is broken", which
is the opposite of what the runs show.
"""
from __future__ import annotations

import pytest

from app.loaders.trajectory import build_session, classify_result

#: (content, isError, expected kind). Every string is a real message from a ledger.
CASES = [
    ("ls refused: /app is outside the workspace. Switch to Full permissions "
     "(完全权限) or select the target directory as the workspace (选择目标目录作为工作区).",
     True, "guardrail"),
    ("read failed: E_FS_DENIED: /root/working/check.py is outside every allow rule; "
     "请选择“完全权限”或将目标所在目录设为工作区。",
     True, "guardrail"),
    ("the user rejected this action; do not retry the same call without asking",
     True, "approval"),
    ("WEB_NETWORK_ERROR: Public web retrieval failed (DNS, TLS, connection or response stream)",
     True, "model"),
    ("WEB_FETCH_TIMEOUT: Public web retrieval timed out", True, "model"),
    ("read failed: ENOENT: no such file or directory, open 'x.py'", True, "model"),
    ("read failed: EISDIR: illegal operation on a directory, read", True, "model"),
    ("tool arguments do not match the registered schema: /path: missing required parameter.",
     True, "model"),
    ("check.py\ntally.py", False, "ok"),
]


@pytest.mark.parametrize(("content", "is_error", "expected"), CASES,
                         ids=[c[2] + ":" + c[0][:28] for c in CASES])
def test_result_classification(content, is_error, expected):
    assert classify_result(is_error, content) == expected


def test_a_success_is_ok_whatever_it_says():
    # A tool result that quotes a refusal (a model reading a log file, say) is still a success.
    assert classify_result(False, "ls refused: /app is outside the workspace") == "ok"


def _events(results):
    """A minimal session: one step, one call per (name, content, isError)."""
    events = [{"seq": 1, "type": "step/start", "data": {"turn": 1, "step": 1}}]
    for i, (name, content, is_err) in enumerate(results):
        tid = f"t{i}"
        events.append({"seq": 10 + i * 2, "type": "tool/call",
                       "data": {"name": name, "args": {}, "ordinal": i, "toolUseId": tid}})
        events.append({"seq": 11 + i * 2, "type": "tool/result",
                       "data": {"toolUseId": tid, "isError": is_err,
                                "content": [{"type": "text", "text": content}]}})
    return events


def test_a_guardrail_refusal_is_marked_as_a_recovery_not_an_error():
    session = build_session({}, _events([
        ("ls", "ls refused: /app is outside the workspace. Switch to Full permissions", True),
        ("ls", "check.py", False),
    ]))
    call = session["steps"][0]["calls"][0]
    assert call["result"]["kind"] == "guardrail"
    assert call["result"]["isError"] is True
    # the kind travels on the step too, so the label can name what was refused
    assert session["steps"][0]["recovery_kind"] is None


def test_the_next_step_names_what_it_is_recovering_from():
    events = _events([
        ("shell", "the user rejected this action", True),
    ])
    events.append({"seq": 30, "type": "step/start", "data": {"turn": 1, "step": 2}})
    events.append({"seq": 31, "type": "tool/call",
                   "data": {"name": "ls", "args": {}, "ordinal": 0, "toolUseId": "t9"}})
    events.append({"seq": 32, "type": "tool/result",
                   "data": {"toolUseId": "t9", "isError": False,
                            "content": [{"type": "text", "text": "check.py"}]}})
    session = build_session({}, events)
    assert session["steps"][0]["recovery_kind"] is None
    second = session["steps"][1]
    assert second["recovery"] is True
    assert second["recovery_kind"] == "approval"


def test_a_real_failure_is_not_dressed_up_as_a_boundary():
    # The whole point of the classification is that this one still reads as an error.
    session = build_session({}, _events([
        ("read", "read failed: ENOENT: no such file or directory, open 'missing.py'", True),
    ]))
    assert session["steps"][0]["calls"][0]["result"]["kind"] == "model"

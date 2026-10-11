"""Provenance and holdout checks for the exploratory tool-anchor sample builder."""

from __future__ import annotations

import pytest

from distill.build_tool_anchor import build_rows


def _source(task_id: str = "fix-assertion") -> dict:
    return {
        "task_id": task_id,
        "prefix_digest": "p" * 64,
        "harness_prefix_digest": "h" * 64,
        "tools": [{"name": "ls"}],
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Find the defect"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "ls", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "check.py", "tool_call_id": "x"},
            {"role": "user", "content": "Diagnose"},
            {"role": "assistant", "content": "The expected value is wrong."},
        ],
    }


def test_shared_tool_label_is_explicitly_student_sourced() -> None:
    shared, teacher = build_rows([_source()], ["fix-assertion"])
    assert shared["label_source"] == "student_shared_tool_anchor"
    assert shared["teacher_turn_indices"] == []
    assert len(shared["messages"]) == 3
    assert teacher["label_source"] == "teacher_on_student_state"
    assert teacher["teacher_turn_indices"] == [5]
    assert teacher["messages"][-1]["content"] == "The expected value is wrong."


def test_builder_refuses_held_out_tasks_and_non_ls_anchors() -> None:
    with pytest.raises(ValueError, match="held-out"):
        build_rows([_source("fix-none-cap")], ["fix-none-cap"])
    row = _source()
    row["messages"][2]["tool_calls"][0]["function"]["name"] = "read"
    with pytest.raises(ValueError, match="first student action"):
        build_rows([row], ["fix-assertion"])

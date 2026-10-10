"""The control arm must train on the teacher's own observed states, not failed runs."""

from __future__ import annotations

import json
import hashlib

import pytest

from distill.baseline import build_candidates
from harness.export_baseline import export_collection
from tests.ledger_fixture import build_ledger


def _collection(tmp_path, *, path: str = "notes.txt"):
    ledger = tmp_path / "sessions.db"
    session = "agnes:local:local-dev:cli:workspace:" + "a" * 16
    builder = build_ledger(ledger)
    builder.session(session).start(session, preset="minimal-rl")
    builder.user(session, "Find the defect")
    builder.tool_call(session, "read", {"path": path}, tool_use_id="call-1")
    builder.tool_result(session, "call-1", text="broken line")
    builder.assistant(session, "The defect is on that line.")
    builder.close()

    collection = tmp_path / "collection"
    (collection / "tasks").mkdir(parents=True)
    (collection / "collection-manifest.json").write_text(json.dumps({
        "task_ids": ["good", "failed"], "holdout_ids": ["holdout"]
    }), encoding="utf-8")
    (collection / "tasks" / "good.json").write_text(json.dumps({
        "task_id": "good", "passed": True,
        "record": {"task_id": "good", "arm": "teacher", "track_sha256": "a" * 64,
                   "ledger_path": str(ledger), "session_key": session},
    }), encoding="utf-8")
    (collection / "tasks" / "failed.json").write_text(json.dumps({
        "task_id": "failed", "passed": False, "record": None,
    }), encoding="utf-8")
    prefix = tmp_path / "prefix.json"
    prefix.write_text(json.dumps({
        "provenance": {"preset": "minimal-rl"},
        "system": "You are on the frozen track.",
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }), encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "collection_manifest_sha256": hashlib.sha256(
            (collection / "collection-manifest.json").read_bytes()).hexdigest(),
        "accepted_task_ids": ["good"],
        "rejected": {"failed": "teacher run failed"},
    }), encoding="utf-8")
    return collection, prefix, review


def test_control_samples_use_teacher_state_and_exclude_failed_task(tmp_path) -> None:
    collection, prefix, review = _collection(tmp_path)
    output = tmp_path / "out"
    manifest = build_candidates(collection, prefix, review, output)
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["candidate_count"] == 2
    assert manifest["refusal_count"] == 1
    assert all(row["task_id"] == "good" for row in rows)
    assert [message["role"] for message in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert rows[0]["messages"][-1]["tool_calls"][0]["function"]["name"] == "read"
    assert [message["role"] for message in rows[1]["messages"]] == [
        "system", "user", "assistant", "tool", "assistant"
    ]


def test_control_refuses_absolute_path_labels(tmp_path) -> None:
    collection, prefix, review = _collection(tmp_path, path="/outside/notes.txt")
    manifest = build_candidates(collection, prefix, review, tmp_path / "out")
    assert manifest["candidate_count"] == 1
    refusals = (tmp_path / "out" / "refusals.jsonl").read_text(encoding="utf-8")
    assert "non-portable absolute path" in refusals


def test_control_refuses_train_holdout_overlap(tmp_path) -> None:
    collection, prefix, review = _collection(tmp_path)
    (collection / "collection-manifest.json").write_text(json.dumps({
        "task_ids": ["good"], "holdout_ids": ["good"]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="train/holdout task overlap"):
        build_candidates(collection, prefix, review, tmp_path / "out")


def test_control_refuses_incomplete_manual_review(tmp_path) -> None:
    collection, prefix, review = _collection(tmp_path)
    review.write_text(json.dumps({
        "collection_manifest_sha256": hashlib.sha256(
            (collection / "collection-manifest.json").read_bytes()).hexdigest(),
        "accepted_task_ids": ["good"], "rejected": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="partition every training task"):
        build_candidates(collection, prefix, review, tmp_path / "out")


def test_portable_export_rebuilds_identical_samples(tmp_path) -> None:
    collection, prefix, review = _collection(tmp_path)
    original = build_candidates(collection, prefix, review, tmp_path / "original")
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("sk-synthetic-test-credential", encoding="utf-8")
    exported = tmp_path / "portable"
    summary = export_collection(collection, exported, prefix, review, secret_file)
    assert summary["samples_sha256"] == original["samples_sha256"]
    result = json.loads((exported / "collection/tasks/good.json").read_text(encoding="utf-8"))
    assert result["record"]["ledger_path"] == "sessions.db"

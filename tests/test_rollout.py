"""Rollouts are recorded, not remembered. These tests use a replay runner, not a model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.rollout import (
    ReplayRunner,
    RolloutRecord,
    Task,
    collect,
    load_tasks,
    read_records,
    read_track_digest,
    write_records,
)
from tests.ledger_fixture import build_ledger

TRACK = "a" * 64


def _prepared_ledger(directory: Path, *, session: str = "student", decisions: int = 2) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    builder = build_ledger(directory / "sessions.db")
    builder.session(session).start(session)
    builder.user(session, "do the thing")
    for index in range(decisions):
        builder.tool_call(session, "read", {"path": f"f{index}.txt"})
    builder.close()
    return directory


def _tasks_file(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_task_set_round_trips(tmp_path: Path) -> None:
    path = _tasks_file(
        tmp_path,
        [{"id": "t1", "prompt": "fix the test"}, {"id": "t2", "prompt": "add a flag", "cwd": "/w"}],
    )
    tasks = load_tasks(path)
    assert [t.id for t in tasks] == ["t1", "t2"]
    assert tasks[1].cwd == "/w"


def test_a_duplicate_task_id_is_refused(tmp_path: Path) -> None:
    # Two runs colliding on one output file: the later silently replaces the earlier, which looks
    # exactly like a task that never ran.
    path = _tasks_file(tmp_path, [{"id": "t1", "prompt": "a"}, {"id": "t1", "prompt": "b"}])
    with pytest.raises(ValueError, match="repeats task id t1"):
        load_tasks(path)


def test_an_empty_task_set_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no tasks"):
        load_tasks(path)


def test_a_task_without_a_prompt_is_refused(tmp_path: Path) -> None:
    path = _tasks_file(tmp_path, [{"id": "t1"}])
    with pytest.raises(ValueError, match="non-empty string prompt"):
        load_tasks(path)


def test_collect_records_a_row_per_task_and_keeps_the_ledgers_apart(tmp_path: Path) -> None:
    tasks = (Task(id="t1", prompt="a"), Task(id="t2", prompt="b"))
    sources = {
        "t1": str(_prepared_ledger(tmp_path / "src-t1")),
        "t2": str(_prepared_ledger(tmp_path / "src-t2")),
    }
    outcome = collect(
        tasks,
        ReplayRunner(sources),
        arm="student",
        model="qwen-small",
        route="student",
        home_root=tmp_path / "homes",
        track_sha256=TRACK,
    )
    assert len(outcome.records) == 2
    assert all(record.track_sha256 == TRACK for record in outcome.records)
    assert all(record.arm == "student" for record in outcome.records)
    assert outcome.records[0].decisions == 2
    # Each run got its own home, so its ledger is its own record.
    assert outcome.records[0].ledger_path != outcome.records[1].ledger_path
    assert set(outcome.trajectories) == {"student/t1", "student/t2"}


def test_collect_refuses_a_task_with_no_prepared_run(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no prepared ledger"):
        collect(
            (Task(id="missing", prompt="a"),),
            ReplayRunner({}),
            arm="student",
            model="m",
            route="r",
            home_root=tmp_path / "homes",
            track_sha256=TRACK,
        )


def test_records_survive_a_round_trip(tmp_path: Path) -> None:
    record = RolloutRecord(
        task_id="t1",
        arm="teacher",
        session_key="s",
        ledger_path="/l/sessions.db",
        track_sha256=TRACK,
        preset="chtd",
        route="teacher",
        model="qwen-large",
        outcome="succeeded",
        decisions=3,
    )
    path = write_records([record], tmp_path / "records.jsonl")
    assert read_records(path) == (record,)


def test_the_track_digest_is_read_from_the_file_the_gate_reads(tmp_path: Path) -> None:
    # Read rather than recomputed: a recomputed digest would be one no gate ever checks, and the
    # point of the digest is that it is the one the package refuses to load without.
    path = tmp_path / "chtd.sha256"
    path.write_text(TRACK + "\n", encoding="utf-8")
    assert read_track_digest(path) == TRACK


def test_the_real_track_digest_is_present_and_well_formed() -> None:
    # Against the repository's own recorded digest, so a typo in the path would fail here rather
    # than at rollout time.
    digest = read_track_digest()
    assert len(digest) == 64
    assert int(digest, 16) >= 0


def test_a_missing_track_digest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no recorded track digest"):
        read_track_digest(tmp_path / "nope.sha256")


def test_a_malformed_track_digest_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "chtd.sha256"
    path.write_text("not-a-digest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 hex digest"):
        read_track_digest(path)

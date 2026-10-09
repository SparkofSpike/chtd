"""A round as a dataset: what it includes, what it skips, and what it refuses.

`distill/rounds.py` is the step between a collection round and a trainable dataset, and it is where
two kinds of incompleteness are told apart: a run that failed in the world, which is data and should
be counted, and a run present in the records but missing its outcome, which is an inconsistency in
this repository and should stop the build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.rounds import RoundIncomplete, build_round_dataset
from harness.tasks.build_workspaces import FINAL_HOLDOUT, TASKS, split
from tests.ledger_fixture import build_ledger

STUDENT = "agnes:local:local-dev:cli:workspace:" + "a" * 16
TEACHER = "agnes:local:local-dev:cli:workspace:" + "b" * 16
TRACK = "1eb6b3ffaee3fb07228abebf4d3ce67112debb525c49e2dd0bc1eb363be0a97a"


def _session(path: Path, key: str, *, tool_call: bool) -> None:
    """A minimal session: a prompt, then one decision.

    Both sides start from the same conversation, which is what makes their first decision points
    share a prefix digest — the alignment this module depends on.
    """
    builder = build_ledger(path)
    builder.session(key).start(key, preset="minimal-rl", resolved_profile_hash="sha256-" + "c" * 64)
    builder.user(key, "Something in this directory is broken.")
    if tool_call:
        builder.append(key, "assistant/message", {"content": [], "stopReason": "tool_use"},
                       origin="model")
        builder.tool_call(key, "ls", {}, tool_use_id="t0-one")
    else:
        builder.assistant(key, "Hello! How can I help?")
    builder.close()


def _round(tmp_path: Path, *, tasks: list[str], omit: dict[str, list[str]] | None = None,
           no_outcome: dict[str, list[str]] | None = None) -> tuple[Path, Path]:
    """Write a round directory: records, checks, and one ledger per run."""
    omit = omit or {}
    no_outcome = no_outcome or {}
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    ledger_dir = tmp_path / "ledgers"
    ledger_dir.mkdir()

    records = []
    checks = []
    for task_id in tasks:
        for arm, key in (("student", STUDENT), ("teacher", TEACHER)):
            if arm in omit.get(task_id, []):
                continue
            ledger = ledger_dir / f"{task_id}-{arm}.db"
            _session(ledger, key, tool_call=(arm == "teacher"))
            records.append({
                "task_id": task_id,
                "arm": arm,
                "session_key": key,
                "ledger_path": str(ledger),
                "track_sha256": TRACK,
                "preset": "minimal-rl",
                "route": arm,
                "model": "m",
                "outcome": "unknown",
                "decisions": 1,
                "notes": "",
            })
            if arm not in no_outcome.get(task_id, []):
                checks.append({
                    "task_id": task_id,
                    "arm": arm,
                    "passed": arm == "teacher",
                    "output": "",
                    "workspace": "",
                })

    (round_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
    )
    (round_dir / "checks.jsonl").write_text(
        "\n".join(json.dumps(row) for row in checks) + "\n", encoding="utf-8"
    )
    return round_dir / "records.jsonl", round_dir / "checks.jsonl"


# ------------------------------------------------------------- the task split


def test_the_holdout_is_not_in_the_training_set() -> None:
    """The property that makes it a holdout. A caller who filters wrongly trains on the test set."""
    rows = [{"id": task["id"], "prompt": "p", "check": "c", "verify": "v", "expect": []}
            for task in TASKS]
    training, held_out = split(rows)
    assert {row["id"] for row in held_out} == set(FINAL_HOLDOUT)
    assert not ({row["id"] for row in training} & FINAL_HOLDOUT)
    assert len(training) + len(held_out) == len(TASKS)


def test_a_holdout_naming_nothing_is_refused() -> None:
    """A training set with no held-out tasks cannot be measured on anything it has not seen, and
    finding that out after a run is finding it out too late."""
    with pytest.raises(ValueError, match="names nothing"):
        split([])


def test_the_holdout_tasks_have_expected_phrases() -> None:
    """A held-out task still has to be judgeable, or the measurement it exists for cannot happen."""
    from harness.tasks.build_workspaces import EXPECT

    for task_id in FINAL_HOLDOUT:
        assert task_id in EXPECT, task_id
        assert EXPECT[task_id], task_id


# ------------------------------------------------------- building the dataset


def test_a_whole_round_becomes_one_dataset(tmp_path) -> None:
    records, checks = _round(tmp_path, tasks=["t1", "t2"])
    dataset, alignments, incomplete = build_round_dataset(
        records_path=records, checks_path=checks
    )
    assert len(alignments) == 2
    assert dataset.manifest is not None
    assert dataset.manifest.entries == 2, "one divergence per task, on the shared first state"
    assert dataset.manifest.refusals == 0
    assert incomplete == {}


def test_a_task_missing_a_side_is_skipped_not_fatal(tmp_path) -> None:
    """A run that failed in the world is data. Aborting the build would throw away every other
    task's data to punish one, and the round that prompted this check had exactly that shape: a
    teacher turn that ended blocked."""
    records, checks = _round(tmp_path, tasks=["t1", "t2"], omit={"t1": ["teacher"]})
    dataset, alignments, incomplete = build_round_dataset(
        records_path=records, checks_path=checks
    )
    assert set(alignments) == {"t2"}
    assert len(dataset.entries) == 1
    assert incomplete == {"t1": ["teacher"]}
    # And it is carried on the dataset, because every report is generated from these objects and a
    # limitation that only reached a console would not reach the write-up.
    assert dataset.reasons.get("task-missing-a-side") == 1


def test_a_record_without_an_outcome_stops_the_build(tmp_path) -> None:
    """The other kind of incompleteness, and the opposite response: a run present in the records but
    missing its check is this repository disagreeing with itself, not the world being untidy."""
    records, checks = _round(tmp_path, tasks=["t1"], no_outcome={"t1": ["student"]})
    with pytest.raises(RoundIncomplete, match="no observed outcome"):
        build_round_dataset(records_path=records, checks_path=checks)


def test_every_task_missing_a_side_is_refused(tmp_path) -> None:
    """Skipping everything leaves nothing, and a dataset with no rows looks exactly like a
    successful build."""
    records, checks = _round(tmp_path, tasks=["t1"], omit={"t1": ["teacher"]})
    with pytest.raises(RoundIncomplete, match="nothing to align"):
        build_round_dataset(records_path=records, checks_path=checks)


def test_a_failed_teacher_is_refused_by_the_gate(tmp_path) -> None:
    """The signal gate's job, seen from this module: a divergence where the teacher also failed has
    nothing to teach, so it is a refusal rather than a row."""
    records, checks = _round(tmp_path, tasks=["t1"])
    # Rewrite the checks so the teacher failed too.
    lines = [json.loads(line) for line in Path(checks).read_text(encoding="utf8").splitlines()]
    for row in lines:
        row["passed"] = False
    Path(checks).write_text("\n".join(json.dumps(row) for row in lines) + "\n", encoding="utf-8")

    dataset, alignments, _ = build_round_dataset(records_path=records, checks_path=checks)
    assert len(alignments) == 1
    assert dataset.manifest is not None
    assert dataset.manifest.entries == 0
    assert dataset.manifest.refusals == 1
    assert dataset.reasons.get("outcome") == 1


def test_no_checks_file_is_refused(tmp_path) -> None:
    records, _ = _round(tmp_path, tasks=["t1"])
    with pytest.raises(RoundIncomplete, match="no checks at"):
        build_round_dataset(records_path=records, checks_path=tmp_path / "absent.jsonl")

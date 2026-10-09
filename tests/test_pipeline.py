"""End to end, over real ledgers on disk: the chain the write-up tells a reader to rerun."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.pipeline import build_from_ledgers, main
from tests.ledger_fixture import build_ledger

TRACK = "a" * 64


def _rollouts(root: Path) -> Path:
    """One ledger holding the student's run and a fork of it, the way the harness records them.

    All of a deployment's sessions live in one ledger, and a fork's child is a row in it too — that
    is what makes the teacher's answer comparable to the student's decision rather than a fact from
    another file. Splitting them across ledgers would model something the harness never writes.

    The fork boundary is the state *before* the decision, so each fork answers exactly one of them.
    """
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "sessions.db"
    builder = build_ledger(ledger)
    builder.session("student").start("student")
    builder.user("student", "run the suite")
    # Decision 1 at seq 3, made in the state after seq 2.
    builder.tool_call("student", "shell", {"cmd": "pytest"}, tool_use_id="c1")
    builder.tool_result("student", "c1", text="3 failed")
    # Decision 2 at seq 5, made in the state after seq 4.
    builder.assistant("student", "the fixture is stale")

    # Fork A answers decision 1 the same way: an agreement, not signal.
    builder.fork_from("student", "teacher-a", 2, created_at="2026-10-01T00:01:00Z")
    builder.start("teacher-a", parent={"key": "student", "boundarySeq": 2})
    builder.tool_call("teacher-a", "shell", {"cmd": "pytest"}, tool_use_id="a1")

    # Fork B answers decision 1 differently: the divergence the dataset is built from.
    builder.fork_from("student", "teacher-b", 2, created_at="2026-10-01T00:02:00Z")
    builder.start("teacher-b", parent={"key": "student", "boundarySeq": 2})
    builder.tool_call("teacher-b", "shell", {"cmd": "pytest -x --lf"}, tool_use_id="b1")

    builder.close()
    return ledger


def _digest_of_first_state(ledger_path: Path, session_key: str) -> str:
    from harness.ledger import open_ledger
    from harness.trajectory import read_trajectory

    with open_ledger(ledger_path) as ledger:
        return read_trajectory(ledger, session_key).points[0].prefix_digest


def test_the_chain_runs_from_ledgers_to_a_written_dataset(tmp_path: Path) -> None:
    ledger = _rollouts(tmp_path / "rollouts")
    state = _digest_of_first_state(ledger, "student")

    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                # The teacher's branch succeeded; the student's failed. One admitted divergence.
                {"session_key": "teacher-b", "prefix_digest": state, "outcome": "succeeded"},
                {"session_key": "student", "prefix_digest": state, "outcome": "failed"},
            )
        ),
        encoding="utf-8",
    )

    dataset, alignment, student = build_from_ledgers(
        student_ledger=ledger,
        student_session="student",
        teacher_ledgers=[(ledger, "teacher-b")],
        task_id="t1",
        track_sha256=TRACK,
        outcomes_path=outcomes,
    )

    assert len(student.points) == 2
    assert len(alignment.divergences) == 1
    assert dataset.manifest is not None
    assert dataset.manifest.entries == 1
    assert dataset.manifest.track_sha256 == TRACK
    student_action = json.loads(dataset.entries[0].student_action)
    teacher_action = json.loads(dataset.entries[0].teacher_action)
    assert student_action == {"kind": "tool_call", "name": "shell", "payload": {"cmd": "pytest"}}
    assert teacher_action["payload"] == {"cmd": "pytest -x --lf"}


def test_a_fork_that_agrees_is_an_agreement_and_not_signal(tmp_path: Path) -> None:
    ledger = _rollouts(tmp_path / "rollouts")
    state = _digest_of_first_state(ledger, "student")
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"session_key": "teacher-a", "prefix_digest": state, "outcome": "succeeded"},
                {"session_key": "student", "prefix_digest": state, "outcome": "failed"},
            )
        ),
        encoding="utf-8",
    )
    dataset, alignment, _ = build_from_ledgers(
        student_ledger=ledger,
        student_session="student",
        teacher_ledgers=[(ledger, "teacher-a")],
        task_id="t1",
        track_sha256=TRACK,
        outcomes_path=outcomes,
    )
    assert alignment.divergences == ()
    assert len(alignment.agreements) == 1
    assert dataset.manifest is not None
    assert dataset.manifest.entries == 0


def test_an_ungated_build_is_refused(tmp_path: Path) -> None:
    # A gate with no validators admits everything, which is the behaviour it exists to prevent.
    ledger = _rollouts(tmp_path / "rollouts")
    with pytest.raises(SystemExit, match="refusing to build an ungated dataset"):
        build_from_ledgers(
            student_ledger=ledger,
            student_session="student",
            teacher_ledgers=[(ledger, "teacher-b")],
            task_id="t1",
            track_sha256=TRACK,
        )


def test_the_cli_writes_the_dataset_and_reports_what_it_did(tmp_path: Path, capsys) -> None:
    ledger = _rollouts(tmp_path / "rollouts")
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "prefix_digest": _digest_of_first_state(ledger, "student"),
                "admitted": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "dataset",
            "--student-ledger",
            str(ledger),
            "--student-session",
            "student",
            "--teacher",
            f"{ledger}:teacher-b",
            "--task",
            "t1",
            "--track",
            TRACK,
            "--out",
            str(tmp_path / "out"),
            "--verdicts",
            str(verdicts),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "admitted: 1" in printed
    assert "dataset digest:" in printed
    assert "aligned: 1 divergences" in printed

    entries = (tmp_path / "out" / "entries.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(entries) == 1
    row = json.loads(entries[0])
    assert set(row) == {
        "task_id",
        "prefix_digest",
        "student_session",
        "teacher_session",
        "student_seq",
        "teacher_seq",
        "student_action",
        "teacher_action",
        "teacher_action_fingerprint",
    }
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["track_sha256"] == TRACK


def test_the_same_inputs_produce_the_same_digest(tmp_path: Path) -> None:
    # The property the write-up promises: a reader reruns the recorded rollouts and reaches the
    # same numbers.
    ledger = _rollouts(tmp_path / "rollouts")
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {"prefix_digest": _digest_of_first_state(ledger, "student"), "admitted": True}
        )
        + "\n",
        encoding="utf-8",
    )
    kwargs = dict(
        student_ledger=ledger,
        student_session="student",
        teacher_ledgers=[(ledger, "teacher-b")],
        task_id="t1",
        track_sha256=TRACK,
        verdicts_path=verdicts,
    )
    first, _, _ = build_from_ledgers(**kwargs)
    second, _, _ = build_from_ledgers(**kwargs)
    assert first.manifest is not None and second.manifest is not None
    assert first.manifest.dataset_sha256 == second.manifest.dataset_sha256


def test_a_malformed_teacher_spec_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "dataset",
                "--student-ledger",
                str(tmp_path / "x.db"),
                "--student-session",
                "s",
                "--teacher",
                "no-colon-here",
                "--task",
                "t1",
                "--track",
                TRACK,
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_a_teacher_spec_survives_a_session_key_that_contains_colons(tmp_path: Path) -> None:
    """An Agnes session key is full of colons, and the split must not land inside it.

    `rpartition(":")` did exactly that: the ledger path absorbed
    `:agnes:local:local-dev:cli:workspace` and the command failed with "no ledger at
    <path>:agnes:local:...", naming the ledger when the argument format was at fault.
    The separator is found by requiring the left side to be a file, which a session key's
    prefix never is.
    """
    from distill.pipeline import _split_ledger_and_session

    ledger = tmp_path / "sessions.db"
    ledger.write_bytes(b"")
    key = "agnes:local:local-dev:cli:workspace:8ad1758e07cde6e0"

    assert _split_ledger_and_session(f"{ledger}:{key}") == (str(ledger), key)
    # A key without colons still splits, so the older form keeps working.
    assert _split_ledger_and_session(f"{ledger}:plain-key") == (str(ledger), "plain-key")
    # Nothing to split when the ledger is not there to find.
    assert _split_ledger_and_session(f"{tmp_path / 'missing.db'}:{key}") is None
    assert _split_ledger_and_session("no-colon-here") is None
    # A path with a drive letter and a colon of its own is not mistaken for a separator.
    assert _split_ledger_and_session(f"{ledger}:a:b:c") == (str(ledger), "a:b:c")

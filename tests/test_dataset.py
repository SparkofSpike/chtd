"""The dataset refuses to mix tracks, and refuses to carry gate evidence into a student's input."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from distill.dataset import (
    DatasetEntry,
    MixedTracksError,
    build_dataset,
    dataset_digest,
    mixed_tracks,
    write_dataset,
)
from distill.divergence import Divergence
from distill.gate import Outcome, OutcomeTable, OutcomeValidator, SignalGate
from harness.rollout import RolloutRecord
from harness.trajectory import Action, DecisionPoint

TRACK = "a" * 64
STATE = "d" * 64


def _divergence(state: str = STATE, *, student: str = "shell", teacher: str = "shell -x") -> Divergence:
    return Divergence(
        prefix_digest=state,
        student=DecisionPoint(1, 0, state, Action("tool_call", student, {"cmd": "pytest"})),
        teacher=DecisionPoint(2, 0, state, Action("tool_call", teacher, {"cmd": "pytest -x"})),
        student_session="student",
        teacher_session="teacher",
    )


class _Alignment:
    def __init__(self, divergences):
        self.divergences = tuple(divergences)


def _record(track: str = TRACK, arm: str = "student", task: str = "t1") -> RolloutRecord:
    return RolloutRecord(
        task_id=task,
        arm=arm,
        session_key="student",
        ledger_path="/l/sessions.db",
        track_sha256=track,
        preset="chtd",
        route="student",
        model="qwen-small",
    )


def _gate(*, teacher: Outcome = Outcome.SUCCEEDED, student: Outcome = Outcome.FAILED) -> SignalGate:
    return SignalGate(
        [
            OutcomeValidator(
                OutcomeTable({("teacher", STATE): teacher, ("student", STATE): student})
            )
        ]
    )


def test_mixing_two_tracks_is_refused() -> None:
    # Two tracks are two experiments. Merging them would produce a dataset measuring something
    # nobody chose, and nothing downstream could tell.
    with pytest.raises(MixedTracksError, match="refusing to build a dataset from more than one track"):
        build_dataset(
            task_ids=["t1"],
            alignments={"t1": _Alignment([_divergence()])},
            gate=_gate(),
            records=[_record(TRACK), _record("b" * 64)],
            arms=["student"],
        )


def test_one_track_builds() -> None:
    dataset = build_dataset(
        task_ids=["t1"],
        alignments={"t1": _Alignment([_divergence()])},
        gate=_gate(),
        records=[_record()],
        arms=["student", "teacher"],
        now=datetime(2026, 10, 9, tzinfo=timezone.utc),
    )
    assert dataset.manifest is not None
    assert dataset.manifest.track_sha256 == TRACK
    assert dataset.manifest.entries == 1
    assert dataset.manifest.created_at == "2026-10-09T00:00:00Z"
    assert dataset.manifest.arms == ("student", "teacher")


def test_no_records_means_the_dataset_cannot_say_what_track_it_came_from() -> None:
    with pytest.raises(ValueError, match="cannot say what track"):
        build_dataset(
            task_ids=["t1"], alignments={"t1": _Alignment([])}, gate=_gate(), records=[], arms=[]
        )


def test_a_gate_refusal_is_recorded_with_its_reason() -> None:
    # A dataset handed to a reviewer without refusals cannot be checked for whether it is small
    # because the method is selective or small because the gate rejected everything.
    dataset = build_dataset(
        task_ids=["t1"],
        alignments={"t1": _Alignment([_divergence()])},
        gate=_gate(teacher=Outcome.UNKNOWN),
        records=[_record()],
        arms=["student"],
    )
    assert dataset.entries == ()
    assert len(dataset.refusals) == 1
    assert "unvalidated" in dataset.refusals[0].reason
    assert dataset.reasons == {"outcome": 1}


def test_an_entry_carries_no_field_for_gate_evidence() -> None:
    # Privilege isolation, enforced structurally rather than promised. The teacher may see outcomes
    # and verifier results and they may decide admission, but a field that could hold them would
    # eventually hold them into the student's input.
    keys = set(json.loads(DatasetEntry.from_divergence("t1", _divergence()).to_json()))
    assert keys == {
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
    assert not {"outcome", "verifier", "reference", "privileged"} & keys


def test_an_entry_points_at_the_state_rather_than_copying_it() -> None:
    # One copy of the transcript, and every row traceable to the run it came from.
    entry = DatasetEntry.from_divergence("t1", _divergence())
    assert entry.prefix_digest == STATE
    assert entry.student_session == "student"
    assert entry.teacher_session == "teacher"
    assert entry.student_seq == 1 and entry.teacher_seq == 2


def test_the_digest_changes_when_a_row_changes() -> None:
    first = DatasetEntry.from_divergence("t1", _divergence())
    second = DatasetEntry.from_divergence("t1", _divergence(teacher="shell -x --lf"))
    assert dataset_digest([first]) != dataset_digest([second])
    assert dataset_digest([first]) == dataset_digest([first])


def test_the_digest_changes_when_rows_are_reordered() -> None:
    a = DatasetEntry.from_divergence("t1", _divergence("1" * 64))
    b = DatasetEntry.from_divergence("t1", _divergence("2" * 64))
    assert dataset_digest([a, b]) != dataset_digest([b, a])


def test_writing_puts_all_three_files_on_disk(tmp_path: Path) -> None:
    dataset = build_dataset(
        task_ids=["t1"],
        alignments={"t1": _Alignment([_divergence(), _divergence("e" * 64)])},
        gate=_gate(teacher=Outcome.UNKNOWN),
        records=[_record()],
        arms=["student"],
    )
    paths = write_dataset(dataset, tmp_path / "out")
    assert paths["entries"].read_text(encoding="utf-8") == ""
    assert len(paths["refusals"].read_text(encoding="utf-8").strip().splitlines()) == 2
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["track_sha256"] == TRACK
    assert manifest["refusals"] == 2


def test_mixed_tracks_reports_the_distinct_digests() -> None:
    assert mixed_tracks([_record(TRACK), _record(TRACK)]) == (TRACK,)
    assert mixed_tracks([_record(TRACK), _record("b" * 64)]) == tuple(sorted([TRACK, "b" * 64]))
    assert mixed_tracks([]) == ()

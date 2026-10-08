"""Alignment: what pairs, what differs, and what failed to meet."""

from __future__ import annotations

from pathlib import Path

import pytest

from distill.divergence import Divergence, align, align_all, divergence_rate
from harness.ledger import open_ledger
from harness.trajectory import Action, DecisionPoint, Trajectory
from tests.ledger_fixture import build_ledger


def _trajectories(tmp_path: Path):
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("student").start("student")
    builder.user("student", "run the suite")
    builder.tool_call("student", "shell", {"cmd": "pytest"}, tool_use_id="c1")
    builder.tool_result("student", "c1", text="3 failed")
    builder.assistant("student", "three tests fail")

    builder.fork_from("student", "teacher", 2)
    builder.start("teacher", parent={"key": "student", "boundarySeq": 2})
    builder.append("teacher", "budget.state", None)
    builder.tool_call("teacher", "shell", {"cmd": "pytest -x --lf"}, tool_use_id="t1")
    builder.tool_result("teacher", "t1", text="1 failed")
    builder.assistant("teacher", "that narrows it to one test")
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        from harness.trajectory import read_trajectory

        return read_trajectory(ledger, "student"), read_trajectory(ledger, "teacher")


def test_a_different_choice_in_the_same_state_is_a_divergence(tmp_path: Path) -> None:
    student, teacher = _trajectories(tmp_path)
    result = align(student, teacher)
    assert len(result.divergences) == 1
    divergence = result.divergences[0]
    assert divergence.student.action.name == "shell"
    assert divergence.student.action.payload == {"cmd": "pytest"}
    assert divergence.teacher.action.payload == {"cmd": "pytest -x --lf"}
    assert divergence.student_session == "student"
    assert divergence.teacher_session == "teacher"


def test_a_choice_the_teacher_did_not_answer_is_reported_separately(tmp_path: Path) -> None:
    # The student's final message stands in a state the teacher never reached. Counting it as
    # agreement would overstate coverage, which is how a thin dataset looks complete.
    student, teacher = _trajectories(tmp_path)
    result = align(student, teacher)
    assert len(result.unanswered) == 1
    assert result.unanswered[0].action.kind == "message"
    assert result.agreements == ()


def test_the_teachers_own_continuation_is_not_a_divergence(tmp_path: Path) -> None:
    # A fork's teacher goes on past the boundary in states the student never visited. Those are
    # facts about the teacher's run, not supervision the student needs — the student was never in
    # those states, so there is nothing to correct there.
    student, teacher = _trajectories(tmp_path)
    result = align(student, teacher)
    divergence_states = {d.prefix_digest for d in result.divergences}
    novel_states = {p.prefix_digest for p in result.novel}
    assert divergence_states.isdisjoint(novel_states)
    # The teacher's own tool result is conversation the student never saw, so the teacher's next
    # decision stands in a state the student has no counterpart for.
    assert novel_states, "a fork that continued past the boundary should have states of its own"


def test_agreement_is_counted_and_not_called_signal(tmp_path: Path) -> None:
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("student").start("student")
    builder.user("student", "run the suite")
    builder.tool_call("student", "shell", {"cmd": "pytest"}, tool_use_id="c1")
    builder.fork_from("student", "teacher", 2)
    builder.start("teacher", parent={"key": "student", "boundarySeq": 2})
    builder.tool_call("teacher", "shell", {"cmd": "pytest"}, tool_use_id="t1")
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        from harness.trajectory import read_trajectory

        result = align(read_trajectory(ledger, "student"), read_trajectory(ledger, "teacher"))
    assert result.divergences == ()
    assert len(result.agreements) == 1
    assert divergence_rate(result) == 0.0


def test_the_rate_is_over_answered_decisions_only(tmp_path: Path) -> None:
    # Against all decisions, a fork that covered half a run would look like a well-behaved student.
    student, teacher = _trajectories(tmp_path)
    result = align(student, teacher)
    assert result.student_decisions == 2
    assert divergence_rate(result) == 1.0  # the one answered decision diverged


def test_a_state_reached_in_two_runs_is_two_branches() -> None:
    # The same state in two runs is not the same branch, and an outcome table that keyed on the
    # state alone would let the student's failure condemn the teacher's choice.
    from distill.gate import Outcome, OutcomeTable

    table = OutcomeTable({("student", "d" * 64): Outcome.FAILED})
    assert table.get("student", "d" * 64) is Outcome.FAILED
    assert table.get("teacher", "d" * 64) is Outcome.UNKNOWN


def test_merging_forks_refuses_two_answers_for_one_state() -> None:
    # Two labels for one row: a dataset that picked one silently would be teaching a coin flip.
    point = DecisionPoint(seq=1, ordinal=0, prefix_digest="d" * 64, action=Action("tool_call", "a", {}))
    student = Trajectory("s", "main", "chtd", None, (), (point,), 1, ())
    first = Trajectory(
        "t1", "main", "chtd", None, (), (DecisionPoint(1, 0, "d" * 64, Action("tool_call", "x", {})),), 1, ()
    )
    second = Trajectory(
        "t2", "main", "chtd", None, (), (DecisionPoint(1, 0, "d" * 64, Action("tool_call", "y", {})),), 1, ()
    )
    with pytest.raises(ValueError, match="different answers"):
        align_all(student, [first, second])


def test_merging_forks_agrees_when_the_answers_agree() -> None:
    point = DecisionPoint(seq=1, ordinal=0, prefix_digest="d" * 64, action=Action("tool_call", "a", {}))
    student = Trajectory("s", "main", "chtd", None, (), (point,), 1, ())
    answer = DecisionPoint(1, 0, "d" * 64, Action("tool_call", "x", {}))
    first = Trajectory("t1", "main", "chtd", None, (), (answer,), 1, ())
    second = Trajectory("t2", "main", "chtd", None, (), (answer,), 1, ())
    merged = align_all(student, [first, second])
    assert len(merged.divergences) == 1


def test_a_divergence_describes_both_sides(tmp_path: Path) -> None:
    student, teacher = _trajectories(tmp_path)
    divergence = align(student, teacher).divergences[0]
    assert "shell" in divergence.describe()
    assert "->" in divergence.describe()


def test_an_empty_alignment_has_a_zero_rate() -> None:
    empty = Trajectory("s", "main", "chtd", None, (), (), 0, ())
    assert divergence_rate(align(empty, empty)) == 0.0

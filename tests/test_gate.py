"""The gate's job is to refuse, and to say why. Both halves are tested."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.divergence import Divergence
from distill.gate import (
    Outcome,
    OutcomeTable,
    OutcomeValidator,
    RecordedVerdicts,
    SignalGate,
    Verdict,
    WorkspaceRelativePaths,
)
from harness.trajectory import Action, DecisionPoint

_STATE = "d" * 64


def _divergence(digest: str = _STATE) -> Divergence:
    return Divergence(
        prefix_digest=digest,
        student=DecisionPoint(seq=3, ordinal=0, prefix_digest=digest, action=Action("tool_call", "shell", {"cmd": "pytest"})),
        teacher=DecisionPoint(seq=6, ordinal=0, prefix_digest=digest, action=Action("tool_call", "shell", {"cmd": "pytest -x"})),
        student_session="student",
        teacher_session="teacher",
    )


def _table(teacher: Outcome, student: Outcome) -> OutcomeTable:
    return OutcomeTable({("teacher", _STATE): teacher, ("student", _STATE): student})


def test_admits_when_the_teacher_succeeded_where_the_student_failed() -> None:
    verdict = OutcomeValidator(_table(Outcome.SUCCEEDED, Outcome.FAILED)).assess(_divergence())
    assert verdict.admitted


def test_refuses_when_both_succeeded() -> None:
    # A difference of style, not a failure to correct. Admitting it would teach the teacher's
    # preferences as if they were the capability.
    verdict = OutcomeValidator(_table(Outcome.SUCCEEDED, Outcome.SUCCEEDED)).assess(_divergence())
    assert not verdict.admitted
    assert "difference of style" in verdict.reason


def test_refuses_when_the_teachers_own_choice_failed() -> None:
    verdict = OutcomeValidator(_table(Outcome.FAILED, Outcome.FAILED)).assess(_divergence())
    assert not verdict.admitted
    assert "nothing here to teach" in verdict.reason


def test_an_unknown_outcome_refuses_rather_than_admits() -> None:
    # The two mistakes are not symmetric: a refused row loses signal the write-up can count, while
    # an admitted row that should not have been there is afterwards indistinguishable from one that
    # should have been.
    verdict = OutcomeValidator(_table(Outcome.UNKNOWN, Outcome.FAILED)).assess(_divergence())
    assert not verdict.admitted
    assert "teacher" in verdict.reason
    assert "unvalidated" in verdict.reason


def test_both_sides_missing_are_named() -> None:
    verdict = OutcomeValidator(OutcomeTable({})).assess(_divergence())
    assert not verdict.admitted
    assert "teacher and student" in verdict.reason


def test_outcomes_are_keyed_per_branch_not_per_state() -> None:
    # The same state in two runs is two branches. Keying on the state alone would let the student's
    # failure vouch for the teacher's choice.
    table = OutcomeTable({("student", _STATE): Outcome.FAILED})
    verdict = OutcomeValidator(table).assess(_divergence())
    assert not verdict.admitted
    assert "teacher" in verdict.reason


def test_a_gate_with_no_validators_is_refused() -> None:
    # It would admit everything, which is the behaviour the gate exists to prevent.
    with pytest.raises(ValueError, match="admits everything"):
        SignalGate([])


def test_an_empty_iterator_of_validators_is_refused_too() -> None:
    # `if not self.validators` is True for `[]` and False for `iter([])`, because every iterator is
    # truthy. A gate built from an empty iterator therefore held nothing and admitted everything.
    with pytest.raises(ValueError, match="admits everything"):
        SignalGate(iter([]))


def test_a_one_shot_iterator_is_materialised_so_the_gate_keeps_working() -> None:
    # A generator handed to this gate used to be consumed by the first `assess`; every call after
    # that admitted with zero verdicts. A gate that stops checking after one use is worse than no
    # gate, because it reports zero refusals.

    class Refuses:
        name = "always-refuses"

        def assess(self, divergence: Divergence) -> Verdict:
            return Verdict(False, "no", self.name)

    gate = SignalGate(validators=iter([Refuses()]))
    for attempt in range(3):
        admission = gate.assess(_divergence())
        assert admission.admitted is False, f"admitted on attempt {attempt}"
        assert len(admission.verdicts) == 1
        assert admission.reason == "always-refuses: no"


def test_a_table_holding_plain_strings_behaves_like_one_holding_members() -> None:
    # The validator identified outcomes with `is`, so a table holding plain strings — which is what
    # a rollout record carries — matched none of the three cases and fell through to the admitting
    # default. Every divergence was admitted, including the two the validator exists to refuse.
    for teacher, student, expected in (
        ("succeeded", "failed", True),
        ("succeeded", "succeeded", False),
        ("failed", "failed", False),
    ):
        table = OutcomeTable({("teacher", _STATE): teacher, ("student", _STATE): student})
        assert OutcomeValidator(table).assess(_divergence()).admitted is expected, (
            f"teacher={teacher} student={student}"
        )


def test_a_table_refuses_an_outcome_outside_the_vocabulary() -> None:
    # "SUCCESS" is a mis-spelling, not a fourth state. Folding it into UNKNOWN would send the reader
    # looking for a missing row instead of a wrong word.
    with pytest.raises(ValueError, match="is not an outcome"):
        OutcomeTable({("teacher", _STATE): "SUCCESS"})
    with pytest.raises(ValueError, match="is not an outcome"):
        OutcomeTable.from_records(
            [{"session_key": "t", "prefix_digest": _STATE, "outcome": "nonsense"}]
        )


def test_recording_coerces_so_a_later_write_cannot_reintroduce_a_string() -> None:
    table = OutcomeTable({})
    table.record("teacher", _STATE, "succeeded")  # type: ignore[arg-type]
    assert table.get("teacher", _STATE) is Outcome.SUCCEEDED


def test_a_non_boolean_verdict_is_refused() -> None:
    # `from_path` always checked this; the dataclass did not, so a hand-built instance holding the
    # string "false" admitted what it was meant to refuse — truthiness is not a verdict.
    for bad in ("false", "", 0, 1, None):
        with pytest.raises(ValueError, match="not a boolean"):
            RecordedVerdicts({_STATE: bad})  # type: ignore[dict-item]
    assert RecordedVerdicts({_STATE: False}).assess(_divergence()).admitted is False
    assert RecordedVerdicts({_STATE: True}).assess(_divergence()).admitted is True


def test_every_validator_must_admit() -> None:
    class Refuses:
        name = "always-refuses"

        def assess(self, divergence: Divergence) -> Verdict:
            return Verdict(False, "no", self.name)

    gate = SignalGate([OutcomeValidator(_table(Outcome.SUCCEEDED, Outcome.FAILED)), Refuses()])
    admission = gate.assess(_divergence())
    assert not admission.admitted
    assert admission.reason == "always-refuses: no"
    assert len(admission.verdicts) == 2


def test_recorded_verdicts_refuse_a_state_nobody_judged() -> None:
    # Fail-closed, so adding this validator can only ever narrow the admitted set.
    verdict = RecordedVerdicts({}).assess(_divergence())
    assert not verdict.admitted
    assert "no recorded judgement" in verdict.reason


def test_recorded_verdicts_carry_a_humans_decision(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    path.write_text(
        "\n".join(
            [
                "# a comment line, ignored",
                json.dumps({"prefix_digest": _STATE, "admitted": True}),
                json.dumps({"prefix_digest": "e" * 64, "admitted": False}),
            ]
        ),
        encoding="utf-8",
    )
    recorded = RecordedVerdicts.from_path(path)
    assert recorded.assess(_divergence()).admitted
    assert not recorded.assess(_divergence("e" * 64)).admitted
    assert not recorded.assess(_divergence("f" * 64)).admitted


def test_a_malformed_verdict_file_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    path.write_text('{"prefix_digest": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="needs a string prefix_digest and a boolean admitted"):
        RecordedVerdicts.from_path(path)


def test_a_verdict_file_that_is_not_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not JSON"):
        RecordedVerdicts.from_path(path)


def test_the_filter_reports_what_it_refused_and_why() -> None:
    gate = SignalGate([OutcomeValidator(_table(Outcome.SUCCEEDED, Outcome.FAILED))])
    admitted, refused = gate.filter([_divergence(), _divergence()])
    assert len(admitted) == 2
    assert refused == ()

    gate = SignalGate([OutcomeValidator(OutcomeTable({}))])
    admitted, refused = gate.filter([_divergence()])
    assert admitted == ()
    assert len(refused) == 1
    assert "unvalidated" in refused[0].reason


def test_a_state_with_no_outcome_is_refused_even_beside_an_admitted_one() -> None:
    # The table covers one state; the other is unknown and must not ride along with it.
    gate = SignalGate([OutcomeValidator(_table(Outcome.SUCCEEDED, Outcome.FAILED))])
    admitted, refused = gate.filter([_divergence(), _divergence("a" * 64)])
    assert len(admitted) == 1
    assert len(refused) == 1
    assert refused[0].divergence.prefix_digest == "a" * 64


def test_outcome_table_builds_from_rollout_records() -> None:
    table = OutcomeTable.from_records(
        [
            {"session_key": "s", "prefix_digest": _STATE, "outcome": "failed"},
            {"session_key": "t", "prefix_digest": _STATE, "outcome": "succeeded"},
        ]
    )
    assert table.get("s", _STATE) is Outcome.FAILED
    assert table.get("t", _STATE) is Outcome.SUCCEEDED


def test_a_malformed_outcome_record_is_refused() -> None:
    with pytest.raises(ValueError, match="session_key and prefix_digest"):
        OutcomeTable.from_records([{"outcome": "failed"}])


# ------------------------------------ paths a co-harness run cannot share


def _call(tool: str, args: object, digest: str = _STATE) -> Divergence:
    return Divergence(
        prefix_digest=digest,
        student=DecisionPoint(seq=3, ordinal=0, prefix_digest=digest,
                              action=Action("message", None, "hello")),
        teacher=DecisionPoint(seq=6, ordinal=0, prefix_digest=digest,
                              action=Action("tool_call", tool, args)),
        student_session="student",
        teacher_session="teacher",
    )


def test_a_relative_path_is_admitted() -> None:
    verdict = WorkspaceRelativePaths().assess(_call("read", {"path": "average.py"}))
    assert verdict.admitted


def test_an_absolute_path_is_refused() -> None:
    """The two arms run in different directories, so a path that is right for one is wrong for the
    other. The first dataset admitted three of these and the finished adapter learned the habit."""
    verdict = WorkspaceRelativePaths().assess(_call("read", {"path": "/answers.py"}))
    assert not verdict.admitted
    assert "absolute path" in verdict.reason


def test_a_drive_qualified_path_is_refused() -> None:
    verdict = WorkspaceRelativePaths().assess(_call("read", {"path": "E:/chtd/answers.py"}))
    assert not verdict.admitted
    assert "drive-qualified" in verdict.reason


def test_a_leading_backslash_is_refused() -> None:
    verdict = WorkspaceRelativePaths().assess(_call("ls", {"path": r"\server\share"}))
    assert not verdict.admitted


def test_a_bare_or_dotted_path_is_admitted() -> None:
    """`ls {}` is what fifteen of the nineteen admitted rows carry; `.` is the same idea written
    out."""
    assert WorkspaceRelativePaths().assess(_call("ls", {})).admitted
    assert WorkspaceRelativePaths().assess(_call("ls", {"path": "."})).admitted


def test_a_tool_that_takes_no_path_is_admitted() -> None:
    """A shell command may legitimately name an absolute interpreter. The rule reads one argument,
    on the tools whose first parameter is a workspace path."""
    assert WorkspaceRelativePaths().assess(_call("shell", {"command": "python /usr/bin/x"})).admitted
    assert WorkspaceRelativePaths().assess(_call("todo", {"items": []})).admitted


def test_a_message_action_is_admitted() -> None:
    """Only tool calls carry paths. A message that mentions one is prose, not an instruction to the
    filesystem, and three of the nineteen admitted rows are messages."""
    teacher = DecisionPoint(seq=6, ordinal=0, prefix_digest=_STATE,
                            action=Action("message", None, "the file /etc/hosts is not it"))
    divergence = Divergence(prefix_digest=_STATE, student=_call("read", {}).student,
                            teacher=teacher, student_session="student",
                            teacher_session="teacher")
    assert WorkspaceRelativePaths().assess(divergence).admitted


def test_a_malformed_payload_is_admitted_rather_than_crashing() -> None:
    """`assess` runs over every divergence in a dataset, so one that raised would take the build with
    it — and the shapes here come from a ledger that is pre-alpha."""
    assert WorkspaceRelativePaths().assess(_call("read", None)).admitted
    assert WorkspaceRelativePaths().assess(_call("read", "not a dict")).admitted
    assert WorkspaceRelativePaths().assess(_call("read", {"path": 17})).admitted
    assert WorkspaceRelativePaths().assess(_call("read", {"path": ""})).admitted

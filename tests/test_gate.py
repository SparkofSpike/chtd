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

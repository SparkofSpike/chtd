"""A co-mover may not be kept as evidence unless somebody wrote down why."""

from __future__ import annotations

import pytest

from eval.arms import BASELINE, CHTD, ArmConditions, compare
from eval.distractors import (
    CO_MOVER,
    DEFAULT_CANDIDATES,
    SEPARABLE,
    UNDETERMINED,
    DistractorDecision,
    ReasonRequired,
    analyse,
    kept_metrics,
    pearson,
    stripped_metrics,
)
from tests.metrics_fixture import FAILED, SUCCEEDED, run

TRACK = "a" * 64
RECIPE = "recipe" * 8
TASKS = tuple(f"t{index}" for index in range(6))

#: ``(baseline succeeded, CHTD succeeded, extra decisions the CHTD run took)`` per task.
#:
#: The capability has to *vary* for a correlation to exist at all — a constant column correlates
#: with nothing — so the two arms disagree on some tasks and agree on others. On the tasks where
#: CHTD gained, its runs are longer; that is the shape the design warns about.
GAIN_WITH_LONGER_RUNS = (
    (False, False, 0),
    (False, True, 4),
    (True, True, 0),
    (True, True, 0),
    (False, True, 4),
    (True, True, 0),
)

#: The same capability movement, with the run length held constant.
GAIN_WITH_EVEN_RUNS = tuple((base, chtd, 0) for base, chtd, _ in GAIN_WITH_LONGER_RUNS)


def _conditions(arm: str) -> ArmConditions:
    return ArmConditions(
        arm=arm, track_sha256=TRACK, tasks=TASKS, budget_steps=60, recipe_fingerprint=RECIPE
    )


def _comparison(rows) -> object:
    # Names chosen to be unalike on purpose: `cthd_rows` and `chtd_rows` differ by two swapped
    # letters, and a reader (and the author) cannot reliably tell them apart at a glance.
    control = []
    variant = []
    for task, (baseline_ok, chtd_ok, extra) in zip(TASKS, rows):
        control.append(run(task, BASELINE, outcome=SUCCEEDED if baseline_ok else FAILED, decisions=3))
        variant.append(run(task, CHTD, outcome=SUCCEEDED if chtd_ok else FAILED, decisions=3 + extra))
    return compare(
        tuple(control),
        tuple(variant),
        baseline_conditions=_conditions(BASELINE),
        chtd_conditions=_conditions(CHTD),
    )


def test_a_metric_that_moves_with_the_capability_is_flagged() -> None:
    findings = analyse(_comparison(GAIN_WITH_LONGER_RUNS))
    by_metric = {finding.metric: finding for finding in findings}
    assert by_metric["decisions"].correlation is not None
    assert by_metric["decisions"].correlation > 0.9
    assert by_metric["decisions"].verdict == CO_MOVER


def test_a_flagged_metric_is_stripped_by_default() -> None:
    findings = analyse(_comparison(GAIN_WITH_LONGER_RUNS))
    assert "decisions" in stripped_metrics(findings)
    assert "decisions" not in kept_metrics(findings)
    assert "stripped from the reported evidence" in {
        finding.metric: finding for finding in findings
    }["decisions"].reason


def test_keeping_a_co_mover_requires_a_reason() -> None:
    # Without a reason a reader cannot tell this number from the capability it moved with, and the
    # claim stops being falsifiable.
    with pytest.raises(ReasonRequired, match="requires a stated reason"):
        DistractorDecision(keep=True)
    with pytest.raises(ReasonRequired):
        DistractorDecision(keep=True, reason="   ")


def test_a_co_mover_can_be_kept_with_a_reason() -> None:
    findings = analyse(
        _comparison(GAIN_WITH_LONGER_RUNS),
        decisions={"decisions": DistractorDecision(keep=True, reason="steps are the capability here")},
    )
    decisions = {finding.metric: finding for finding in findings}
    assert decisions["decisions"].kept
    assert decisions["decisions"].reason == "steps are the capability here"
    assert "decisions" in kept_metrics(findings)


def test_a_metric_that_never_varies_has_no_relationship_to_report() -> None:
    # Undefined rather than zero: a candidate that took the same value on every task has no
    # relationship to report, and returning 0.0 would let it pass as "separable" on the strength of
    # a number that does not exist.
    findings = analyse(_comparison(GAIN_WITH_EVEN_RUNS))
    by_metric = {finding.metric: finding for finding in findings}
    assert by_metric["decisions"].correlation is None
    assert by_metric["decisions"].verdict == UNDETERMINED
    assert not by_metric["decisions"].kept
    assert "did not vary" in by_metric["decisions"].reason


def test_a_constant_capability_column_correlates_with_nothing() -> None:
    # Every task gains: the capability delta is all ones, so there is no variance to correlate
    # against and the honest answer is that this data cannot tell the two apart.
    rows = tuple((False, True, 4) for _ in TASKS)
    findings = analyse(_comparison(rows))
    by_metric = {finding.metric: finding for finding in findings}
    assert by_metric["decisions"].correlation is None
    assert by_metric["decisions"].verdict == UNDETERMINED


def test_a_candidate_with_no_column_is_undetermined_and_not_kept() -> None:
    findings = analyse(
        _comparison(GAIN_WITH_LONGER_RUNS), candidates=("decisions", "tokens_spent")
    )
    by_metric = {finding.metric: finding for finding in findings}
    assert by_metric["tokens_spent"].verdict == UNDETERMINED
    assert not by_metric["tokens_spent"].kept
    assert "no 'tokens_spent' column" in by_metric["tokens_spent"].reason


def test_an_undefined_correlation_is_not_reported_as_zero() -> None:
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson([1.0], [2.0]) is None
    assert pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_a_negative_co_mover_is_flagged_too() -> None:
    # A metric that falls as the capability rises is no more evidence for it than one that rises.
    rows = (
        (False, False, 0),
        (False, True, -4),
        (True, True, 0),
        (True, True, 0),
        (False, True, -4),
        (True, True, 0),
    )
    findings = analyse(_comparison(rows))
    by_metric = {finding.metric: finding for finding in findings}
    assert by_metric["decisions"].correlation is not None
    assert by_metric["decisions"].correlation < -0.9
    assert by_metric["decisions"].verdict == CO_MOVER


def test_pearson_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        pearson([1.0, 2.0], [1.0])


def test_every_finding_can_be_written_as_a_row() -> None:
    findings = analyse(_comparison(GAIN_WITH_LONGER_RUNS))
    assert findings
    for finding in findings:
        row = finding.to_row()
        assert set(row) == {"metric", "correlation", "paired_tasks", "verdict", "kept", "reason"}


def test_the_default_candidates_are_the_designs_list() -> None:
    assert DEFAULT_CANDIDATES == ("decisions", "tool_calls", "messages", "failed_calls", "events")
    assert len(DEFAULT_CANDIDATES) == len(set(DEFAULT_CANDIDATES))


def test_separable_is_reserved_for_a_measurable_non_relationship() -> None:
    # A metric that varies across tasks but not with the capability is separable; one that never
    # varied is undetermined. The two are different facts about the data.
    rows = (
        (False, False, 0),
        (False, True, 0),
        (True, True, 3),
        (True, True, 0),
        (False, True, 0),
        (True, True, 3),
    )
    findings = analyse(_comparison(rows))
    by_metric = {finding.metric: finding for finding in findings}
    assert by_metric["decisions"].correlation is not None
    assert by_metric["decisions"].verdict == SEPARABLE
    assert by_metric["decisions"].correlation < 0.8

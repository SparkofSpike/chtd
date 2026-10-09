"""The ladder's whole value is the verdict it reaches when the gain does not survive."""

from __future__ import annotations

import pytest

from eval.arms import BASELINE, CHTD, ArmConditions, ComparisonResult, compare
from eval.stripping import (
    DIFFERENT_PRESET,
    NO_GAIN,
    NO_TOOLS,
    RUNG_ORDER,
    SAME_TRACK,
    SCAFFOLD_OVERFIT,
    SINGLE_TURN,
    SURVIVED,
    RungResult,
    assess,
    describe,
)
from tests.metrics_fixture import FAILED, SUCCEEDED, run

TRACK = "a" * 64
RECIPE = "recipe" * 8
TASKS = ("t1", "t2", "t3", "t4")


def _conditions(arm: str) -> ArmConditions:
    return ArmConditions(
        arm=arm, track_sha256=TRACK, tasks=TASKS, budget_steps=60, recipe_fingerprint=RECIPE
    )


def _comparison(baseline_successes: int, chtd_successes: int, tasks: tuple[str, ...] = TASKS) -> ComparisonResult:
    """Two arms over ``tasks``.

    A zero on either side is ambiguous on its own, so the two callers below say which they mean: an
    arm can fail every task, or be undecided on every task. ``_undecided_comparison`` is the second.
    """
    return compare(
        tuple(
            run(task, BASELINE, outcome=SUCCEEDED if index < baseline_successes else FAILED)
            for index, task in enumerate(tasks)
        ),
        tuple(
            run(task, CHTD, outcome=SUCCEEDED if index < chtd_successes else FAILED)
            for index, task in enumerate(tasks)
        ),
        baseline_conditions=_conditions(BASELINE),
        chtd_conditions=_conditions(CHTD),
    )


def _undecided_comparison(tasks: tuple[str, ...] = TASKS) -> ComparisonResult:
    """Both arms unjudged: no decided pair, so no lift to report at all."""
    return compare(
        tuple(run(task, BASELINE) for task in tasks),
        tuple(run(task, CHTD) for task in tasks),
        baseline_conditions=_conditions(BASELINE),
        chtd_conditions=_conditions(CHTD),
    )


def test_a_gain_that_holds_everywhere_survives() -> None:
    verdict = assess(
        {
            SAME_TRACK: _comparison(0, 4),
            DIFFERENT_PRESET: _comparison(0, 4),
            NO_TOOLS: _comparison(0, 3),
            SINGLE_TURN: _comparison(0, 2),
        }
    )
    assert verdict.verdict == SURVIVED
    assert verdict.same_track_lift == 1.0


def test_a_gain_that_holds_only_on_the_training_track_is_scaffold_overfit() -> None:
    # The result the design says must be reported as such, not as a capability.
    verdict = assess(
        {
            SAME_TRACK: _comparison(0, 4),
            DIFFERENT_PRESET: _comparison(0, 0),
            NO_TOOLS: _comparison(0, 0),
        }
    )
    assert verdict.verdict == SCAFFOLD_OVERFIT
    assert "scaffold overfit" in verdict.reason
    assert "do not survive" in verdict.reason or "does not survive" in verdict.reason


def test_no_same_track_gain_means_there_is_nothing_to_explain() -> None:
    verdict = assess({SAME_TRACK: _comparison(2, 2), DIFFERENT_PRESET: _comparison(0, 4)})
    assert verdict.verdict == NO_GAIN
    assert "no gain to explain" in verdict.reason


def test_a_ladder_with_no_stripped_rung_cannot_separate_capability_from_fit() -> None:
    # Not "survived": nothing ran that could have falsified it.
    verdict = assess({SAME_TRACK: _comparison(0, 4)})
    assert verdict.verdict == SCAFFOLD_OVERFIT
    assert "no stripped rung was run" in verdict.reason


def test_the_threshold_is_read_from_the_verdict_rather_than_assumed() -> None:
    # A parameter, because where a gain stops counting as surviving is an argument about the
    # experiment and not a fact about the arithmetic.
    rungs = {SAME_TRACK: _comparison(0, 4), NO_TOOLS: _comparison(0, 1)}
    strict = assess(rungs, survival_threshold=0.5)
    lenient = assess(rungs, survival_threshold=0.2)
    assert strict.verdict == SCAFFOLD_OVERFIT
    assert lenient.verdict == SURVIVED


def test_a_ladder_needs_its_primary_measurement() -> None:
    with pytest.raises(ValueError, match="no primary measurement"):
        assess({NO_TOOLS: _comparison(0, 4)})


def test_an_unknown_rung_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown rung"):
        assess({SAME_TRACK: _comparison(0, 4), "with-subagents": _comparison(0, 4)})


def test_rungs_are_reported_in_ladder_order() -> None:
    # Read as a ladder: each rung removes something, and a rung out of order would report a
    # survival that never happened.
    verdict = assess(
        {
            SINGLE_TURN: _comparison(0, 4),
            SAME_TRACK: _comparison(0, 4),
            NO_TOOLS: _comparison(0, 4),
            DIFFERENT_PRESET: _comparison(0, 4),
        }
    )
    assert [rung.rung for rung in verdict.rungs] == list(RUNG_ORDER)


def test_the_weakest_rung_is_what_the_verdict_turns_on() -> None:
    verdict = assess(
        {
            SAME_TRACK: _comparison(0, 4),
            DIFFERENT_PRESET: _comparison(0, 4),
            NO_TOOLS: _comparison(0, 4),
            SINGLE_TURN: _comparison(0, 0),
        }
    )
    assert verdict.verdict == SCAFFOLD_OVERFIT
    assert SINGLE_TURN in verdict.reason


def test_a_rung_can_be_looked_up_by_name() -> None:
    verdict = assess({SAME_TRACK: _comparison(0, 4), NO_TOOLS: _undecided_comparison()})
    assert verdict.rung(NO_TOOLS).lift == 0.0
    with pytest.raises(KeyError):
        verdict.rung(SINGLE_TURN)


def test_describe_lists_every_rung() -> None:
    text = describe(assess({SAME_TRACK: _comparison(0, 4), NO_TOOLS: _comparison(0, 1)}))
    assert SCAFFOLD_OVERFIT in text
    assert SAME_TRACK in text
    assert NO_TOOLS in text


def test_a_rung_that_decided_nothing_is_not_a_measurement() -> None:
    # A rung whose tasks all went undecided has lift 0.0 — a number that looks like a result. Read
    # as "no gain here", it turned an unfinished ladder into a positive claim of overfitting.
    from eval.stripping import UNDETERMINED

    no_data = _undecided_comparison()
    verdict = assess({SAME_TRACK: _comparison(0, 4), NO_TOOLS: no_data})
    assert verdict.verdict == UNDETERMINED
    assert "decided no tasks" in verdict.reason
    assert not verdict.rung(NO_TOOLS).measured
    assert verdict.rung(SAME_TRACK).measured


def test_a_primary_rung_with_no_data_is_undetermined_not_no_gain() -> None:
    from eval.stripping import UNDETERMINED

    verdict = assess({SAME_TRACK: _undecided_comparison()})
    assert verdict.verdict == UNDETERMINED
    assert "absence of data" in verdict.reason


def test_the_survival_threshold_must_be_a_fraction() -> None:
    # At 0 a rung that measured nothing would count as surviving; above 1 nothing could ever.
    for bad in (0.0, -0.5, 1.5, 2.0):
        with pytest.raises(ValueError, match="survival_threshold must be in"):
            assess({SAME_TRACK: _comparison(0, 4), NO_TOOLS: _comparison(0, 1)}, survival_threshold=bad)


def test_describe_says_when_a_rung_decided_nothing() -> None:
    text = describe(assess({SAME_TRACK: _comparison(0, 4), NO_TOOLS: _undecided_comparison()}))
    assert "(no tasks decided)" in text


def test_a_rung_result_carries_its_task_count() -> None:
    rung = RungResult(rung=SAME_TRACK, comparison=_comparison(0, 4))
    assert rung.tasks == 4

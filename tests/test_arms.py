"""The comparison refuses to compare, which is most of its value."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.arms import (
    BASELINE,
    CHTD,
    ArmConditions,
    IncomparableArms,
    compare,
    conditions_from_records,
    summarise,
)
from eval.metrics import UNKNOWN_OUTCOME, measure, measure_all
from harness.ledger import open_ledger
from harness.trajectory import read_trajectory
from tests.ledger_fixture import build_ledger
from tests.metrics_fixture import FAILED, SUCCEEDED, run

TRACK = "a" * 64
RECIPE = "recipe" * 8


def _conditions(arm: str, **over: object) -> ArmConditions:
    base = {
        "arm": arm,
        "track_sha256": TRACK,
        "tasks": ("t1", "t2"),
        "budget_steps": 60,
        "recipe_fingerprint": RECIPE,
    }
    base.update(over)
    return ArmConditions(**base)  # type: ignore[arg-type]


def _rows(arm: str, outcomes: dict[str, str]) -> tuple:
    return tuple(run(task, arm, outcome=outcome) for task, outcome in outcomes.items())


def test_comparing_arms_on_different_tracks_is_refused() -> None:
    with pytest.raises(IncomparableArms, match="tracks differ"):
        compare(
            _rows(BASELINE, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            _rows(CHTD, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            baseline_conditions=_conditions(BASELINE),
            chtd_conditions=_conditions(CHTD, track_sha256="b" * 64),
        )


def test_comparing_arms_over_different_task_sets_is_refused() -> None:
    # A comparison over whatever happened to overlap is how an inconvenient task disappears.
    with pytest.raises(IncomparableArms, match="task sets differ"):
        compare(
            _rows(BASELINE, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            _rows(CHTD, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            baseline_conditions=_conditions(BASELINE),
            chtd_conditions=_conditions(CHTD, tasks=("t1",)),
        )


def test_comparing_arms_with_different_budgets_is_refused() -> None:
    with pytest.raises(IncomparableArms, match="step budgets differ"):
        compare(
            _rows(BASELINE, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            _rows(CHTD, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            baseline_conditions=_conditions(BASELINE),
            chtd_conditions=_conditions(CHTD, budget_steps=100),
        )


def test_comparing_arms_with_different_recipes_is_refused() -> None:
    with pytest.raises(IncomparableArms, match="recipes differ"):
        compare(
            _rows(BASELINE, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            _rows(CHTD, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            baseline_conditions=_conditions(BASELINE),
            chtd_conditions=_conditions(CHTD, recipe_fingerprint="other" * 8),
        )


def test_a_well_formed_comparison_reports_its_lift() -> None:
    result = compare(
        _rows(BASELINE, {"t1": FAILED, "t2": FAILED}),
        _rows(CHTD, {"t1": SUCCEEDED, "t2": FAILED}),
        baseline_conditions=_conditions(BASELINE),
        chtd_conditions=_conditions(CHTD),
    )
    assert result.baseline_success_rate == 0.0
    assert result.chtd_success_rate == 0.5
    assert result.lift == 0.5
    assert result.summary()["gained"] == 1
    assert "lift +0.500" in summarise(result)


def test_an_undecided_task_is_reported_and_not_counted() -> None:
    # An unjudged run and a failed run are different facts. Folding them together would understate
    # failures while overstating coverage.
    result = compare(
        _rows(BASELINE, {"t1": FAILED, "t2": UNKNOWN_OUTCOME}),
        _rows(CHTD, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
        baseline_conditions=_conditions(BASELINE),
        chtd_conditions=_conditions(CHTD),
    )
    assert len(result.pairs) == 2
    assert len(result.undecided) == 1
    assert len(result.decided_pairs) == 1
    assert result.lift == 1.0
    assert result.summary()["tasks_undecided"] == 1


def test_a_task_missing_from_an_arm_is_named() -> None:
    with pytest.raises(ValueError, match="missing from the CHTD arm"):
        compare(
            _rows(BASELINE, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            (run("t1", CHTD, outcome=SUCCEEDED),),
            baseline_conditions=_conditions(BASELINE),
            chtd_conditions=_conditions(CHTD),
        )


def test_conditions_must_name_the_arms_they_describe() -> None:
    with pytest.raises(ValueError, match="baseline conditions name arm"):
        compare(
            _rows(BASELINE, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            _rows(CHTD, {"t1": SUCCEEDED, "t2": SUCCEEDED}),
            baseline_conditions=_conditions(CHTD),
            chtd_conditions=_conditions(CHTD),
        )


def test_conditions_come_from_the_records_not_from_the_caller() -> None:
    # A caller-supplied track digest would let a comparison be declared comparable while the
    # records said otherwise.
    from harness.rollout import RolloutRecord

    def record(task: str, arm: str, track: str = TRACK) -> RolloutRecord:
        return RolloutRecord(
            task_id=task,
            arm=arm,
            session_key="s",
            ledger_path="/l",
            track_sha256=track,
            preset="chtd",
            route="r",
            model="m",
        )

    conditions = conditions_from_records(
        [record("t1", BASELINE), record("t2", BASELINE)],
        arm=BASELINE,
        budget_steps=60,
        recipe_fingerprint=RECIPE,
    )
    assert conditions.tasks == ("t1", "t2")
    assert conditions.track_sha256 == TRACK

    with pytest.raises(IncomparableArms, match="distinct tracks"):
        conditions_from_records(
            [record("t1", BASELINE), record("t2", BASELINE, "b" * 64)],
            arm=BASELINE,
            budget_steps=60,
            recipe_fingerprint=RECIPE,
        )


# -- metrics ------------------------------------------------------------------------------------


def _trajectory(tmp_path: Path):
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s").start("s")
    builder.user("s", "go")
    builder.tool_call("s", "read", {"path": "a"}, tool_use_id="c1")
    builder.tool_result("s", "c1", is_error=False)
    builder.tool_call("s", "shell", {"cmd": "x"}, tool_use_id="c2")
    builder.tool_result("s", "c2", is_error=True)
    builder.assistant("s", "done")
    builder.close()
    with open_ledger(tmp_path / "sessions.db") as ledger:
        return read_trajectory(ledger, "s")


def test_measure_reads_counts_off_the_trajectory(tmp_path: Path) -> None:
    metrics = measure(_trajectory(tmp_path), task_id="t1", arm=CHTD, outcome=SUCCEEDED)
    assert metrics.decisions == 3
    assert metrics.tool_calls == 2
    assert metrics.messages == 1
    assert metrics.failed_calls == 1
    assert metrics.succeeded
    assert metrics.decided


def test_an_unjudged_run_is_not_a_failure(tmp_path: Path) -> None:
    metrics = measure(_trajectory(tmp_path), task_id="t1", arm=CHTD)
    assert not metrics.succeeded
    assert not metrics.decided


def test_a_run_that_ended_before_its_result_is_not_counted_as_failed(tmp_path: Path) -> None:
    # `errored is None` is missing information, not a failure. Counting it as one would make a
    # truncated run look like a bad one.
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s").start("s")
    builder.user("s", "go")
    builder.tool_call("s", "read", {"path": "a"}, tool_use_id="c1")
    builder.close()
    with open_ledger(tmp_path / "sessions.db") as ledger:
        metrics = measure(read_trajectory(ledger, "s"), task_id="t1", arm=CHTD)
    assert metrics.tool_calls == 1
    assert metrics.failed_calls == 0


def test_measure_all_selects_the_arm_it_was_asked_for(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path)
    report = measure_all(
        {"chtd/t1": trajectory, "baseline/t1": trajectory},
        arm=CHTD,
        outcomes={"t1": SUCCEEDED},
    )
    assert [row.arm for row in report.rows] == [CHTD]
    assert report.tasks() == ("t1",)
    assert report.rows[0].succeeded

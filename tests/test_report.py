"""A report that omitted a refuted rung or a stripped number would describe an experiment nobody ran."""

from __future__ import annotations

from pathlib import Path

import pytest

from distill.dataset import DatasetManifest
from eval.arms import BASELINE, CHTD, ArmConditions, compare
from eval.distractors import DistractorDecision, analyse
from eval.report import STANDING_LIMITATIONS, build, write
from eval.stripping import DIFFERENT_PRESET, NO_TOOLS, SAME_TRACK, assess
from tests.metrics_fixture import FAILED, SUCCEEDED, run

TRACK = "a" * 64
RECIPE = "recipe" * 8
TASKS = tuple(f"t{index}" for index in range(6))
ROWS = (
    (False, False, 0),
    (False, True, 4),
    (True, True, 0),
    (True, True, 0),
    (False, True, 4),
    (True, True, 0),
)


def _conditions(arm: str) -> ArmConditions:
    return ArmConditions(
        arm=arm, track_sha256=TRACK, tasks=TASKS, budget_steps=60, recipe_fingerprint=RECIPE
    )


def _comparison(rows=ROWS):
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


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        track_sha256=TRACK,
        dataset_sha256="b" * 64,
        entries=2,
        refusals=1,
        tasks=TASKS,
        arms=(BASELINE, CHTD),
        created_at="2026-10-09T00:00:00Z",
    )


def _report(**over):
    comparison = over.get("comparison") or _comparison()
    ladder = over.get("ladder", assess({SAME_TRACK: _comparison(), NO_TOOLS: _comparison()}))
    distractors = over.get("distractors", analyse(_comparison()))
    return build(
        title="CHTD result",
        track_sha256=TRACK,
        conditions=[_conditions(BASELINE), _conditions(CHTD)],
        comparison=comparison,
        ladder=ladder,
        distractors=distractors,
        dataset=over.get("dataset", _manifest()),
        judge=over.get("judge", "a human reading each trajectory"),
    )


def test_the_report_carries_every_section() -> None:
    text = _report().render()
    for heading in (
        "# CHTD result",
        "## Conditions",
        "## The deciding comparison",
        "## Harness-stripping ladder",
        "## Distractors",
        "## Dataset",
        "## Limitations",
        "## Reproducing this",
    ):
        assert heading in text


def test_a_refuted_ladder_is_reported_as_scaffold_overfit() -> None:
    # The design says it must be reported as such, not as a capability, so the report has to say it.
    # Here the stripped rung shows no gain at all: the training-track lift does not survive.
    overfit = assess({SAME_TRACK: _comparison(), NO_TOOLS: _comparison(tuple((b, b, 0) for b, _, _ in ROWS))})
    text = _report(ladder=overfit).render()
    assert "scaffold-overfit" in text
    assert "scaffold overfit" in text


def test_a_survived_ladder_is_reported_as_survived() -> None:
    ladder = assess({SAME_TRACK: _comparison(), NO_TOOLS: _comparison()})
    assert "survived" in _report(ladder=ladder).render()


def test_a_stripped_distractor_is_named_in_the_report() -> None:
    text = _report(distractors=analyse(_comparison(), candidates=("decisions",))).render()
    assert "Stripped from the evidence" in text
    assert "decisions" in text


def test_a_co_mover_kept_with_a_reason_reads_as_kept() -> None:
    findings = analyse(
        _comparison(), decisions={"decisions": DistractorDecision(True, "why it is causal")}
    )
    text = _report(distractors=findings).render()
    # The reason has to be visible, or the exception is invisible and the rule is pointless.
    assert "Kept as evidence, with the reason it is causal:" in text
    assert "**decisions** — why it is causal" in text
    # And it must not also be listed among the stripped ones. With `decisions` kept there may be no
    # stripped line at all, and a line that is absent cannot be contradicted.
    stripped_lines = [line for line in text.splitlines() if line.startswith("Stripped from")]
    for line in stripped_lines:
        assert "decisions" not in line


def test_an_undetermined_metric_is_not_reported_as_a_stripped_co_mover() -> None:
    # "Could not be judged" is not "was found wanting". Printing it under the stripping sentence
    # claims a finding the analysis never reached.
    findings = analyse(_comparison(), candidates=("decisions", "tokens_spent"))
    text = _report(distractors=findings).render()
    stripped_lines = [line for line in text.splitlines() if line.startswith("Stripped from")]
    for line in stripped_lines:
        assert "tokens_spent" not in line
    assert "Not determined by this data" in text
    assert "tokens_spent" in next(
        line for line in text.splitlines() if line.startswith("Not determined")
    )


def test_the_standing_limitations_are_always_there() -> None:
    text = _report().render()
    for limitation in STANDING_LIMITATIONS:
        assert limitation.split(".")[0][:40] in text


def test_extra_limitations_are_added_not_substituted() -> None:
    report = build(
        title="t",
        track_sha256=TRACK,
        conditions=[_conditions(BASELINE), _conditions(CHTD)],
        comparison=_comparison(),
        ladder=None,
        distractors=(),
        dataset=None,
        judge="j",
        extra_limitations=("The task set was small.",),
    )
    assert "The task set was small." in report.render()
    assert len(report.limitations()) == len(STANDING_LIMITATIONS) + 1


def test_a_missing_ladder_is_stated_rather_than_left_blank() -> None:
    text = _report(ladder=None).render()
    assert "Not run" in text
    assert "unseparated" in text


def test_a_missing_dataset_is_stated() -> None:
    text = _report(dataset=None).render()
    assert "No dataset manifest attached" in text


def test_a_report_with_no_conditions_is_refused() -> None:
    with pytest.raises(ValueError, match="no conditions"):
        build(
            title="t",
            track_sha256=TRACK,
            conditions=[],
            comparison=_comparison(),
            ladder=None,
            distractors=(),
            dataset=None,
            judge="j",
        )


def test_the_report_writes_to_disk(tmp_path: Path) -> None:
    path = write(_report(), tmp_path / "out" / "report.md")
    assert path.read_text(encoding="utf-8").startswith("# CHTD result")

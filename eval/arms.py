"""The comparison that decides whether co-harness earns its name.

The question is not "did the student improve" but "did sharing the harness cause the improvement".
Answering it needs two arms whose conditions are identical, so this module's first job is to refuse
a comparison whose arms were not run under the same conditions. The design names them: same track,
same tasks, same budget, same LoRA recipe. The baseline arm draws its teacher data from the
teacher's own trajectory; the CHTD arm draws it from the student's states. That is the only
permitted difference, and it is checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from eval.metrics import RunMetrics

__all__ = [
    "ArmComparison",
    "ArmConditions",
    "ComparisonResult",
    "IncomparableArms",
    "compare",
    "summarise",
]

BASELINE = "baseline"
CHTD = "chtd"


class IncomparableArms(ValueError):
    """The two arms were not run under the same conditions, so any difference between them is
    unattributable."""


@dataclass(frozen=True)
class ArmConditions:
    """Everything that must be equal across the arms for the comparison to mean anything."""

    arm: str
    track_sha256: str
    tasks: tuple[str, ...]
    budget_steps: int | None
    recipe_fingerprint: str


def _assert_comparable(baseline: ArmConditions, chtd: ArmConditions) -> None:
    problems: list[str] = []
    if baseline.track_sha256 != chtd.track_sha256:
        problems.append(
            f"tracks differ ({baseline.track_sha256[:12]} vs {chtd.track_sha256[:12]})"
        )
    if baseline.tasks != chtd.tasks:
        only_baseline = set(baseline.tasks) - set(chtd.tasks)
        only_chtd = set(chtd.tasks) - set(baseline.tasks)
        problems.append(
            f"task sets differ (baseline only: {sorted(only_baseline)}, chtd only: {sorted(only_chtd)})"
        )
    if baseline.budget_steps != chtd.budget_steps:
        problems.append(f"step budgets differ ({baseline.budget_steps} vs {chtd.budget_steps})")
    if baseline.recipe_fingerprint != chtd.recipe_fingerprint:
        problems.append("LoRA recipes differ; comparing arms would compare the recipes")
    if problems:
        raise IncomparableArms(
            "refusing to compare arms that were not run identically: " + "; ".join(problems)
        )


@dataclass(frozen=True)
class ArmComparison:
    """One task, both arms."""

    task_id: str
    baseline: RunMetrics
    chtd: RunMetrics

    @property
    def both_decided(self) -> bool:
        return self.baseline.decided and self.chtd.decided

    @property
    def gained(self) -> bool:
        return self.chtd.succeeded and not self.baseline.succeeded

    @property
    def lost(self) -> bool:
        return self.baseline.succeeded and not self.chtd.succeeded

    @property
    def delta_decisions(self) -> int:
        return self.chtd.decisions - self.baseline.decisions

    @property
    def delta_failed_calls(self) -> int:
        return self.chtd.failed_calls - self.baseline.failed_calls


@dataclass(frozen=True)
class ComparisonResult:
    """Every task the two arms share, paired. Filtering is a read, not a second collection."""

    pairs: tuple[ArmComparison, ...]

    @property
    def decided_pairs(self) -> tuple[ArmComparison, ...]:
        return tuple(pair for pair in self.pairs if pair.both_decided)

    @property
    def undecided(self) -> tuple[ArmComparison, ...]:
        return tuple(pair for pair in self.pairs if not pair.both_decided)

    @property
    def baseline_success_rate(self) -> float:
        decided = self.decided_pairs
        if not decided:
            return 0.0
        return sum(1 for pair in decided if pair.baseline.succeeded) / len(decided)

    @property
    def chtd_success_rate(self) -> float:
        decided = self.decided_pairs
        if not decided:
            return 0.0
        return sum(1 for pair in decided if pair.chtd.succeeded) / len(decided)

    @property
    def lift(self) -> float:
        return self.chtd_success_rate - self.baseline_success_rate

    def summary(self) -> dict[str, object]:
        decided = self.decided_pairs
        return {
            "tasks_compared": len(decided),
            "tasks_undecided": len(self.undecided),
            "baseline_success_rate": round(self.baseline_success_rate, 6),
            "chtd_success_rate": round(self.chtd_success_rate, 6),
            "lift": round(self.lift, 6),
            "gained": sum(1 for pair in decided if pair.gained),
            "lost": sum(1 for pair in decided if pair.lost),
            "mean_delta_decisions": (
                round(sum(pair.delta_decisions for pair in decided) / len(decided), 3)
                if decided
                else 0.0
            ),
            "mean_delta_failed_calls": (
                round(sum(pair.delta_failed_calls for pair in decided) / len(decided), 3)
                if decided
                else 0.0
            ),
        }


def compare(
    baseline_rows: Iterable[RunMetrics],
    chtd_rows: Iterable[RunMetrics],
    *,
    baseline_conditions: ArmConditions,
    chtd_conditions: ArmConditions,
) -> ComparisonResult:
    """Pair the arms by task and compare, after refusing mismatched conditions.

    A task present in one arm and not the other is not dropped silently: the conditions check fails
    first, because a comparison over whatever happened to overlap is how an inconvenient task
    disappears from a result.
    """
    if baseline_conditions.arm != BASELINE:
        raise ValueError(f"the baseline conditions name arm {baseline_conditions.arm!r}")
    if chtd_conditions.arm != CHTD:
        raise ValueError(f"the CHTD conditions name arm {chtd_conditions.arm!r}")
    _assert_comparable(baseline_conditions, chtd_conditions)
    baseline_rows = tuple(baseline_rows)
    chtd_rows = tuple(chtd_rows)
    for label, rows, expected_arm, expected_tasks in (
        ("baseline", baseline_rows, BASELINE, set(baseline_conditions.tasks)),
        ("CHTD", chtd_rows, CHTD, set(chtd_conditions.tasks)),
    ):
        task_ids = [row.task_id for row in rows]
        duplicates = sorted({task for task in task_ids if task_ids.count(task) > 1})
        if duplicates:
            raise ValueError(f"{label} rows contain duplicate tasks: {duplicates}")
        wrong_arm = sorted({row.arm for row in rows if row.arm != expected_arm})
        if wrong_arm:
            raise ValueError(f"{label} rows contain unexpected arm labels: {wrong_arm}")
        unexpected = sorted(set(task_ids) - expected_tasks)
        if unexpected:
            raise ValueError(f"{label} rows contain tasks outside conditions: {unexpected}")

    by_task_baseline = {row.task_id: row for row in baseline_rows}
    by_task_chtd = {row.task_id: row for row in chtd_rows}
    pairs: list[ArmComparison] = []
    for task_id in baseline_conditions.tasks:
        baseline = by_task_baseline.get(task_id)
        chtd = by_task_chtd.get(task_id)
        if baseline is None or chtd is None:
            raise ValueError(
                f"task {task_id} is in the conditions but missing from "
                f"{'the baseline arm' if baseline is None else 'the CHTD arm'}"
            )
        pairs.append(ArmComparison(task_id=task_id, baseline=baseline, chtd=chtd))

    return ComparisonResult(pairs=tuple(pairs))


def summarise(result: ComparisonResult, *, label: str = "same track") -> str:
    """One line a reader can put in a report without re-deriving the arithmetic."""
    data = result.summary()
    return (
        f"{label}: {data['tasks_compared']} tasks compared, "
        f"baseline {data['baseline_success_rate']:.3f} -> CHTD {data['chtd_success_rate']:.3f} "
        f"(lift {data['lift']:+.3f}); gained {data['gained']}, lost {data['lost']}"
    )


def conditions_from_records(
    records: Sequence[object], *, arm: str, budget_steps: int | None, recipe_fingerprint: str
) -> ArmConditions:
    """Build conditions from rollout records, taking the track from them rather than from a caller.

    The track digest has to come from the runs: a caller-supplied one would let a comparison be
    declared comparable while the records said otherwise.
    """
    tracks = {getattr(record, "track_sha256") for record in records if getattr(record, "arm", None) == arm}
    if len(tracks) != 1:
        raise IncomparableArms(
            f"arm {arm} has {len(tracks)} distinct tracks; a single condition cannot describe it"
        )
    if any(not track for track in tracks):
        raise IncomparableArms(f"arm {arm} contains a missing track digest")
    tasks = tuple(sorted({getattr(record, "task_id") for record in records if getattr(record, "arm", None) == arm}))
    return ArmConditions(
        arm=arm,
        track_sha256=tracks.pop(),
        tasks=tasks,
        budget_steps=budget_steps,
        recipe_fingerprint=recipe_fingerprint,
    )

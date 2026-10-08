"""Measure one run.

Every number here is taken from the trajectory, which is taken from the ledger — not from a summary
the run wrote about itself. The one thing this module cannot read off the ledger is whether the task
succeeded: that is a judgement, it belongs to a judge, and the outcome is carried in rather than
invented.

Counts are deliberately more than a success bit. The design's own risk list names the failure this
guards: aggregate metrics move together with the target capability, so a result has to be able to
say *which* numbers moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.trajectory import Trajectory

__all__ = ["MeasureReport", "RunMetrics", "UNKNOWN_OUTCOME", "measure", "measure_all"]

#: A run whose success nobody judged. Kept distinct from a failure for the same reason the gate
#: keeps `Outcome.UNKNOWN`: an unjudged run and a failed run are different facts, and averaging them
#: together understates the failure count while overstating the coverage.
UNKNOWN_OUTCOME = "unknown"

SUCCEEDED = "succeeded"
FAILED = "failed"


@dataclass(frozen=True)
class RunMetrics:
    """What one run did, and whether it worked."""

    task_id: str
    arm: str
    session_key: str
    outcome: str
    decisions: int
    tool_calls: int
    messages: int
    failed_calls: int
    events: int

    @property
    def succeeded(self) -> bool:
        return self.outcome == SUCCEEDED

    @property
    def decided(self) -> bool:
        """Whether this run can carry a verdict at all."""
        return self.outcome != UNKNOWN_OUTCOME

    def to_row(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "arm": self.arm,
            "session_key": self.session_key,
            "outcome": self.outcome,
            "decisions": self.decisions,
            "tool_calls": self.tool_calls,
            "messages": self.messages,
            "failed_calls": self.failed_calls,
            "events": self.events,
        }


def measure(
    trajectory: Trajectory, *, task_id: str, arm: str, outcome: str = UNKNOWN_OUTCOME
) -> RunMetrics:
    """Read one trajectory into a row of numbers.

    ``failed_calls`` counts decisions whose own tool call reported an error. A run that ended between
    a call and its result has a point with ``errored is None``, which is not a failure — it is
    missing information, and counting it as one would make a truncated run look like a bad one.
    """
    tool_calls = [point for point in trajectory.points if point.action.kind == "tool_call"]
    messages = [point for point in trajectory.points if point.action.kind == "message"]
    return RunMetrics(
        task_id=task_id,
        arm=arm,
        session_key=trajectory.session_key,
        outcome=outcome,
        decisions=len(trajectory.points),
        tool_calls=len(tool_calls),
        messages=len(messages),
        failed_calls=sum(1 for point in tool_calls if point.errored is True),
        events=trajectory.last_seq,
    )


@dataclass
class MeasureReport:
    rows: tuple[RunMetrics, ...] = field(default_factory=tuple)

    def by_arm(self, arm: str) -> tuple[RunMetrics, ...]:
        return tuple(row for row in self.rows if row.arm == arm)

    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted({row.task_id for row in self.rows}))


def measure_all(
    trajectories: dict[str, Trajectory],
    *,
    arm: str,
    outcomes: dict[str, str] | None = None,
) -> MeasureReport:
    """Measure a batch keyed ``arm/task``, the way `harness.rollout.collect` produces them."""
    rows = []
    for key, trajectory in sorted(trajectories.items()):
        key_arm, _, task_id = key.partition("/")
        if key_arm != arm:
            continue
        rows.append(
            measure(
                trajectory,
                task_id=task_id,
                arm=arm,
                outcome=(outcomes or {}).get(task_id, UNKNOWN_OUTCOME),
            )
        )
    return MeasureReport(rows=tuple(rows))

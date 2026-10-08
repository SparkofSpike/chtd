"""Find the numbers that moved with the capability without being caused by it.

The design's second stated failure of cross-harness distillation is an evaluation that rewards the
wrong thing: aggregate metrics — length, latency, token spend, steps taken — move together with the
target capability, and an improvement is then attributed to the wrong cause. Stripping them has to
happen in both places: from what the training optimises, and from what the report claims.

This module identifies candidates, and refuses to let one be kept without a stated reason. A
co-mover that a reader cannot distinguish from the capability is not evidence for the capability, so
the default is to strip and the exception is an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from eval.arms import ComparisonResult

__all__ = [
    "DEFAULT_CANDIDATES",
    "DistractorDecision",
    "DistractorFinding",
    "ReasonRequired",
    "analyse",
    "pearson",
]

#: The co-movers the design names. Each is a column of `RunMetrics` that can rise without the
#: capability rising: a longer run spends more, a more verbose model writes more, and a run that
#: fails faster takes fewer steps.
DEFAULT_CANDIDATES = ("decisions", "tool_calls", "messages", "failed_calls", "events")

CO_MOVER = "co-mover"
SEPARABLE = "separable"
UNDETERMINED = "undetermined"


class ReasonRequired(ValueError):
    """A co-mover was kept without a reason, which makes the claim unfalsifiable."""


@dataclass(frozen=True)
class DistractorDecision:
    """What to do about one candidate, and why."""

    keep: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.keep and not self.reason.strip():
            raise ReasonRequired(
                "keeping a co-mover requires a stated reason: without one, a reader cannot tell "
                "this number from the capability it moved with"
            )


@dataclass(frozen=True)
class DistractorFinding:
    metric: str
    correlation: float | None
    paired_tasks: int
    verdict: str
    kept: bool
    reason: str

    def to_row(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "correlation": None if self.correlation is None else round(self.correlation, 6),
            "paired_tasks": self.paired_tasks,
            "verdict": self.verdict,
            "kept": self.kept,
            "reason": self.reason,
        }


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """The correlation coefficient, or ``None`` when it is not defined.

    Undefined rather than zero when one side never varies: a candidate that took the same value on
    every task has no relationship to report, and returning 0.0 would let it pass as "separable"
    on the strength of a number that does not exist.
    """
    if len(xs) != len(ys):
        raise ValueError("pearson needs two sequences of the same length")
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    covariance = sum(a * b for a, b in zip(dx, dy))
    variance_x = sum(a * a for a in dx)
    variance_y = sum(b * b for b in dy)
    if variance_x == 0 or variance_y == 0:
        return None
    return covariance / (variance_x**0.5 * variance_y**0.5)


def analyse(
    comparison: ComparisonResult,
    *,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
    decisions: Mapping[str, DistractorDecision] | None = None,
    threshold: float = 0.8,
) -> tuple[DistractorFinding, ...]:
    """Correlate each candidate's arm-to-arm change with the capability's.

    The correlation is across tasks, over the *differences* between arms rather than over raw values:
    a candidate that differs between arms in step with success is the candidate that would be
    credited for the capability if nobody looked.

    ``threshold`` is a parameter because where a co-mover becomes inseparable is a judgement about
    the experiment, not a fact about the arithmetic, and burying it in the code would make the
    conclusion look like a measurement.
    """
    decisions = decisions or {}
    decided = comparison.decided_pairs
    capability = [1.0 if pair.chtd.succeeded else 0.0 for pair in decided]
    baseline_capability = [1.0 if pair.baseline.succeeded else 0.0 for pair in decided]
    capability_delta = [a - b for a, b in zip(capability, baseline_capability)]

    findings: list[DistractorFinding] = []
    for metric in candidates:
        if not decided or not hasattr(decided[0].chtd, metric):
            findings.append(
                DistractorFinding(
                    metric=metric,
                    correlation=None,
                    paired_tasks=0,
                    verdict=UNDETERMINED,
                    kept=False,
                    reason=f"no '{metric}' column on the measured rows",
                )
            )
            continue
        deltas = [
            float(getattr(pair.chtd, metric)) - float(getattr(pair.baseline, metric))
            for pair in decided
        ]
        correlation = pearson(capability_delta, deltas)
        if correlation is None:
            findings.append(
                DistractorFinding(
                    metric=metric,
                    correlation=None,
                    paired_tasks=len(decided),
                    verdict=UNDETERMINED,
                    kept=False,
                    reason=(
                        "no relationship to measure: this metric or the capability did not vary "
                        "across tasks, so they cannot be told apart from this data"
                    ),
                )
            )
            continue
        co_moving = abs(correlation) >= threshold
        decision = decisions.get(metric)
        kept = bool(decision and decision.keep)
        reason = decision.reason if decision else ""
        findings.append(
            DistractorFinding(
                metric=metric,
                correlation=correlation,
                paired_tasks=len(decided),
                verdict=CO_MOVER if co_moving else SEPARABLE,
                kept=kept,
                reason=(
                    reason
                    if kept
                    else (
                        "moved with the capability and was stripped from the reported evidence"
                        if co_moving
                        else "did not move with the capability at this threshold"
                    )
                ),
            )
        )
    return tuple(findings)


def kept_metrics(findings: Sequence[DistractorFinding]) -> tuple[str, ...]:
    """The candidates a report may still cite as evidence."""
    return tuple(finding.metric for finding in findings if finding.kept)


def stripped_metrics(findings: Sequence[DistractorFinding]) -> tuple[str, ...]:
    """The candidates the report must not cite, and the count is a result in its own right."""
    return tuple(finding.metric for finding in findings if not finding.kept)

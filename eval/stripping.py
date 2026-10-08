"""The stripping ladder: does the gain survive the harness being taken away?

A student can learn to behave well inside one harness without acquiring the capability, and the
difference between those two outcomes is the difference between a result and an artifact. The ladder
is the guard: repeat the evaluation under progressively less scaffolding, and require the gain to
survive.

* **same track** — the training conditions. A gain here is the primary measurement.
* **different preset** — another recipe: different prompt sections, different budgets, same harness.
  Surviving this means the gain is not merely a fit to the track's own prompt format.
* **no tools** — the task attempted without the action space.
* **single turn** — no loop at all.

A gain that exists only on the first rung is reported as scaffold overfit, not as a capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from eval.arms import ComparisonResult

__all__ = [
    "RUNG_ORDER",
    "LadderVerdict",
    "RungResult",
    "SCAFFOLD_OVERFIT",
    "SURVIVED",
    "NO_GAIN",
    "assess",
]

SAME_TRACK = "same-track"
DIFFERENT_PRESET = "different-preset"
NO_TOOLS = "no-tools"
SINGLE_TURN = "single-turn"

#: In order of decreasing scaffolding. The ladder is ordered because it is read as a ladder: each
#: rung removes something, and a rung out of order would report a survival that never happened.
RUNG_ORDER = (SAME_TRACK, DIFFERENT_PRESET, NO_TOOLS, SINGLE_TURN)

SURVIVED = "survived"
SCAFFOLD_OVERFIT = "scaffold-overfit"
NO_GAIN = "no-gain"


@dataclass(frozen=True)
class RungResult:
    """One rung of the ladder."""

    rung: str
    comparison: ComparisonResult

    @property
    def lift(self) -> float:
        return self.comparison.lift

    @property
    def tasks(self) -> int:
        return len(self.comparison.decided_pairs)


@dataclass(frozen=True)
class LadderVerdict:
    rungs: tuple[RungResult, ...]
    #: How much of the same-track lift must remain on a stripped rung for it to count as survival.
    #: A parameter rather than a constant because the right threshold is an argument about the
    #: experiment, and burying it would make the conclusion look like arithmetic.
    survival_threshold: float
    verdict: str
    reason: str

    @property
    def same_track_lift(self) -> float:
        for rung in self.rungs:
            if rung.rung == SAME_TRACK:
                return rung.lift
        raise ValueError("the ladder has no same-track rung, so it has no primary measurement")

    def rung(self, name: str) -> RungResult:
        for rung in self.rungs:
            if rung.rung == name:
                return rung
        raise KeyError(name)


def assess(
    rungs: Mapping[str, ComparisonResult],
    *,
    survival_threshold: float = 0.5,
) -> LadderVerdict:
    """Decide what a completed ladder says.

    The three outcomes are kept distinct because a report that merged them would be unable to say
    the difference between a method that failed and a method that worked for the wrong reason:

    * **no-gain** — the same-track lift is not positive, so there is nothing to explain.
    * **survived** — the lift is positive on the training track and remains at least
      ``survival_threshold`` of it on every stripped rung.
    * **scaffold-overfit** — the lift is positive on the training track and does not survive.

    Every rung the caller supplied is used. A ladder missing a rung is neither completed nor refused
    here: it is assessed on what it has, and the missing rungs stay visible in ``rungs``, because
    refusing to assess a partial ladder would push the caller towards not running the missing rungs.
    """
    missing = [name for name in rungs if name not in RUNG_ORDER]
    if missing:
        raise ValueError(f"unknown rung(s) {sorted(missing)}; the ladder is {list(RUNG_ORDER)}")
    if SAME_TRACK not in rungs:
        raise ValueError("a ladder without the same-track rung has no primary measurement")

    ordered = tuple(RungResult(rung=name, comparison=rungs[name]) for name in RUNG_ORDER if name in rungs)
    primary = next(rung for rung in ordered if rung.rung == SAME_TRACK)

    if primary.lift <= 0:
        return LadderVerdict(
            rungs=ordered,
            survival_threshold=survival_threshold,
            verdict=NO_GAIN,
            reason=(
                f"the same-track lift is {primary.lift:+.3f}, so there is no gain to explain; "
                "the remaining rungs describe where the student already stood"
            ),
        )

    stripped = [rung for rung in ordered if rung.rung != SAME_TRACK]
    if not stripped:
        return LadderVerdict(
            rungs=ordered,
            survival_threshold=survival_threshold,
            verdict=SCAFFOLD_OVERFIT,
            reason=(
                f"the gain is {primary.lift:+.3f} on the training track and no stripped rung was "
                "run, so nothing separates the capability from a fit to this harness"
            ),
        )

    required = primary.lift * survival_threshold
    weakest = min(stripped, key=lambda rung: rung.lift)
    if weakest.lift >= required:
        return LadderVerdict(
            rungs=ordered,
            survival_threshold=survival_threshold,
            verdict=SURVIVED,
            reason=(
                f"the gain survives stripping: {primary.lift:+.3f} on the training track, "
                f"{weakest.lift:+.3f} at its weakest on {weakest.rung}, "
                f"against a required {required:+.3f}"
            ),
        )
    return LadderVerdict(
        rungs=ordered,
        survival_threshold=survival_threshold,
        verdict=SCAFFOLD_OVERFIT,
        reason=(
            f"the gain does not survive stripping: {primary.lift:+.3f} on the training track but "
            f"{weakest.lift:+.3f} on {weakest.rung}, below the required {required:+.3f}. "
            "This is scaffold overfit and must be reported as such, not as a capability."
        ),
    )


def describe(verdict: LadderVerdict) -> str:
    """The ladder as a few lines a report can carry."""
    lines = [f"verdict: {verdict.verdict}", f"reason: {verdict.reason}", "rungs:"]
    for rung in verdict.rungs:
        lines.append(
            f"  {rung.rung}: lift {rung.lift:+.3f} over {rung.tasks} task(s)"
        )
    return "\n".join(lines)

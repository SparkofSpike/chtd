"""Write the result up, including the parts that do not flatter it.

The design says a reader must be able to rerun the pipeline and reach the same numbers — including
the negative ones. That makes the report a product of the pipeline rather than a document written
beside it: the numbers come from the objects the other modules produced, and the limitations are a
list the code carries rather than a paragraph somebody remembers to add.

A report that omitted a stripped distractor, an unanswered state or an unrun rung would describe an
experiment nobody ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from distill.dataset import DatasetManifest
from eval.arms import ArmConditions, ComparisonResult, summarise
from eval.distractors import DistractorFinding, stripped_metrics
from eval.stripping import LadderVerdict

__all__ = ["Report", "build", "write"]

#: The limitations every report carries because they are true of every run of this pipeline. A
#: limitation that depends on the run is passed in instead; this list is the floor, not the ceiling.
STANDING_LIMITATIONS = (
    "One capability. A transfer shown on one axis says nothing about others.",
    "One harness. The stripping ladder varies the preset, the tools and the turn count; a genuinely "
    "foreign harness is the stronger test and is left as future work.",
    "One DAgger round. The student is not retrained and re-rolled; the dataset is collected once.",
    "Outcomes are judged, not computed. Every success/failure figure depends on the judge named "
    "above, and a different judge is a different result.",
)


@dataclass(frozen=True)
class Report:
    title: str
    track_sha256: str
    conditions: tuple[ArmConditions, ...]
    comparison: ComparisonResult
    ladder: LadderVerdict | None
    distractors: tuple[DistractorFinding, ...]
    dataset: DatasetManifest | None
    judge: str
    extra_limitations: tuple[str, ...] = field(default_factory=tuple)

    def limitations(self) -> tuple[str, ...]:
        return STANDING_LIMITATIONS + self.extra_limitations

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"Track: `{self.track_sha256}`")
        lines.append(f"Judge: {self.judge}")
        lines.append("")

        lines.append("## Conditions")
        lines.append("")
        for condition in self.conditions:
            lines.append(
                f"- **{condition.arm}**: track `{condition.track_sha256[:12]}`, "
                f"{len(condition.tasks)} task(s), step budget {condition.budget_steps}, "
                f"recipe `{condition.recipe_fingerprint[:12]}`"
            )
        lines.append("")

        lines.append("## The deciding comparison")
        lines.append("")
        summary = self.comparison.summary()
        lines.append(summarise(self.comparison))
        lines.append("")
        lines.append("| | baseline | CHTD |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| success rate | {summary['baseline_success_rate']:.3f} | "
            f"{summary['chtd_success_rate']:.3f} |"
        )
        lines.append(f"| lift | | {summary['lift']:+.3f} |")
        lines.append(f"| gained | | {summary['gained']} |")
        lines.append(f"| lost | | {summary['lost']} |")
        lines.append(f"| tasks undecided | | {summary['tasks_undecided']} |")
        lines.append("")

        lines.append("## Harness-stripping ladder")
        lines.append("")
        if self.ladder is None:
            lines.append("Not run. Without it, a gain on the training track cannot be separated "
                         "from a fit to this harness, and the result must be reported as unseparated.")
        else:
            lines.append(f"**{self.ladder.verdict}** — {self.ladder.reason}")
            lines.append("")
            lines.append("| rung | lift | tasks |")
            lines.append("| --- | --- | --- |")
            for rung in self.ladder.rungs:
                lines.append(f"| {rung.rung} | {rung.lift:+.3f} | {rung.tasks} |")
        lines.append("")

        lines.append("## Distractors")
        lines.append("")
        if not self.distractors:
            lines.append("None analysed.")
        else:
            lines.append("| metric | correlation | paired tasks | verdict | kept |")
            lines.append("| --- | --- | --- | --- | --- |")
            for finding in self.distractors:
                correlation = (
                    "n/a" if finding.correlation is None else f"{finding.correlation:+.3f}"
                )
                lines.append(
                    f"| {finding.metric} | {correlation} | {finding.paired_tasks} | "
                    f"{finding.verdict} | {'yes' if finding.kept else 'no'} |"
                )
            stripped = stripped_metrics(self.distractors)
            if stripped:
                lines.append("")
                lines.append(
                    f"Stripped from the evidence: {', '.join(stripped)}. A number that moved with "
                    "the capability is not evidence for it unless it was caused by it."
                )
            kept = [finding for finding in self.distractors if finding.kept]
            if kept:
                # The reason is the whole point of requiring one: a report that listed a kept
                # co-mover without it would make the exception invisible and the rule pointless.
                lines.append("")
                lines.append("Kept as evidence, with the reason it is causal:")
                for finding in kept:
                    lines.append(f"- **{finding.metric}** — {finding.reason}")
        lines.append("")

        lines.append("## Dataset")
        lines.append("")
        if self.dataset is None:
            lines.append("No dataset manifest attached.")
        else:
            lines.append(f"- digest: `{self.dataset.dataset_sha256}`")
            lines.append(f"- entries: {self.dataset.entries}")
            lines.append(f"- refusals: {self.dataset.refusals}")
            lines.append(f"- tasks: {', '.join(self.dataset.tasks)}")
        lines.append("")

        lines.append("## Limitations")
        lines.append("")
        for limitation in self.limitations():
            lines.append(f"- {limitation}")
        lines.append("")

        lines.append("## Reproducing this")
        lines.append("")
        lines.append("1. Build the dataset: `python -m distill.dataset` over the collected rollouts.")
        lines.append("2. Train each arm from its own plan under the same recipe.")
        lines.append("3. Re-run every rung of the ladder and rebuild this report.")
        lines.append("")
        lines.append("A reproduction that reaches different numbers is a different experiment; "
                     "compare the track digest and the dataset digest first.")
        lines.append("")
        return "\n".join(lines)


def build(
    *,
    title: str,
    track_sha256: str,
    conditions: Sequence[ArmConditions],
    comparison: ComparisonResult,
    ladder: LadderVerdict | None,
    distractors: Sequence[DistractorFinding],
    dataset: DatasetManifest | None,
    judge: str,
    extra_limitations: Sequence[str] = (),
) -> Report:
    """Assemble a report. Deliberately not a function that writes one — see `write`."""
    if not conditions:
        raise ValueError("a report with no conditions cannot say what was compared")
    return Report(
        title=title,
        track_sha256=track_sha256,
        conditions=tuple(conditions),
        comparison=comparison,
        ladder=ladder,
        distractors=tuple(distractors),
        dataset=dataset,
        judge=judge,
        extra_limitations=tuple(extra_limitations),
    )


def write(report: Report, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.render(), encoding="utf-8")
    return target

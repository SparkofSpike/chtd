"""A whole collection round as one dataset.

`distill/pipeline.py` builds a dataset from one task: one student session, the teacher forks that
answered in its states. A round is a dozen of those, and taking them one at a time would produce a
dozen datasets — each with its own digest, none of them the thing that gets trained on.

Two things this module does that a per-task command cannot.

**Outcomes are derived, not supplied.** The signal gate admits a divergence only when the teacher's
branch succeeded and the student's did not, so it needs an outcome for each side at the shared
state. What a round observes is simpler than that: each run's own check ran in its own workspace
afterwards, so there is one verdict per run, and it applies to every state in it. Deriving the rows
here means the gate's evidence and the round's evidence cannot drift apart.

**Every task must be on one track.** The records carry the track observed from each session, and
`distill.dataset` refuses to mix two — which is checked here rather than assumed, because a round
that quietly spanned two tracks would produce a dataset that measures neither.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from distill.dataset import Dataset, build_dataset, write_dataset
from distill.divergence import Alignment, align
from distill.gate import (
    Outcome,
    OutcomeTable,
    OutcomeValidator,
    SignalGate,
    WorkspaceRelativePaths,
)
from harness.ledger import open_ledger
from harness.rollout import RolloutRecord, read_records
from harness.trajectory import Trajectory, read_trajectory

__all__ = ["RoundIncomplete", "build_round_dataset", "main"]


class RoundIncomplete(RuntimeError):
    """A round is missing a side, or the sides disagree about something they must agree on."""


def _load_checks(path: str | Path) -> dict[tuple[str, str], bool]:
    """``(task_id, arm) -> passed``, from the file `harness.round` writes."""
    source = Path(path)
    if not source.exists():
        raise RoundIncomplete(f"no checks at {source}; a round without outcomes cannot be gated")
    checks: dict[tuple[str, str], bool] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("passed") is None:
            continue  # a run that never produced a check; its record is absent too
        checks[(row["task_id"], row["arm"])] = bool(row["passed"])
    return checks


def build_round_dataset(
    *,
    records_path: str | Path,
    checks_path: str | Path,
    arms: Sequence[str] = ("student", "teacher"),
) -> tuple[Dataset, dict[str, Alignment]]:
    """Read a round and build the dataset it produced. Writes nothing."""
    records = read_records(records_path)
    if not records:
        raise RoundIncomplete(f"{records_path} holds no records")

    checks = _load_checks(checks_path)

    by_task: dict[str, dict[str, RolloutRecord]] = {}
    for record in records:
        by_task.setdefault(record.task_id, {})[record.arm] = record
    if not by_task:
        raise RoundIncomplete("no tasks in the round")

    # A task with only one side is skipped and counted, not fatal. A run that failed is a fact about
    # the round — the endpoint refused it, a turn was blocked, an adapter was not loaded — and
    # aborting the build would throw away every other task's data to punish one. What would be fatal
    # is a task whose side *is* present and whose outcome is missing, which is an inconsistency in
    # this repository rather than in the world; that is checked below.
    complete: dict[str, dict[str, RolloutRecord]] = {}
    incomplete: dict[str, list[str]] = {}
    for task_id, sides in sorted(by_task.items()):
        missing = [arm for arm in arms if arm not in sides]
        if missing:
            incomplete[task_id] = missing
            continue
        complete[task_id] = sides

    if not complete:
        raise RoundIncomplete(
            "every task in the round is missing one side; there is nothing to align"
        )

    # Every state a row is built from has to be gated on both sides, so the outcomes are derived
    # from each run's own check rather than passed in: one verdict per run, applied to every state
    # in it. The alternative — a file somebody maintains — drifts from the checks silently.
    table = OutcomeTable({})
    for task_id, sides in sorted(complete.items()):
        for arm in arms:
            record = sides[arm]
            passed = checks.get((task_id, arm))
            if passed is None:
                raise RoundIncomplete(
                    f"{task_id}/{arm} has a record but no observed outcome; the round is internally "
                    "inconsistent and a dataset built from it would gate on nothing"
                )
            outcome = Outcome.SUCCEEDED if passed else Outcome.FAILED
            with open_ledger(record.ledger_path) as ledger:
                trajectory = read_trajectory(ledger, record.session_key)
            for point in trajectory.points:
                table.record(record.session_key, point.prefix_digest, outcome)

    gate = SignalGate([OutcomeValidator(table), WorkspaceRelativePaths()])

    alignments: dict[str, Alignment] = {}
    for task_id, sides in sorted(complete.items()):
        student_record = sides["student"]
        teacher_record = sides["teacher"]
        with open_ledger(student_record.ledger_path) as ledger:
            student: Trajectory = read_trajectory(ledger, student_record.session_key)
        with open_ledger(teacher_record.ledger_path) as ledger:
            teacher: Trajectory = read_trajectory(ledger, teacher_record.session_key)
        alignments[task_id] = align(student, teacher)

    dataset = build_dataset(
        # De-duplicated and ordered. `build_dataset` walks this list and gates each task's
        # alignment on every pass, so a repeated id duplicates every row it produces — and the list
        # built from the records has one entry per *run*, which is two per task. Measured: the first
        # round produced 24 rows for 12 divergences before this line existed.
        task_ids=list(dict.fromkeys(record.task_id for record in records)),
        alignments=alignments,
        gate=gate,
        records=records,
        arms=list(arms),
    )
    if incomplete:
        # Carried out on the dataset rather than printed, because every report is generated from
        # these objects and a limitation that only reached a console would not reach the write-up.
        dataset.reasons = {
            **dataset.reasons,
            "task-missing-a-side": sum(len(v) for v in incomplete.values()),
        }
    return dataset, alignments, incomplete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one dataset from a whole collection round."
    )
    parser.add_argument("--round", required=True, help="a directory harness.round wrote")
    parser.add_argument("--out", required=True, help="where the dataset is written")
    args = parser.parse_args(argv)

    source = Path(args.round)
    dataset, alignments, incomplete = build_round_dataset(
        records_path=source / "records.jsonl",
        checks_path=source / "checks.jsonl",
    )
    paths = write_dataset(dataset, args.out)

    manifest = dataset.manifest
    assert manifest is not None
    total_divergences = sum(len(a.divergences) for a in alignments.values())
    total_agreements = sum(len(a.agreements) for a in alignments.values())
    print(f"tasks      : {len(alignments)} aligned, {len(incomplete)} skipped")
    for task_id, missing in sorted(incomplete.items()):
        print(f"  skipped {task_id}: no {' and '.join(missing)} run")
    print(f"aligned    : {total_divergences} divergences, {total_agreements} agreements")
    print(f"admitted   : {manifest.entries}, refused: {manifest.refusals}")
    for reason, count in sorted(dataset.reasons.items()):
        print(f"  refused by {reason}: {count}")
    print(f"dataset digest: {manifest.dataset_sha256}")
    print(f"track      : {manifest.track_sha256}")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main(sys.argv[1:]))

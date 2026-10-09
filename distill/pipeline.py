"""The pipeline as a command, so the reproduction steps in a report are something a reader can run.

    python -m distill.pipeline dataset \\
        --student-ledger data/rollouts/student/t1/sessions.db --student-session <key> \\
        --teacher-ledger data/rollouts/teacher/t1/sessions.db --teacher-session <key> \\
        --task t1 --track <track-sha256> --out data/datasets/round1

It reads ledgers, aligns the teacher's forks against the student's run, applies the gate, and writes
the dataset with its manifest. Everything it needs is a file, so a reader can rerun it on the
recorded rollouts and compare the dataset digest — which is the property the write-up promises.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from distill.dataset import Dataset, build_dataset, write_dataset
from distill.divergence import Alignment, align, align_all, divergence_rate
from distill.gate import OutcomeTable, OutcomeValidator, RecordedVerdicts, SignalGate, Validator
from harness.ledger import open_ledger
from harness.trajectory import Trajectory, read_trajectory

__all__ = ["build_from_ledgers", "main"]


def build_from_ledgers(
    *,
    student_ledger: str | Path,
    student_session: str,
    teacher_ledgers: Sequence[tuple[str | Path, str]],
    task_id: str,
    track_sha256: str,
    verdicts_path: str | Path | None = None,
    outcomes_path: str | Path | None = None,
) -> tuple[Dataset, Alignment, Trajectory]:
    """Read the ledgers, align, gate, and build. Returns the dataset and what it was built from."""
    with open_ledger(student_ledger) as ledger:
        student = read_trajectory(ledger, student_session)

    teachers: list[Trajectory] = []
    for ledger_path, session_key in teacher_ledgers:
        with open_ledger(ledger_path) as ledger:
            teachers.append(read_trajectory(ledger, session_key))

    alignment = align(student, teachers[0]) if len(teachers) == 1 else align_all(student, teachers)

    validators: list[Validator] = []
    outcomes_given = outcomes_path is not None
    verdicts_given = verdicts_path is not None
    if outcomes_given:
        outcomes = _read_outcomes(outcomes_path)
        if not outcomes:
            # Given but empty is not the same as absent. Folding it into "not requested" would drop
            # the outcome gate silently, and the refusal below would then tell the caller to pass
            # the flag they already passed.
            raise SystemExit(
                f"{outcomes_path} has no outcome records; a branch outcome is what validates a "
                "divergence, so an empty file is an unfinished export, not a choice to skip it"
            )
        validators.append(OutcomeValidator(OutcomeTable.from_records(outcomes)))
    if verdicts_given:
        validators.append(RecordedVerdicts.from_path(verdicts_path))
    if not validators:
        # A gate with no validators admits everything, which is the behaviour it exists to prevent,
        # so this is a refusal and not a default.
        raise SystemExit(
            "refusing to build an ungated dataset: pass --outcomes (branch outcomes) or "
            "--verdicts (recorded judgements)"
        )

    dataset = build_dataset(
        task_ids=[task_id],
        alignments={task_id: alignment},
        gate=SignalGate(validators),
        records=[
            _record(task_id, "student", student.session_key, track_sha256),
            *[_record(task_id, "teacher", t.session_key, track_sha256) for t in teachers],
        ],
        arms=["baseline", "chtd"],
    )
    return dataset, alignment, student


def _record(task_id: str, arm: str, session_key: str, track_sha256: str):
    from harness.rollout import RolloutRecord

    return RolloutRecord(
        task_id=task_id,
        arm=arm,
        session_key=session_key,
        ledger_path="",
        track_sha256=track_sha256,
        preset="chtd",
        route=arm,
        model="",
    )


def _read_outcomes(path: str | Path) -> list[dict]:
    import json

    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                rows.append(json.loads(text))
            except ValueError as error:
                raise SystemExit(f"{source}:{number} is not JSON: {error}") from error
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distill.pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = sub.add_parser("dataset", help="build a dataset from recorded rollouts")
    dataset.add_argument("--student-ledger", required=True)
    dataset.add_argument("--student-session", required=True)
    dataset.add_argument(
        "--teacher",
        action="append",
        required=True,
        metavar="LEDGER:SESSION",
        help="a teacher fork's ledger and session key; repeat for one fork per state",
    )
    dataset.add_argument("--task", required=True, dest="task_id")
    dataset.add_argument("--track", required=True, help="the frozen track digest the runs used")
    dataset.add_argument("--out", required=True)
    dataset.add_argument("--outcomes", help="JSONL of {session_key, prefix_digest, outcome}")
    dataset.add_argument("--verdicts", help="JSONL of {prefix_digest, admitted}")

    args = parser.parse_args(argv)
    if args.command != "dataset":  # pragma: no cover - argparse refuses unknown commands first
        parser.error(f"unknown command {args.command}")

    teacher_pairs: list[tuple[str, str]] = []
    for spec in args.teacher:
        ledger_path, sep, session_key = spec.rpartition(":")
        if not sep or not ledger_path or not session_key:
            parser.error(f"--teacher needs LEDGER:SESSION, got {spec!r}")
        teacher_pairs.append((ledger_path, session_key))

    dataset_result, alignment, student = build_from_ledgers(
        student_ledger=args.student_ledger,
        student_session=args.student_session,
        teacher_ledgers=teacher_pairs,
        task_id=args.task_id,
        track_sha256=args.track,
        verdicts_path=args.verdicts,
        outcomes_path=args.outcomes,
    )
    paths = write_dataset(dataset_result, args.out)

    manifest = dataset_result.manifest
    assert manifest is not None
    print(f"student: {student.session_key} ({len(student.points)} decisions)")
    print(
        f"aligned: {len(alignment.divergences)} divergences, {len(alignment.agreements)} agreements, "
        f"{len(alignment.unanswered)} unanswered, {len(alignment.novel)} novel"
    )
    print(f"divergence rate over answered decisions: {divergence_rate(alignment):.3f}")
    print(f"admitted: {manifest.entries}, refused: {manifest.refusals}")
    if dataset_result.reasons:
        for reason, count in sorted(dataset_result.reasons.items()):
            print(f"  refused by {reason}: {count}")
    print(f"dataset digest: {manifest.dataset_sha256}")
    print(f"track: {manifest.track_sha256}")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())

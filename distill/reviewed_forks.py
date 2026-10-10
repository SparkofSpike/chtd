"""Build gated CHTD samples from reviewed, ledger-verified after-turn forks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distill.dataset import build_dataset, write_dataset
from distill.divergence import align
from distill.gate import (
    Outcome, OutcomeTable, OutcomeValidator, RecordedVerdicts, SignalGate, WorkspaceRelativePaths,
)
from distill.sft import HarnessPrefix, build_samples, sample_rows, samples_digest, write_samples
from harness.digest import raw_sha256, text_sha256
from harness.ledger import open_ledger
from harness.rollout import RolloutRecord
from harness.trajectory import read_trajectory


def build(student_run: Path, fork_run: Path, review_file: Path, ledger_file: Path,
          prefix_file: Path, output: Path) -> dict:
    manifest_file = student_run / "collection-manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    review = json.loads(review_file.read_text(encoding="utf-8"))
    task_ids = manifest["task_ids"]
    if review.get("status") != "complete" or set(review.get("tasks", {})) != set(task_ids):
        raise ValueError("manual review must cover every training task")
    if set(task_ids) & set(manifest["holdout_ids"]):
        raise ValueError("training and holdout tasks overlap")
    track = manifest["expected_track_sha256"]
    prefix = HarnessPrefix.load(prefix_file)
    if prefix.provenance.get("preset") != "minimal-rl":
        raise ValueError("training prefix is not the frozen minimal-rl preset")
    alignments = {}
    outcomes = OutcomeTable({})
    label_verdicts = {}
    records = []
    with open_ledger(ledger_file) as ledger:
        for task_id in task_ids:
            student = json.loads((student_run / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
            fork = json.loads((fork_run / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
            judgement = review["tasks"][task_id]
            if not isinstance(judgement.get("teacher_eligible"), bool) or not isinstance(
                    judgement.get("student_succeeded"), bool):
                raise ValueError(f"{task_id}: review needs boolean decisions")
            correction_candidate = judgement["teacher_eligible"] and not judgement["student_succeeded"]
            if correction_candidate and not isinstance(judgement.get("label_eligible"), bool):
                raise ValueError(f"{task_id}: teacher-win state needs a label-level judgement")
            if student["session_key"] != fork["parent_key"] or student["record"]["track_sha256"] != track:
                raise ValueError(f"{task_id}: student or track differs from the collection")
            if fork["teacher_reason"] != "completed" or fork["student_reason"] != "completed":
                raise ValueError(f"{task_id}: continuation did not complete")
            student_trajectory = read_trajectory(ledger, fork["parent_key"])
            teacher_trajectory = read_trajectory(ledger, fork["child_key"])
            if teacher_trajectory.parent_key != student_trajectory.session_key or (
                    teacher_trajectory.boundary_seq != fork["boundary_seq"]):
                raise ValueError(f"{task_id}: teacher was not forked from the student boundary")
            alignment = align(student_trajectory, teacher_trajectory)
            if any(d.student.seq <= fork["boundary_seq"] or d.teacher.seq <= fork["boundary_seq"]
                   for d in alignment.divergences):
                raise ValueError(f"{task_id}: divergence predates the fork")
            if correction_candidate and len(alignment.divergences) != 1:
                raise ValueError(f"{task_id}: review names one label but found "
                                 f"{len(alignment.divergences)} divergences")
            alignments[task_id] = alignment
            for divergence in alignment.divergences:
                outcome_teacher = Outcome.SUCCEEDED if judgement["teacher_eligible"] else Outcome.FAILED
                outcome_student = Outcome.SUCCEEDED if judgement["student_succeeded"] else Outcome.FAILED
                outcomes.record(divergence.teacher_session, divergence.prefix_digest, outcome_teacher)
                outcomes.record(divergence.student_session, divergence.prefix_digest, outcome_student)
                if correction_candidate:
                    label_verdicts[divergence.prefix_digest] = judgement["label_eligible"]
            records.extend((
                RolloutRecord(task_id, "student", student_trajectory.session_key,
                              str(ledger_file), track, "minimal-rl", "student", "qwen3-4b-instruct-2507"),
                RolloutRecord(task_id, "teacher", teacher_trajectory.session_key,
                              str(ledger_file), track, "minimal-rl", "teacher", "agnes-3.0-flash"),
            ))
    dataset = build_dataset(
        task_ids=task_ids, alignments=alignments,
        gate=SignalGate((OutcomeValidator(outcomes), RecordedVerdicts(label_verdicts),
                         WorkspaceRelativePaths())),
        records=records, arms=("baseline", "chtd"),
    )
    if not dataset.entries:
        raise ValueError("manual and path gate admitted no teacher decision")
    output.mkdir(parents=True, exist_ok=True)
    write_dataset(dataset, output / "dataset")
    samples = build_samples(dataset=dataset, student_ledger=ledger_file, prefix=prefix)
    write_samples(samples, output / "samples.jsonl")
    selected = sorted({entry.task_id for entry in dataset.entries})
    report = {
        "definition": "teacher decisions at student-reached completed-turn states",
        "limitation": "AGH cannot fork within a turn; this is an after-turn recovery experiment",
        "student_collection_sha256": text_sha256(manifest_file),
        "review_sha256": text_sha256(review_file),
        "ledger_sha256": raw_sha256(ledger_file),
        "harness_prefix_file_sha256": text_sha256(prefix_file),
        "track_sha256": track,
        "dataset_sha256": dataset.manifest.dataset_sha256,
        "samples_sha256": samples_digest(sample_rows(samples)),
        "admitted_entries": len(dataset.entries),
        "refused_divergences": len(dataset.refusals),
        "selected_task_ids": selected,
        "task_ids": task_ids,
        "holdout_ids": manifest["holdout_ids"],
    }
    (output / "reviewed-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--fork-run", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.student_run, args.fork_run, args.review, args.ledger,
                   args.prefix, args.out)
    print(f"admitted {report['admitted_entries']} decisions from {len(report['selected_task_ids'])} "
          f"tasks; refused {report['refused_divergences']} divergences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

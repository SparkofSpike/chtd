"""Verify the recorded after-turn forks against AGH's durable event ledger."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from distill.divergence import align
from harness.collect_baseline import expected_track, observed_tracks
from harness.collect_forks import FOLLOWUP, _events, _score, _text
from harness.ledger import open_ledger
from harness.tasks.build_workspaces import EXPECT
from harness.trajectory import read_trajectory


def verify(student_run: Path, fork_run: Path, agh_home: Path, *, allow_partial: bool = False) -> dict:
    manifest = json.loads((student_run / "collection-manifest.json").read_text(encoding="utf-8"))
    task_ids = manifest["task_ids"]
    if set(task_ids) & set(manifest["holdout_ids"]):
        raise ValueError("training and holdout tasks overlap")
    track = expected_track(manifest, observed_tracks(student_run, task_ids))
    db = agh_home / "data" / "sessions.db"
    rows = []
    with sqlite3.connect(db) as connection, open_ledger(db) as ledger:
        for task_id in task_ids:
            student_file = student_run / "tasks" / f"{task_id}.json"
            fork_file = fork_run / "tasks" / f"{task_id}.json"
            if not fork_file.exists():
                if allow_partial:
                    continue
                raise ValueError(f"missing fork report for {task_id}")
            task = json.loads(student_file.read_text(encoding="utf-8"))
            fork = json.loads(fork_file.read_text(encoding="utf-8"))
            parent, child, boundary = fork["parent_key"], fork["child_key"], fork["boundary_seq"]
            if task["session_key"] != parent or task["record"]["track_sha256"] != track:
                raise ValueError(f"{task_id}: session or track mismatch")
            ancestry = connection.execute(
                "SELECT parent_key, boundary_seq FROM sessions WHERE session_key=?", (child,)
            ).fetchone()
            if ancestry != (parent, boundary):
                raise ValueError(f"{task_id}: child ancestry mismatch")
            first_end = connection.execute(
                "SELECT data FROM events WHERE session_key=? AND seq=? AND type='turn/end'",
                (parent, boundary),
            ).fetchone()
            if not first_end or json.loads(first_end[0]).get("reason") != "completed":
                raise ValueError(f"{task_id}: boundary is not a completed student turn")
            new_events = {side: _events(connection, key, boundary)
                          for side, key in (("teacher", child), ("student", parent))}
            for side, events in new_events.items():
                prompts = [_text(data) for _, kind, data in events if kind == "user/message"]
                if prompts != [FOLLOWUP] or fork["followup"] != FOLLOWUP:
                    raise ValueError(f"{task_id}: {side} did not receive the frozen follow-up")
                switches = [data for _, kind, data in events if kind == "x/core/model-switch"]
                if not switches or switches[-1].get("to", {}).get("route") != side:
                    raise ValueError(f"{task_id}: {side} route was not selected")
                ends = [data.get("reason") for _, kind, data in events if kind == "turn/end"]
                if ends != ["completed"] or fork[f"{side}_reason"] != "completed":
                    raise ValueError(f"{task_id}: {side} continuation did not complete")
                if _score(events, EXPECT[task_id]) != fork[f"{side}_score"]:
                    raise ValueError(f"{task_id}: {side} report differs from the ledger")
            aligned = align(read_trajectory(ledger, parent), read_trajectory(ledger, child))
            if any(d.student.seq <= boundary or d.teacher.seq <= boundary for d in aligned.divergences):
                raise ValueError(f"{task_id}: apparent divergence predates the fork")
            rows.append({"task_id": task_id, "boundary_seq": boundary,
                         "divergences": len(aligned.divergences),
                         "agreements": len(aligned.agreements),
                         "unanswered": len(aligned.unanswered),
                         "novel": len(aligned.novel),
                         "teacher_new_tools": [data.get("name") for _, kind, data
                                               in new_events["teacher"] if kind == "tool/call"],
                         "student_new_tools": [data.get("name") for _, kind, data
                                               in new_events["student"] if kind == "tool/call"]})
    return {"expected_tasks": len(task_ids), "verified_tasks": len(rows),
            "track_sha256": track, "tasks": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--fork-run", required=True, type=Path)
    parser.add_argument("--agh-home", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = verify(args.student_run, args.fork_run, args.agh_home,
                    allow_partial=args.allow_partial)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(f"verified {report['verified_tasks']}/{report['expected_tasks']} forks on "
          f"track {report['track_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

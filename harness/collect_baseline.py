"""Collect teacher-own-trajectory records for the control arm, one durable task at a time.

This only collects evidence. It does not turn a failed teacher run into a training label,
and it never reads the held-out task set as work to execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.round import Arm, load_round_tasks, run_round
from harness.rollout import CliRunner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_lines(path: Path, lines: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--agh-source", type=Path, required=True)
    parser.add_argument("--agh-home", type=Path, required=True)
    parser.add_argument("--agh-entry", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--throttle-seconds", type=float, default=6.0)
    args = parser.parse_args(argv)
    args.agh_home = args.agh_home.resolve()
    args.agh_entry = args.agh_entry.resolve()
    args.out = args.out.resolve()

    tasks = load_round_tasks(args.tasks)
    holdout_tasks = load_round_tasks(args.holdout)
    holdout_ids = {task.id for task in holdout_tasks}
    overlap = {task.id for task in tasks} & holdout_ids
    if overlap:
        raise ValueError(f"train/holdout task overlap: {sorted(overlap)}")
    key = args.key_file.read_text(encoding="utf-8-sig").strip()
    if not key.startswith("sk-"):
        raise ValueError("the key file does not contain an Agnes sk- credential")

    manifest = {
        "arm": "baseline",
        "definition": "teacher decisions on the teacher's own trajectory",
        "task_set_sha256": _sha(args.tasks),
        "holdout_sha256": _sha(args.holdout),
        "profile_file_sha256": _sha(args.profile_file),
        "collector_sha256": _sha(Path(__file__)),
        "agh_preset_sha256": (args.agh_source / "packages/code/presets/minimal-rl.sha256")
        .read_text(encoding="utf-8").strip(),
        "train_fixture_sha256": {task.id: _fixture_sha(Path(task.fixture)) for task in tasks},
        "holdout_fixture_sha256": {task.id: _fixture_sha(Path(task.fixture))
                                   for task in holdout_tasks},
        "agh_commit": _revision(args.agh_source),
        "chtd_commit": _revision(Path(__file__).resolve().parent.parent),
        "model": "agnes-3.0-flash",
        "preset": "minimal-rl",
        "task_ids": [task.id for task in tasks],
        "holdout_ids": sorted(holdout_ids),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "collection-manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("collection inputs changed; use a new output directory")
    else:
        _write_json(manifest_path, manifest)

    task_dir = args.out / "tasks"
    task_dir.mkdir(exist_ok=True)
    runner = CliRunner(
        entry=str(args.agh_entry),
        profile="local-dev",
        standalone=True,
        env={
            "AGNES_SECRET_TEACHER_DEFAULT": key,
            "AGNES_SECRET_STUDENT_DEFAULT": "local-no-auth",
        },
    )
    arm = Arm(name="teacher", route="teacher", model="agnes-3.0-flash")
    for index, task in enumerate(tasks, 1):
        result_path = task_dir / f"{task.id}.json"
        if result_path.exists():
            print(f"[{index}/{len(tasks)}] {task.id}: recorded", flush=True)
            continue
        print(f"[{index}/{len(tasks)}] {task.id}: running", flush=True)
        outcomes = run_round(
            tasks=[task],
            arms=[arm],
            runner=runner,
            home_root=args.agh_home,
            out_dir=args.out / "workspaces",
            throttle_seconds=args.throttle_seconds,
            log=lambda message: print(message.replace(key, "[REDACTED]"), flush=True),
        )
        outcome = outcomes[0]
        _write_json(
            result_path,
            {
                "task_id": task.id,
                "arm": "teacher",
                "passed": outcome.check_passed,
                "check_output": outcome.check_output.replace(key, "[REDACTED]")[:2000],
                "workspace": outcome.workspace,
                "session_key": outcome.session_key,
                "ledger_path": outcome.ledger_path,
                "record": json.loads(outcome.record.to_json()) if outcome.record else None,
            },
        )

        rows = [json.loads((task_dir / f"{t.id}.json").read_text(encoding="utf-8"))
                for t in tasks if (task_dir / f"{t.id}.json").exists()]
        _write_lines(args.out / "records.jsonl", [json.dumps(r["record"], sort_keys=True)
                                                  for r in rows if r["record"]])
        _write_lines(args.out / "checks.jsonl", [json.dumps({k: r[k] for k in
                ("task_id", "arm", "passed", "check_output", "workspace")}, ensure_ascii=False)
                for r in rows])
        _write_lines(args.out / "outcomes.jsonl", [json.dumps({k: r[k] for k in
                ("task_id", "arm", "passed", "session_key")})
                for r in rows if r["session_key"]])

    rows = [json.loads((task_dir / f"{t.id}.json").read_text(encoding="utf-8")) for t in tasks]
    tracks = {r["record"]["track_sha256"] for r in rows if r["record"]}
    passed = sum(bool(r["passed"] and r["record"]) for r in rows)
    print(f"baseline collected: {len(rows)} tasks, {passed} passed, tracks={sorted(tracks)}", flush=True)
    return 0 if len(tracks) == 1 and passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Collect paired after-turn teacher forks from completed student sessions.

AGH currently forks only at a completed turn/end. Both models receive the same
follow-up from that student-reached state; this is an after-turn recovery study,
not a claim that AGH can fork inside a tool-use turn.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
from pathlib import Path

from harness.tasks.build_workspaces import EXPECT


FOLLOWUP = (
    "Please revisit your diagnosis of the current directory. Read relevant files "
    "if needed; do not run shell commands or modify files. State the precise "
    "defect and why it occurs."
)


def _events(connection: sqlite3.Connection, key: str, after: int = 0):
    return [(seq, kind, json.loads(data)) for seq, kind, data in connection.execute(
        "SELECT seq, type, data FROM events WHERE session_key=? AND seq>? ORDER BY seq",
        (key, after),
    )]


def _text(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    blocks = data.get("content")
    if isinstance(blocks, list):
        return "\n".join(block["text"] for block in blocks
                         if isinstance(block, dict) and isinstance(block.get("text"), str))
    return data.get("text") if isinstance(data.get("text"), str) else ""


def _score(events, expected: tuple[str, ...]) -> dict:
    answer = "\n".join(_text(data) for _, kind, data in events
                       if kind == "assistant/message")
    missing = [phrase for phrase in expected if phrase.lower() not in answer.lower()]
    return {"passed": bool(answer.strip()) and not missing,
            "missing": missing, "answer": answer}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--agh-home", type=Path, required=True)
    parser.add_argument("--agh-source", type=Path, required=True)
    parser.add_argument("--tsx", type=Path, required=True)
    parser.add_argument("--node-dir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    student_run = args.student_run.resolve()
    agh_home = args.agh_home.resolve()
    repo = Path(__file__).resolve().parent.parent
    source = args.agh_source.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    secret = args.key_file.read_text(encoding="utf-8-sig").strip()
    ledger_path = agh_home / "data" / "sessions.db"
    task_files = sorted((student_run / "tasks").glob("*.json"))
    if args.task_id:
        wanted = set(args.task_id)
        task_files = [path for path in task_files if path.stem in wanted]
    if not task_files:
        raise ValueError("no student task records selected")

    with sqlite3.connect(ledger_path) as connection:
        for task_path in task_files:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task_id = task["task_id"]
            destination = out / "tasks" / f"{task_id}.json"
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                print(f"{task_id}: recorded", flush=True)
                continue
            parent = task.get("session_key")
            if not parent or not task.get("record"):
                print(f"{task_id}: no completed student session", flush=True)
                continue
            parent_events = _events(connection, parent)
            ends = [(seq, data) for seq, kind, data in parent_events if kind == "turn/end"]
            if len(ends) != 1 or ends[0][1].get("reason") != "completed":
                print(f"{task_id}: parent has no single completed first turn", flush=True)
                continue
            boundary = ends[0][0]
            command = [str(args.tsx.resolve()), str(repo / "harness" / "fork_turn.mts"),
                       "--agh-source", str(source), "--agh-home", str(agh_home),
                       "--cwd", task["workspace"], "--parent", parent,
                       "--at", str(boundary), "--prompt", FOLLOWUP,
                       "--key-file", str(args.key_file.resolve())]
            print(f"{task_id}: fork at turn/end #{boundary}", flush=True)
            env = dict(os.environ)
            env["PATH"] = str(args.node_dir.resolve()) + os.pathsep + env.get("PATH", "")
            run = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=900)
            if run.returncode:
                error = (run.stderr or run.stdout).replace(secret, "[REDACTED]")
                print(f"{task_id}: fork failed: {error[-500:]}", flush=True)
                continue
            result = json.loads(run.stdout.splitlines()[-1])
            child = result["child_key"]
            ancestry = connection.execute(
                "SELECT parent_key, boundary_seq FROM sessions WHERE session_key=?", (child,)
            ).fetchone()
            teacher_events = _events(connection, child, boundary)
            student_events = _events(connection, parent, boundary)
            teacher_prompts = [_text(data) for _, kind, data in teacher_events
                               if kind == "user/message"]
            student_prompts = [_text(data) for _, kind, data in student_events
                               if kind == "user/message"]
            if ancestry != (parent, boundary) or teacher_prompts != [FOLLOWUP] or student_prompts != [FOLLOWUP]:
                raise ValueError(f"{task_id}: fork ancestry or follow-up prompt differs")
            expected = EXPECT[task_id]
            teacher_score = _score(teacher_events, expected)
            student_score = _score(student_events, expected)
            report = {
                "task_id": task_id, "parent_key": parent, "child_key": child,
                "boundary_seq": boundary, "followup": FOLLOWUP,
                "student_first_passed": task["passed"],
                "student_first_tool_results": sum(
                    kind == "tool/result" for seq, kind, _data in parent_events if seq <= boundary
                ),
                "teacher_reason": result["teacher_reason"],
                "student_reason": result["student_reason"],
                "teacher_score": teacher_score, "student_score": student_score,
                "teacher_new_decisions": sum(kind in ("tool/call", "assistant/message")
                                             for _, kind, _ in teacher_events),
                "student_new_decisions": sum(kind in ("tool/call", "assistant/message")
                                             for _, kind, _ in student_events),
            }
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            print(f"{task_id}: teacher={teacher_score['passed']} student={student_score['passed']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

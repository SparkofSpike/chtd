"""Run a served endpoint (base or adapter on the box) through a holdout set, one home per task.

This is the per-task-home flow the holdout-2 and holdout-3 runs used: unlike
`eval/holdout_sweep.py` (one shared AGH home, one frozen track for every arm), each task here
gets its own home, its own profile expansion and its own ledger. That isolation is why these
runs carry no per-arm track digest — the configuration identity is the profile bytes, checked
across arms — and it is the flow `runs/holdout-3-20261014/README.md` describes.

The 2026-10-14 runs used a copy of this script with the paths hard-coded to this machine; this
copy reads them from the environment and records the served artifact's identity on every record.
The flow itself is unchanged.

Usage:
    python eval/run_arm_h3.py <label> <port>

Environment:
    CHTD_RUN_ROOT           task set + outputs root  [<repo>/data/holdout-3-20261014]
    CHTD_EVAL_HOST          host serving the arm     [127.0.0.1]
    CHTD_PROFILE_TEMPLATE   profile to expand        [<repo>/data/teacher-round-20261012/...]
    CHTD_PROFILE_STUDENT_URL  the URL in that profile the sweep rewrites
                                                     [http://100.84.151.62:8003/v1]
    CHTD_AGNES_CLI          agnes CLI entry          [E:/Coding/agnes-harness/.../agnes.mjs]
    CHTD_NODE               node binary              [node]
    CHTD_MODEL_ARTIFACT_ID / CHTD_MODEL_ARTIFACT_SHA256
                            written to every record; set both when the arm is an adapter.

Layout produced:

    <CHTD_RUN_ROOT>/tasks.jsonl                          the task rows this sweep reads
    <CHTD_RUN_ROOT>/workspaces/<task_id>/                the fixture sources (checked in)
    <CHTD_RUN_ROOT>/arm-<label>/<task_id>/work/          the workspace that session ran in
    <CHTD_RUN_ROOT>/arm-<label>/<task_id>/_agh_home/     the ledger holding that one session
    <CHTD_RUN_ROOT>/records-<label>.json                 every run, with its artifact identity
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

CHTD = Path(__file__).resolve().parent.parent
AGNES = Path(os.environ.get(
    "CHTD_AGNES_CLI", r"E:\Coding\agnes-harness\packages\cli\dist\local\agnes.mjs"))
NODE = os.environ.get("CHTD_NODE", "node")
ROOT = Path(os.environ.get("CHTD_RUN_ROOT", str(CHTD / "data" / "holdout-3-20261014")))
PROFILE_TEMPLATE = Path(os.environ.get("CHTD_PROFILE_TEMPLATE", str(
    CHTD / "data" / "teacher-round-20261012" / "agh_home" / "profiles" / "local-dev"
    / "profile.yaml")))
PROFILE_STUDENT_URL = os.environ.get(
    "CHTD_PROFILE_STUDENT_URL", "http://100.84.151.62:8003/v1")
HOST = os.environ.get("CHTD_EVAL_HOST", "127.0.0.1")
ARTIFACT_ID = os.environ.get("CHTD_MODEL_ARTIFACT_ID")
ARTIFACT_SHA256 = os.environ.get("CHTD_MODEL_ARTIFACT_SHA256")
STUDENT_MODEL = os.environ.get("CHTD_STUDENT_MODEL", "primary=student/qwen3-4b-instruct-2507")
PRESET = os.environ.get("CHTD_PRESET", "minimal-rl")
TASK_TIMEOUT = 180


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    label = sys.argv[1]
    port = int(sys.argv[2])

    work_root = ROOT / f"arm-{label}"
    records_out = ROOT / f"records-{label}.json"
    rows = [
        json.loads(line)
        for line in (ROOT / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    work_root.mkdir(parents=True, exist_ok=True)

    records = []
    for index, row in enumerate(rows, 1):
        task_id = row["id"]
        base = work_root / task_id
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        workspace = base / "work"
        home = base / "_agh_home"
        (home / "profiles" / "local-dev").mkdir(parents=True)
        (home / "data").mkdir(parents=True)
        text = PROFILE_TEMPLATE.read_text(encoding="utf-8")
        text = text.replace(PROFILE_STUDENT_URL, f"http://{HOST}:{port}/v1")
        (home / "profiles" / "local-dev" / "profile.yaml").write_text(
            text, encoding="utf-8", newline="\n")
        shutil.copytree(ROOT / "workspaces" / task_id, workspace)

        env = dict(os.environ)
        env.update({
            "AGH_HOME": str(home),
            "AGNES_PROFILE": "local-dev",
            "AGNES_SECRET_STUDENT_DEFAULT": "local-no-auth",
            "AGNES_SECRET_TEACHER_DEFAULT": "sk-not-used-in-this-run",
        })
        argv = [NODE, str(AGNES), "-p", "--standalone", "--preset", PRESET,
                "--model", STUDENT_MODEL, row["prompt"]]
        began = time.time()
        timed_out = False
        try:
            completed = subprocess.run(argv, cwd=str(workspace), env=env, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace",
                                       timeout=TASK_TIMEOUT)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = -1
            timed_out = True
        elapsed = round(time.time() - began, 1)

        ledger = home / "data" / "sessions.db"
        session = None
        if ledger.exists():
            with sqlite3.connect(ledger) as connection:
                r = connection.execute(
                    "SELECT session_key FROM sessions ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                session = r[0] if r else None

        records.append({
            "task_id": task_id,
            "exit": exit_code,
            "timed_out": timed_out,
            "seconds": elapsed,
            "session": session,
            "ledger": str(ledger),
            "workspace": str(workspace),
            "artifact_id": ARTIFACT_ID,
            "artifact_sha256": ARTIFACT_SHA256,
        })
        print(f"[{index}/{len(rows)}] {task_id:32} exit={exit_code} {elapsed}s", flush=True)

    records_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {records_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

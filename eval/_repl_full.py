"""Run one seed's full h1 evaluation in the foreground; called from a shell with a long timeout.

The lesson from two failed attempts: a child that outlives its shell gets taken down with it on
Windows. So this process IS the shell's foreground work, and the shell's timeout is set longer
than the evaluation takes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CHTD = Path(r"E:\Coding\chtd")
sys.path.insert(0, str(CHTD))
sys.path.insert(0, str(CHTD / "eval"))

import holdout_sweep as hs  # noqa: E402

seed = int(sys.argv[1])
port = int(sys.argv[2])
arm = f"chtd-s{seed}"
run_name = f"chtd-repl-seed{seed}-20261014"
root = hs.RUN_ROOT / run_name
root.mkdir(parents=True, exist_ok=True)

tasks = hs.load_tasks()
out = root / "records.jsonl"
done = set()
if out.exists():
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["task_id"])

home = Path(os.environ["CHTD_SHARED_AGH_HOME"])
print(f"seed={seed} arm={arm} port={port} home={home}", flush=True)
print(f"tasks: {len(tasks)}, already done: {len(done)}", flush=True)

with out.open("a", encoding="utf-8") as journal:
    for task in tasks:
        if task["id"] in done:
            continue
        rec = hs.run_one(arm, port, task, root, home)
        journal.write(json.dumps(rec, ensure_ascii=False) + "\n")
        journal.flush()
        print(f"  {task['id']:28s} passed={rec['passed']} {rec['seconds']}s", flush=True)

print(f"seed{seed} evaluation complete", flush=True)

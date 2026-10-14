"""Extract every base-arm answer run alongside the dose ladder, to measure the base's own spread.

The dose analysis compares 7 trained points against a single base evaluation. If the base's own
evaluation-to-evaluation spread is as large as the dose effect, the "7/7 beat the base" reading is
an artifact of one unlucky base run rather than evidence about training. Every teacher-demo run
carried its own base column on the same 15 tasks, so the spread is measurable.
"""
from __future__ import annotations

import json
from pathlib import Path

CHTD = Path(r"E:\Coding\chtd")
OUT = CHTD / "runs" / "holdout-4-20261014"
PROMPT = ("Something in this directory is broken. Read the files and tell me what the defect is. "
          "Do not run any commands and do not change any files - I want to know what is wrong first.")

BASE_RUNS = [
    ("teacher-demo-20261012-bf16",    "base-v1run"),
    ("teacher-demo-v2-20261012-bf16", "base-v2run"),
    ("teacher-demo-v2b-20261012-bf16", "base-v2brun"),
    ("teacher-demo-v3-20261012-bf16", "base-v3run"),
    ("2arm-v4c-20261012",             "base-v4run"),
]

rows: list[dict] = []
for d, label in BASE_RUNS:
    p = CHTD / "data" / "rollouts" / d / "records.jsonl"
    if not p.exists():
        print(f"MISSING {p}")
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") != "base":
            continue
        rows.append({
            "set": "h1", "arm": label, "task": r["task_id"],
            "answer": (r.get("answer") or "").strip(),
            "fixture_dir": str(Path(r["workspace"])),
            "prompt": PROMPT,
            "passed_gate": bool(r.get("passed")),
            "dose_rows": 0,
            "dose_note": f"base arm from {d}",
        })

print(f"extracted {len(rows)} base cells across {len(BASE_RUNS)} runs")
print(f"without an answer: {sum(1 for r in rows if not r['answer'])}")
print(f"with a missing fixture dir: {sum(1 for r in rows if not Path(r['fixture_dir']).is_dir())}")

out = OUT / "base-replication-cells.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("wrote", out)

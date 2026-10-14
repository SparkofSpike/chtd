"""Extract the teacher-demo dose-ladder answers (v1..v4) into judge cells.

These arms all ran on the same 15-task holdout (h1) as the already-judged base and v5, so grading
them completes the dose-response curve: 0 / 33 / 63 / 83 / 93 rows, all on one task set.
"""
from __future__ import annotations

import json
from pathlib import Path

CHTD = Path(r"E:\Coding\chtd")
OUT = CHTD / "runs" / "holdout-4-20261014"
PROMPT = ("Something in this directory is broken. Read the files and tell me what the defect is. "
          "Do not run any commands and do not change any files - I want to know what is wrong first.")

# (run dir, label, training rows, notes)
DOSE_ARMS = [
    ("teacher-demo-20261012-bf16",    "v1",  33, "33 demonstrations, 1/task"),
    ("teacher-demo-v2-20261012-bf16", "v2",  63, "63 rows over 34 tasks"),
    ("teacher-demo-v2b-20261012-bf16", "v2b", 63, "same adapter as v2, a second evaluation"),
    ("2arm-v4c-20261012",             "v4",  63, "v2 data, seed 1"),
    ("teacher-demo-v3-20261012-bf16", "v3",  83, "83 rows over 34 tasks"),
]

rows: list[dict] = []
for d, label, dose, note in DOSE_ARMS:
    p = CHTD / "data" / "rollouts" / d / "records.jsonl"
    if not p.exists():
        print(f"MISSING {p}")
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") not in ("adapter", "v4"):
            continue  # skip the base rows: base is already judged from base-h1
        rows.append({
            "set": "h1", "arm": f"{label}-h1", "task": r["task_id"],
            "answer": (r.get("answer") or "").strip(),
            "fixture_dir": str(Path(r["workspace"])),
            "prompt": PROMPT,
            "passed_gate": bool(r.get("passed")),
            "dose_rows": dose,
            "dose_note": note,
        })

print(f"extracted {len(rows)} cells")
empty = [r for r in rows if not r["answer"]]
print(f"without an answer: {len(empty)}")
missing = [r for r in rows if not Path(r["fixture_dir"]).is_dir()]
print(f"with a missing fixture dir: {len(missing)}")
for r in missing[:5]:
    print("  ", r["arm"], r["task"], r["fixture_dir"])

out = OUT / "dose-cells.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("wrote", out)

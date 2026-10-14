"""Extract the CHTD replication seeds' h1 answers (main run + any fill-in run) into judge cells."""
from __future__ import annotations

import json
from pathlib import Path

CHTD = Path(r"E:\Coding\chtd")
OUT = CHTD / "runs" / "holdout-4-20261014"
PROMPT = ("Something in this directory is broken. Read the files and tell me what the defect is. "
          "Do not run any commands and do not change any files - I want to know what is wrong first.")

SEEDS = [3, 4, 5]  # the new seeds; 0-2 were extracted by the batch-A path below
TOTAL_TASKS = 15

rows: list[dict] = []
for s in range(0, 6):
    cells_for_seed: dict[str, dict] = {}
    for suffix in ("20261014", "fill-20261014"):
        run = CHTD / "data" / "rollouts" / f"chtd-repl-seed{s}-{suffix}" / "records.jsonl"
        if not run.exists():
            continue
        for line in run.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cells_for_seed[r["task_id"]] = r
    if not cells_for_seed:
        print(f"seed{s}: no records")
        continue
    print(f"seed{s}: {len(cells_for_seed)}/{TOTAL_TASKS} tasks")
    for tid, r in sorted(cells_for_seed.items()):
        rows.append({
            "set": "h1",
            "arm": f"chtd-s{s}-h1",
            "task": tid,
            "answer": (r.get("answer") or "").strip(),
            "fixture_dir": str(Path(r["workspace"])),
            "prompt": PROMPT,
            "passed_gate": bool(r.get("passed")),
            "dose_rows": 9,
            "dose_note": f"CHTD student-state, replication seed {s}",
        })

print(f"\nextracted {len(rows)} cells")
print(f"without an answer: {sum(1 for r in rows if not r['answer'])}")
missing_fx = [r for r in rows if not Path(r["fixture_dir"]).is_dir()]
print(f"with a missing fixture dir: {len(missing_fx)}")
for r in missing_fx[:5]:
    print("   ", r["arm"], r["task"], r["fixture_dir"])

out = OUT / "repl-cells.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("wrote", out)

"""Two kinds of replication, measured separately.

  * base: six evaluations produced byte-identical answers on all 15 tasks (deterministic serving),
    so the differences in their judge scores are *judge* noise, nothing else.
  * an adapter: v2 and v2b are the same adapter evaluated twice; their answers differ, so that
    pair measures *sampling* noise introduced by the fine-tune.

The two numbers set the floor for every trained-vs-base claim in the project.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}


def load_records(rel: str, arm_filter: set[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    p = CHTD / rel
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") in arm_filter:
            out.setdefault(r["arm"], {})[r["task_id"]] = (r.get("answer") or "").strip()
    return out


# ---- base: same answer every run? ----------------------------------------------------
BASE_RUNS = [
    "data/rollouts/base-h1-20261013/records.jsonl",
    "data/rollouts/teacher-demo-20261012-bf16/records.jsonl",
    "data/rollouts/teacher-demo-v2-20261012-bf16/records.jsonl",
    "data/rollouts/teacher-demo-v2b-20261012-bf16/records.jsonl",
    "data/rollouts/teacher-demo-v3-20261012-bf16/records.jsonl",
    "data/rollouts/2arm-v4c-20261012/records.jsonl",
]
base_texts: dict[str, set[str]] = {}
for rel in BASE_RUNS:
    for arm, tasks in load_records(rel, {"base"}).items():
        for t, txt in tasks.items():
            base_texts.setdefault(t, set()).add(hashlib.sha256(txt.encode()).hexdigest())

n_base_identical = sum(1 for t, s in base_texts.items() if len(s) == 1)
print("=" * 78)
print("BASE: six evaluations, how many distinct answers per task")
print("=" * 78)
for t in sorted(base_texts)[:15]:
    print(f"  {t:30s} distinct answers: {len(base_texts[t])}")
print(f"\n  tasks where all six base runs agree byte-for-byte: "
      f"{n_base_identical}/{len(base_texts)}")

# ---- the adapter pair -----------------------------------------------------------------
v2 = load_records("data/rollouts/teacher-demo-v2-20261012-bf16/records.jsonl", {"adapter"})["adapter"]
v2b = load_records("data/rollouts/teacher-demo-v2b-20261012-bf16/records.jsonl", {"adapter"})["adapter"]

print()
print("=" * 78)
print("THE ADAPTER (v2 vs v2b, same weights, two evaluations)")
print("=" * 78)
common = sorted(set(v2) & set(v2b))
identical = sum(1 for t in common if v2[t] == v2b[t])
print(f"  byte-identical answers: {identical}/{len(common)}")
for t in common:
    if v2[t] != v2b[t]:
        print(f"  {t:30s} differs "
              f"(len {len(v2[t])} vs {len(v2b[t])})")

# ---- judge noise on identical text ----------------------------------------------------
print()
print("=" * 78)
print("JUDGE NOISE on identical answer text")
print("=" * 78)
judge_files = [
    "runs/holdout-4-20261014/h1h2-judge-20261014.json",
    "runs/holdout-4-20261014/base-replication-judge-20261014.json",
]
grades: dict[tuple[str, str], list[str]] = {}
for rel in judge_files:
    d = json.loads((CHTD / rel).read_text(encoding="utf-8"))
    for c in d["cells"]:
        if c["set"] != "h1" or not c["arm"].startswith("base"):
            continue
        grades.setdefault((c["task"], c["llm"]), [])
        grades[(c["task"])].append(c["llm"]) if False else None

# group by task, list the grades from each judge run
per_task: dict[str, list[str]] = {}
for rel in judge_files:
    d = json.loads((CHTD / rel).read_text(encoding="utf-8"))
    for c in d["cells"]:
        if c["set"] != "h1" or not c["arm"].startswith("base"):
            continue
        per_task.setdefault(c["task"], []).append(c["llm"])

disagree = 0
for t in sorted(per_task):
    gs = per_task[t]
    if len(set(gs)) > 1:
        disagree += 1
        print(f"  {t:30s} grades across runs: {gs}")
print(f"\n  tasks where the judge gave different grades to identical text: "
      f"{disagree}/{len(per_task)}  -> judge noise ~= {disagree} cell(s) per 15")

"""The dose-response curve: how much capability does each increment of teacher-demo data buy?

All points below ran on the same 15-task holdout (h1) with the same frozen track and recipe, so
rows-per-arm is the only axis that changes (seeds and row selection vary within a dose, and that
variation is reported rather than averaged away). Scores are the deepseek-flash judge verdicts on
the shared three-level rubric; gate scores are carried alongside for comparison.

Two things this measures that the earlier records could not:
  * where the curve rises, saturates, or moves backwards;
  * the evaluation noise floor - v2 and v2b are the *same adapter* evaluated twice, so their
    spread bounds what a re-run of one arm can do without any training change.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}

judged = json.loads((CHTD / "runs/holdout-4-20261014/h1h2-judge-20261014.json").read_text(encoding="utf-8"))["cells"]
dose = json.loads((CHTD / "runs/holdout-4-20261014/dose-judge-20261014.json").read_text(encoding="utf-8"))["cells"]

rows = list(dose) + [c for c in judged if c["set"] == "h1"]

# (label, rows of training data, note)
POINTS = [
    ("base-h1",  0,  "unadapted"),
    ("v1-h1",   33,  "33 demos, 1/task"),
    ("v2-h1",   63,  "63 rows"),
    ("v2b-h1",  63,  "same adapter as v2, re-evaluated"),
    ("v4-h1",   63,  "v2 data, seed 1"),
    ("v3-h1",   83,  "83 rows"),
    ("v5-h1",   93,  "93 rows"),
    ("v5b-h1",  93,  "93 rows, second evaluation"),
]

by_arm: dict[str, dict[str, dict]] = {}
for r in rows:
    by_arm.setdefault(r["arm"], {})[r["task"]] = r

print("=" * 92)
print("DOSE-RESPONSE ON THE 15-TASK HOLDOUT (judge verdicts; gate shown alongside)")
print("=" * 92)
print(f"{'arm':10s} {'rows':>4s}  {'ordinal mean':>12s}  {'sum':>7s}  "
      f"{'C/P/F':>9s}  {'gate':>5s}  note")
print("-" * 92)

curve = []
for label, dose_rows, note in POINTS:
    cells = by_arm.get(label)
    if not cells:
        print(f"{label:10s} {dose_rows:4d}  (missing)")
        continue
    grades = [c["llm"] for c in cells.values()]
    vals = [SCORE.get(g, 0) for g in grades]
    cnt = Counter(grades)
    gate = sum(1 for c in cells.values() if c.get("passed_gate"))
    m = mean(vals)
    curve.append((label, dose_rows, m, note))
    print(f"{label:10s} {dose_rows:4d}  {m:12.3f}  {sum(vals):3d}/{2*len(vals):<3d}  "
          f"{cnt.get('correct',0):2d}/{cnt.get('partial',0):2d}/{cnt.get('failed',0):2d}  "
          f"{gate:2d}/{len(cells):<2d}  {note}")

print()
print("=" * 92)
print("MONOTONICITY AND NOISE")
print("=" * 92)
seen = [(l, d, m) for l, d, m, _ in curve]
by_dose: dict[int, list[float]] = {}
for l, d, m in seen:
    by_dose.setdefault(d, []).append(m)
print("dose -> ordinal means at that dose")
for d in sorted(by_dose):
    vals = by_dose[d]
    spread = f"  spread {max(vals)-min(vals):.3f}" if len(vals) > 1 else ""
    print(f"  {d:3d} rows: {[round(v,3) for v in vals]}{spread}")

# the same-adapter pair is the noise floor
if "v2-h1" in by_arm and "v2b-h1" in by_arm:
    v2 = by_arm["v2-h1"]; v2b = by_arm["v2b-h1"]
    common = sorted(set(v2) & set(v2b))
    deltas = [SCORE.get(v2[t]["llm"], 0) - SCORE.get(v2b[t]["llm"], 0) for t in common]
    changed = sum(1 for d in deltas if d != 0)
    print(f"\nSAME ADAPTER, TWO EVALUATIONS (v2 vs v2b): "
          f"{changed}/{len(common)} cells differ, mean delta {mean(deltas):+.3f}")
    for t in common:
        a, b = v2[t]["llm"], v2b[t]["llm"]
        if a != b:
            print(f"   {t:28s} v2={a:8s} v2b={b}")

# relative to the base
if curve:
    base_m = dict((l, m) for l, d, m, _ in curve).get("base-h1")
    print(f"\nvs base (ordinal mean {base_m:.3f}):")
    for l, d, m, _ in curve:
        if l == "base-h1":
            continue
        print(f"   {l:10s} {d:3d} rows  delta={m - base_m:+.3f}")

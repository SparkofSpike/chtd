"""The dose-ladder statistics: every trained point against the base, and the noise floor.

The curve is not monotone, and one pair of points is the *same adapter evaluated twice*. That pair
measures the evaluation noise directly, which turns out to be the same size as the dose effect.
The tests below are therefore reported with that floor in view.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}

judged = json.loads((CHTD / "runs/holdout-4-20261014/h1h2-judge-20261014.json").read_text(encoding="utf-8"))["cells"]
dose = json.loads((CHTD / "runs/holdout-4-20261014/dose-judge-20261014.json").read_text(encoding="utf-8"))["cells"]

by_arm: dict[str, dict[str, dict]] = {}
for r in list(dose) + [c for c in judged if c["set"] == "h1"]:
    by_arm.setdefault(r["arm"], {})[r["task"]] = r

POINTS = [
    ("base-h1", 0, "unadapted"),
    ("chtd-h1", 9, "CHTD student-state"),
    ("v1-h1", 33, "33 demos, 1/task"),
    ("v2-h1", 63, "63 rows"),
    ("v2b-h1", 63, "same adapter as v2, re-evaluated"),
    ("v4-h1", 63, "v2 data, seed 1"),
    ("v3-h1", 83, "83 rows"),
    ("v5-h1", 93, "93 rows"),
    ("v5b-h1", 93, "93 rows, second evaluation"),
]


def score(arm: str, task: str) -> int | None:
    c = by_arm.get(arm, {}).get(task)
    return SCORE.get(c["llm"], 0) if c else None


base = {t: score("base-h1", t) for t in by_arm["base-h1"]}

print("=" * 88)
print("EVERY POINT vs THE BASE (task-level paired, 15 tasks)")
print("=" * 88)
print(f"{'arm':10s} {'rows':>4s}  {'delta':>7s}  {'W/L/T':>7s}  {'sign p':>8s}  note")
print("-" * 88)

results = []
for label, rows, note in POINTS:
    if label == "base-h1":
        continue
    cells = by_arm.get(label)
    if not cells:
        continue
    common = sorted(set(cells) & set(base))
    d = [SCORE.get(cells[t]["llm"], 0) - base[t] for t in common]
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    n_eff = w + l
    p = min(1.0, 2 * sum(math.comb(n_eff, i) for i in range(min(w, l) + 1)) / 2 ** n_eff) if n_eff else 1.0
    m = mean(d)
    results.append((label, rows, m, w, l, note))
    print(f"{label:10s} {rows:4d}  {m:+7.3f}  {w:2d}/{l:2d}/{len(common)-w-l:2d}  {p:8.3f}  {note}")

print()
print("=" * 88)
print("POOLED: all trained points vs the base")
print("=" * 88)
all_d = []
for label, rows, m, w, l, note in results:
    if label == "v2b-h1":
        continue  # the duplicate evaluation of v2, excluded from the pooled count
    cells = by_arm[label]
    common = sorted(set(cells) & set(base))
    all_d.extend(SCORE.get(cells[t]["llm"], 0) - base[t] for t in common)
n = len(all_d)
m = mean(all_d)
w = sum(1 for x in all_d if x > 0)
l = sum(1 for x in all_d if x < 0)
sd = (sum((x - m) ** 2 for x in all_d) / (n - 1)) ** 0.5
se = sd / math.sqrt(n)
print(f"  n={n} pair-cells over {len(results) - 1} distinct trained points")
print(f"  mean delta = {m:+.4f}  sd = {sd:.3f}  se = {se:.4f}  t = {m / se:+.2f}")
print(f"  W{w}/L{l}/T{n - w - l}  sign test p = "
      f"{min(1.0, 2 * sum(math.comb(w + l, i) for i in range(min(w, l) + 1)) / 2 ** (w + l)):.5f}")
# per-point wins: how many of the distinct points beat the base
beats = sum(1 for label, rows, mm, w, l, note in results if label != "v2b-h1" and mm > 0)
print(f"  trained points with a positive mean delta: {beats}/{len(results) - 1}")

print()
print("=" * 88)
print("THE NOISE FLOOR: same adapter, two evaluations (v2 vs v2b)")
print("=" * 88)
v2, v2b = by_arm["v2-h1"], by_arm["v2b-h1"]
common = sorted(set(v2) & set(v2b))
d = [SCORE.get(v2[t]["llm"], 0) - SCORE.get(v2b[t]["llm"], 0) for t in common]
print(f"  mean delta = {mean(d):+.3f}  (cells differing: {sum(1 for x in d if x) }/{len(common)})")
print(f"  -> a re-run of one unchanged arm moves the 15-task ordinal score by up to "
      f"{abs(mean(d)):.3f} points, which is the same size as the dose differences above.")

print()
print("=" * 88)
print("DATA EFFICIENCY: CHTD (9 student-state rows) against the teacher-demo ladder")
print("=" * 88)
chtd_m = mean([SCORE.get(by_arm['chtd-h1'][t]['llm'], 0) for t in by_arm['chtd-h1']])
base_m = mean(list(base.values()))
print(f"  base (0 rows)            {base_m:.3f}")
print(f"  CHTD (9 student rows)    {chtd_m:.3f}   delta {chtd_m - base_m:+.3f}")
for label, rows, note in POINTS:
    if label in ("base-h1",):
        continue
    cells = by_arm[label]
    mv = mean([SCORE.get(cells[t]["llm"], 0) for t in cells])
    print(f"  {label:10s} ({rows:2d} rows)      {mv:.3f}   delta {mv - base_m:+.3f}  {note}")

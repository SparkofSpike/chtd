"""Is the trained-vs-base gap larger than the base's own evaluation-to-evaluation spread?

The dose ladder showed 7/7 trained points above one base evaluation. Six base evaluations now exist
on the same 15 tasks (the holdout-3-era base-h1 plus five base columns that ran alongside the
teacher-demo arms). If the base's own spread covers the gap, the ladder's positive direction is
about evaluation luck; if it does not, the direction is about training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}

paths = [
    "runs/holdout-4-20261014/h1h2-judge-20261014.json",
    "runs/holdout-4-20261014/dose-judge-20261014.json",
    "runs/holdout-4-20261014/base-replication-judge-20261014.json",
]
cells: list[dict] = []
for rel in paths:
    d = json.loads((CHTD / rel).read_text(encoding="utf-8"))
    cells.extend(d["cells"])

by_arm: dict[str, dict[str, dict]] = {}
for r in cells:
    if r["set"] != "h1":
        continue
    by_arm.setdefault(r["arm"], {})[r["task"]] = r


def m(arm: str) -> float:
    return mean(SCORE.get(c["llm"], 0) for c in by_arm[arm].values())


def frac_failed(arm: str) -> float:
    v = [c["llm"] for c in by_arm[arm].values()]
    return sum(1 for g in v if g == "failed") / len(v)


print("=" * 80)
print("THE BASE'S OWN SPREAD (six evaluations of the unadapted model, same 15 tasks)")
print("=" * 80)
bases = sorted(a for a in by_arm if a.startswith("base"))
base_scores = []
for a in bases:
    s = m(a)
    base_scores.append(s)
    print(f"  {a:14s} ordinal mean {s:.3f}   failed {frac_failed(a) * 100:5.1f}%  "
          f"gate {sum(1 for c in by_arm[a].values() if c.get('passed_gate'))}/15")
print(f"\n  base mean {mean(base_scores):.3f}  sd {stdev(base_scores):.3f}  "
      f"range [{min(base_scores):.3f}, {max(base_scores):.3f}]")

print()
print("=" * 80)
print("TRAINED POINTS against that spread")
print("=" * 80)
POINTS = [("chtd-h1", 9, "CHTD student-state"), ("v1-h1", 33, "33 demos"), ("v2-h1", 63, "63 rows"),
          ("v2b-h1", 63, "63 rows, re-eval"), ("v4-h1", 63, "v2 data seed 1"), ("v3-h1", 83, "83 rows"),
          ("v5-h1", 93, "93 rows"), ("v5b-h1", 93, "93 rows, re-eval")]
trained_scores = []
for a, rows, note in POINTS:
    s = m(a)
    trained_scores.append(s)
    above_all_base = s > max(base_scores)
    print(f"  {a:10s} ({rows:2d} rows) {s:.3f}  failed {frac_failed(a) * 100:5.1f}%  "
          f"above every base evaluation: {above_all_base}")

print(f"\n  trained mean {mean(trained_scores):.3f}  sd {stdev(trained_scores):.3f}  "
      f"range [{min(trained_scores):.3f}, {max(trained_scores):.3f}]")

# how many trained points exceed the best base evaluation?
n_above = sum(1 for s in trained_scores if s > max(base_scores))
print(f"\n  trained points above the BEST base evaluation: {n_above}/{len(trained_scores)}")

# separation between the two distributions
mb, mt = mean(base_scores), mean(trained_scores)
sb = stdev(base_scores)
print(f"\n  base mean {mb:.3f} (sd {sb:.3f})  vs  trained mean {mt:.3f}")
print(f"  gap {mt - mb:+.3f}; base sd {sb:.3f} -> the gap is "
      f"{(mt - mb) / sb:.2f} base-sds")

# failure-rate view on the same data
print()
print("=" * 80)
print("FAILURE RATE: base evaluations vs trained points")
print("=" * 80)
bf = [frac_failed(a) for a in bases]
print(f"  base:    {[f'{x * 100:.0f}%' for x in bf]}  mean {mean(bf) * 100:.1f}%")
tf = [frac_failed(a) for a, _, _ in POINTS]
print(f"  trained: {[f'{x * 100:.0f}%' for x in tf]}  mean {mean(tf) * 100:.1f}%")

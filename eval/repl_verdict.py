"""The replication's verdict: CHTD's six seeds against the base, the 33-row point, and each other.

Question the run exists to answer: `dose-ladder-20261014.md` claimed CHTD reaches the 33-row
teacher-demo point's failure count on 9 rows, from a single evaluation. With six seeds the claim
gets a distribution - and this script states which half of it survives.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}

cells: list[dict] = []
for rel in ("runs/holdout-4-20261014/h1h2-judge-20261014.json",
            "runs/holdout-4-20261014/dose-judge-20261014.json",
            "runs/holdout-4-20261014/base-replication-judge-20261014.json",
            "runs/holdout-4-20261014/repl-judge-20261014.json"):
    p = CHTD / rel
    if p.exists():
        cells.extend(json.loads(p.read_text(encoding="utf-8"))["cells"])

by_arm: dict[str, dict[str, dict]] = {}
for r in cells:
    if r["set"] == "h1":
        by_arm.setdefault(r["arm"], {})[r["task"]] = r

SEEDS = [f"chtd-s{s}-h1" for s in range(6)]
BASE = "base-h1"
V1 = "v1-h1"


def vec(arm: str) -> dict[str, int]:
    return {t: SCORE.get(c["llm"], 0) for t, c in by_arm[arm].items()}


def fails(arm: str) -> int:
    return sum(1 for c in by_arm[arm].values() if c["llm"] == "failed")


def ordinal(arm: str) -> float:
    return mean(vec(arm).values())


print("=" * 88)
print("THE SIX CHTD SEEDS vs THE BASE (paired, 15 tasks each)")
print("=" * 88)
base = vec(BASE)
base_f = fails(BASE)
all_d = []
for s in SEEDS:
    v = vec(s)
    common = sorted(set(v) & set(base))
    d = [v[t] - base[t] for t in common]
    all_d.extend(d)
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    print(f"  {s:12s} ordinal {ordinal(s):.3f}  failures {fails(s):2d}  "
          f"delta vs base {mean(d):+.3f}  W{w}/L{l}/T{len(d)-w-l}")

n = len(all_d)
m = mean(all_d)
sd = (sum((x - m) ** 2 for x in all_d) / (n - 1)) ** 0.5
se = sd / math.sqrt(n)
print(f"\n  pooled over {n} pair-cells: mean delta {m:+.4f}  sd {sd:.3f}  se {se:.4f}  "
      f"t {m / se:+.2f}")

# failure counts: how many seeds strictly below the base?
below = sum(1 for s in SEEDS if fails(s) < base_f)
print(f"  seeds with fewer failures than the base ({base_f}): {below}/6")

print()
print("=" * 88)
print("THE DATA-EFFICIENCY CLAIM, now with a distribution")
print("=" * 88)
v1_ord, v1_f = ordinal(V1), fails(V1)
print(f"  v1 (33 teacher-demo rows): ordinal {v1_ord:.3f}, failures {v1_f}/15")
chtd_ords = [ordinal(s) for s in SEEDS]
chtd_fails = [fails(s) for s in SEEDS]
print(f"  CHTD (9 rows), 6 seeds:    ordinal mean {mean(chtd_ords):.3f} "
      f"(sd {stdev(chtd_ords):.3f}), failures {sorted(chtd_fails)} (mean {mean(chtd_fails):.2f})")
print()
print(f"  failure endpoint: seeds at or below v1's {v1_f}: "
      f"{sum(1 for f in chtd_fails if f <= v1_f)}/6")
print(f"  ordinal endpoint: seeds at or above v1's {v1_ord:.3f}: "
      f"{sum(1 for o in chtd_ords if o >= v1_ord)}/6")

print()
print("=" * 88)
print("SEED SPREAD vs EVALUATION NOISE (the two floors measured tonight)")
print("=" * 88)
print(f"  across the six CHTD seeds:  sd {stdev(chtd_ords):.3f}  "
      f"range {min(chtd_ords):.3f}-{max(chtd_ords):.3f}")
print(f"  same adapter, re-evaluated (v2 vs v2b): 0.333 ordinal points")
print(f"  -> the seed-to-seed spread is {'SMALLER' if stdev(chtd_ords) < 0.333 else 'LARGER'} "
      f"than a single re-evaluation of one unchanged adapter")

print()
print("=" * 88)
print("CONSISTENCY CHECK: seed0 evaluated twice, months apart in process, same numbers?")
print("=" * 88)
if "chtd-h1" in by_arm:
    a, b = vec("chtd-h1"), vec("chtd-s0-h1")
    common = sorted(set(a) & set(b))
    agree = sum(1 for t in common if a[t] == b[t])
    print(f"  earlier run (chtd-h1) ordinal {ordinal('chtd-h1'):.3f}, failures {fails('chtd-h1')}")
    print(f"  tonight's   (chtd-s0) ordinal {ordinal('chtd-s0-h1'):.3f}, failures {fails('chtd-s0-h1')}")
    print(f"  per-cell agreement: {agree}/{len(common)}")

"""The final test: does training move the failure rate beyond the base's own sampling?

Setup, measured rather than assumed (see `replication_floor.py`):
  * the base is deterministic - six evaluations produced byte-identical answers, so "the base" is
    one point, and the only variation in its judge scores is judge noise (~1 cell per 15);
  * an adapter carries real sampling variance - the same weights evaluated twice differ on 9 of 15
    answers, worth up to 0.33 ordinal points on a 15-task score, i.e. the ceiling of any single-arm
    evaluation at this size.

The test therefore asks the conservative question: under the null that a trained arm's true failure
rate equals the base's, how likely is it that six independent training configurations (CHTD plus
five teacher-demo doses) all land at or below their observed failure counts?
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}

cells: list[dict] = []
for rel in ("runs/holdout-4-20261014/h1h2-judge-20261014.json",
            "runs/holdout-4-20261014/dose-judge-20261014.json",
            "runs/holdout-4-20261014/base-replication-judge-20261014.json"):
    cells.extend(json.loads((CHTD / rel).read_text(encoding="utf-8"))["cells"])

by_arm: dict[str, dict[str, dict]] = {}
for r in cells:
    if r["set"] == "h1":
        by_arm.setdefault(r["arm"], {})[r["task"]] = r

N_TASKS = 15


def fails(arm: str) -> int:
    return sum(1 for c in by_arm[arm].values() if c["llm"] == "failed")


def ordinal(arm: str) -> float:
    return mean(SCORE.get(c["llm"], 0) for c in by_arm[arm].values())


BASE_ARMS = [a for a in by_arm if a.startswith("base")]
base_fails = sorted(fails(a) for a in BASE_ARMS)
base_ord = sorted(ordinal(a) for a in BASE_ARMS)
print("=" * 82)
print("THE BASE (six judge readings of ONE deterministic answer set)")
print("=" * 82)
print(f"  failures per reading: {base_fails}  -> modal {max(set(base_fails), key=base_fails.count)}/15")
print(f"  ordinal mean per reading: {[round(x, 3) for x in base_ord]}")
base_fail_hat = sum(base_fails) / len(base_fails) / N_TASKS
print(f"  pooled failure rate estimate: {base_fail_hat * 100:.1f}%")

TRAINED = [("chtd", "chtd-h1"), ("v1", "v1-h1"), ("v2", "v2-h1"),
           ("v2b (re-eval of v2)", "v2b-h1"), ("v4", "v4-h1"), ("v3", "v3-h1"),
           ("v5", "v5-h1"), ("v5b (re-eval of v5)", "v5b-h1")]
print()
print("=" * 82)
print("TRAINED ARMS")
print("=" * 82)
trained_fails = []
for label, arm in TRAINED:
    f = fails(arm)
    trained_fails.append((label, f))
    print(f"  {label:22s} failures {f:2d}/15 = {f / N_TASKS * 100:5.1f}%   ordinal {ordinal(arm):.3f}")


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


# the six independent configurations (drop the two re-evaluations)
independent = [("chtd", fails("chtd-h1")), ("v1", fails("v1-h1")), ("v2", fails("v2-h1")),
               ("v4", fails("v4-h1")), ("v3", fails("v3-h1")), ("v5", fails("v5-h1"))]

print()
print("=" * 82)
print("CONSERVATIVE JOINT TEST")
print("=" * 82)
print("  For each plausible base failure rate p, the chance that EVERY one of the six")
print("  independent configurations lands at or below its observed failure count:")
for p in (0.40, 0.333, 0.30, 0.267, 0.20):
    probs = [binom_cdf(k, N_TASKS, p) for _, k in independent]
    joint = 1.0
    for x in probs:
        joint *= x
    print(f"    p={p * 100:5.1f}%  per-arm P(X<=k) = "
          f"{[f'{x:.3f}' for x in probs]}  joint = {joint:.3e}")

print()
print("=" * 82)
print("ORDINAL VIEW, same conservative frame")
print("=" * 82)
best_base = max(base_ord)
n_above = sum(1 for _, arm in TRAINED if ordinal(arm) > best_base)
print(f"  best base reading: {best_base:.3f}")
print(f"  trained arms above it: {n_above}/{len(TRAINED)}")
print(f"  lowest trained arm: {min(ordinal(a) for _, a in TRAINED):.3f}")

print()
print("=" * 82)
print("DATA EFFICIENCY (the CHTD claim, stated in the terms the data supports)")
print("=" * 82)
chtd_f = fails("chtd-h1")
dose1 = fails("v1-h1")   # 33 rows
print(f"  CHTD: 9 student-state rows  -> {chtd_f}/15 failures, ordinal {ordinal('chtd-h1'):.3f}")
print(f"  v1:   33 teacher-demo rows  -> {dose1}/15 failures, ordinal {ordinal('v1-h1'):.3f}")
print(f"  v2:   63 teacher-demo rows  -> {fails('v2-h1')}/15 failures, ordinal {ordinal('v2-h1'):.3f}")
print("  -> CHTD reaches the 33-row point's failure count on 9 rows; the ordinal difference")
print("     between them is inside the measured sampling noise (0.33 at n=15).")

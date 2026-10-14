"""The CHTD replication: four seed evaluations on the h1 holdout.

The data-efficiency claim from `dose-ladder-20261014.md` rested on one student-state evaluation
(seed0: 2/15 failures, ordinal 1.600). This script puts the three new seeds beside it and asks
whether the claim survives the seed spread - and what the spread itself says about the recipe.
"""
from __future__ import annotations

import json
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
    if not p.exists():
        print(f"(skipping missing {rel})")
        continue
    cells.extend(json.loads(p.read_text(encoding="utf-8"))["cells"])

by_arm: dict[str, dict[str, dict]] = {}
for r in cells:
    if r["set"] == "h1":
        by_arm.setdefault(r["arm"], {})[r["task"]] = r


def stats(arm: str) -> tuple[float, int, int]:
    gs = [c["llm"] for c in by_arm[arm].values()]
    return mean(SCORE.get(g, 0) for g in gs), sum(1 for g in gs if g == "failed"), len(gs)


print("=" * 84)
print("CHTD STUDENT-STATE ON h1: one evaluation per seed")
print("=" * 84)
seed_arms = [(s, f"chtd-s{s}-h1") for s in (0, 1, 2, 3, 4, 5)]
present = []
for s, arm in seed_arms:
    if arm not in by_arm:
        print(f"  seed{s}: (not yet evaluated)")
        continue
    m, f, n = stats(arm)
    present.append((s, m, f, n))
    print(f"  seed{s}: ordinal {m:.3f}  failures {f}/{n}")

if present:
    ords = [m for _, m, _, _ in present]
    fails = [f for _, _, f, _ in present]
    print(f"\n  across {len(present)} seeds: ordinal mean {mean(ords):.3f} "
          f"(sd {stdev(ords):.3f}, range {min(ords):.3f}-{max(ords):.3f})")
    print(f"                        failures mean {mean(fails):.2f}/15 "
          f"(range {min(fails)}-{max(fails)})")

print()
print("=" * 84)
print("THE DATA-EFFICIENCY CLAIM, against its reference points")
print("=" * 84)
for label, arm, note in [("base", "base-h1", "0 rows"),
                         ("v1 (33 rows)", "v1-h1", "33 teacher-demo rows"),
                         ("v2 (63 rows)", "v2-h1", "63 rows"),
                         ("chtd seed0 (9 rows)", "chtd-h1", "original CHTD evaluation")]:
    if arm in by_arm:
        m, f, n = stats(arm)
        print(f"  {label:22s} ordinal {m:.3f}  failures {f}/{n}  ({note})")

if present:
    print()
    lo, hi = min(fails), max(fails)
    v1 = stats("v1-h1")[1] if "v1-h1" in by_arm else None
    if v1 is not None:
        print(f"  CHTD failure counts across seeds: {sorted(fails)}  vs v1's {v1}/15")
        n_le = sum(1 for f in fails if f <= v1)
        print(f"  seeds at or below the 33-row point: {n_le}/{len(fails)}")

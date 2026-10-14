"""The failure-rate view: how often did an arm produce a flatly wrong answer (grade=failed)?

The ordinal mean treats correct/partial/failed as 2/1/0. This looks at the bottom category alone,
which is the one that matters most for a diagnosis task: "failed" means the answer concluded there
was no defect, blamed the test, or named a mechanism the fixture does not contain.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

CHTD = Path(r"E:\Coding\chtd")

judged = json.loads((CHTD / "runs/holdout-4-20261014/h1h2-judge-20261014.json").read_text(encoding="utf-8"))["cells"]
hand = json.loads((CHTD / "runs/holdout-3-20261014/handread-20261014.json").read_text(encoding="utf-8"))["cells"]

buckets: dict[str, dict[str, dict[str, str]]] = {"h1": defaultdict(dict), "h2": defaultdict(dict), "h3": defaultdict(dict)}
for c in judged:
    buckets[c["set"]][c["arm"]][c["task"]] = c["llm"]
for arm, cells in hand.items():
    for task, cell in cells.items():
        buckets["h3"][arm][task] = cell["grade"]

FAMILY = {
    "base-h1": "base", "chtd-h1": "chtd", "ctrl-h1": "ctrl", "ctrl1-h1": "ctrl", "v5-h1": "v5", "v5b-h1": "v5",
    "base-h2a": "base", "base2-h2a": "base", "base-h2b": "base", "chtd-h2a": "chtd",
    "ctrl-h2a": "ctrl", "ctrl1-h2a": "ctrl", "v4-h2a": "v4", "v5-h2a": "v5", "v5b-h2a": "v5", "v5-h2b": "v5",
    "base": "base", "chtd-s0": "chtd", "chtd-s1": "chtd", "chtd-s2": "chtd",
    "ctrl-s0": "ctrl", "ctrl-s1": "ctrl", "v5": "v5",
}


def family_tasks(setname: str, fam: str) -> dict[str, list[str]]:
    """task -> list of grades across the arms of that family."""
    arms = [a for a in buckets[setname] if FAMILY.get(a) == fam]
    tasks = set.intersection(*[set(buckets[setname][a]) for a in arms]) if arms else set()
    return {t: [buckets[setname][a][t] for a in arms] for t in tasks}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


print("=" * 78)
print("FAILURE RATE (grade=failed), by set and family")
print("=" * 78)
tot = defaultdict(lambda: [0, 0])
for setname in ("h1", "h2", "h3"):
    for fam in ("base", "chtd", "ctrl", "v5"):
        ft = family_tasks(setname, fam)
        if not ft:
            continue
        n_tasks = len(ft)
        # an arm-family task counts as a failure if the majority of its arms failed
        fails = 0
        for t, grades in ft.items():
            n_fail = sum(1 for g in grades if g == "failed")
            if n_fail * 2 >= len(grades):
                fails += 1
        tot[fam][0] += fails
        tot[fam][1] += n_tasks
        print(f"  {setname} {fam:5s}: {fails:2d}/{n_tasks:2d} = {fails / n_tasks * 100:5.1f}%")
    print()

print("=" * 78)
print("POOLED FAILURE RATE")
print("=" * 78)
for fam in ("base", "chtd", "ctrl", "v5"):
    f, n = tot[fam]
    if n:
        print(f"  {fam:5s}: {f:2d}/{n:2d} = {f / n * 100:5.1f}%")

# paired McNemar on failure, CHTD vs base, task level
print()
print("=" * 78)
print("PAIRED: did the CHTD family fail where the base family did not (and vice versa)?")
print("=" * 78)
for setname in ("h1", "h2", "h3"):
    pass
allb = allc = 0
for setname in ("h1", "h2", "h3"):
    fb = family_tasks(setname, "base")
    fc = family_tasks(setname, "chtd")
    common = sorted(set(fb) & set(fc))
    b = c = 0
    for t in common:
        base_fail = sum(1 for g in fb[t] if g == "failed") * 2 >= len(fb[t])
        chtd_fail = sum(1 for g in fc[t] if g == "failed") * 2 >= len(fc[t])
        if base_fail and not chtd_fail:
            b += 1
        elif chtd_fail and not base_fail:
            c += 1
    allb += b
    allc += c
    print(f"  {setname}: base failed / chtd passed = {b};  chtd failed / base passed = {c}  "
          f"(n={len(common)})")
print(f"\n  POOLED: base-only failures {allb}, chtd-only failures {allc}")
print(f"  McNemar exact two-sided p = {mcnemar_exact(allb, allc):.4f}")

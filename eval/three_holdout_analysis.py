"""The three-holdout ordinal analysis: does the CHTD arm beat its base, and by how much?

h1 and h2 use the fresh deepseek-flash judge verdicts (this batch); h3 uses the author's full
human read (handread-20261014.json, 210 cells). Both are the same three-level rubric. The two
readers are different, which is stated wherever the numbers are reported.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}

# ---- h1 + h2: the fresh judge run -----------------------------------------------------
judged = json.loads((CHTD / "runs/holdout-4-20261014/h1h2-judge-20261014.json").read_text(encoding="utf-8"))["cells"]

# ---- h3: the human read ---------------------------------------------------------------
hand = json.loads((CHTD / "runs/holdout-3-20261014/handread-20261014.json").read_text(encoding="utf-8"))["cells"]

buckets: dict[str, dict[str, dict[str, str]]] = {"h1": defaultdict(dict), "h2": defaultdict(dict), "h3": defaultdict(dict)}
for c in judged:
    buckets[c["set"]][c["arm"]][c["task"]] = c["llm"]
for arm, cells in hand.items():
    for task, cell in cells.items():
        buckets["h3"][arm][task] = cell["grade"]

# ---- arm family mapping ---------------------------------------------------------------
FAMILY = {
    "base-h1": "base", "chtd-h1": "chtd", "ctrl-h1": "ctrl", "ctrl1-h1": "ctrl",
    "v5-h1": "v5", "v5b-h1": "v5",
    "base-h2a": "base", "base2-h2a": "base", "base-h2b": "base",
    "chtd-h2a": "chtd", "ctrl-h2a": "ctrl", "ctrl1-h2a": "ctrl",
    "v4-h2a": "v4", "v5-h2a": "v5", "v5b-h2a": "v5", "v5-h2b": "v5",
    "base": "base", "chtd-s0": "chtd", "chtd-s1": "chtd", "chtd-s2": "chtd",
    "ctrl-s0": "ctrl", "ctrl-s1": "ctrl", "v5": "v5",
}


def ordinal(arm_cells: dict[str, str]) -> tuple[int, int]:
    """(sum, n) over the ordinal scale."""
    vals = [SCORE.get(g, 0) for g in arm_cells.values()]
    return sum(vals), len(vals)


print("=" * 78)
print("PER-HOLDOUT ORDINAL TOTALS  (correct=2, partial=1, failed=0)")
print("=" * 78)
for setname in ("h1", "h2", "h3"):
    print(f"\n--- {setname} ---")
    for arm in sorted(buckets[setname]):
        s, n = ordinal(buckets[setname][arm])
        fam = FAMILY.get(arm, "?")
        raw = Counter(buckets[setname][arm].values())
        print(f"  {arm:12s} [{fam:4s}] sum={s:3d}/{2*n:<3d} mean={s/n:.3f}  "
              f"C{raw.get('correct',0):2d}/P{raw.get('partial',0):2d}/F{raw.get('failed',0):2d}")

# ---- family means per holdout ---------------------------------------------------------
print()
print("=" * 78)
print("FAMILY MEANS PER HOLDOUT  (averaged over arms in a family, then over tasks)")
print("=" * 78)
for setname in ("h1", "h2", "h3"):
    fam_scores: dict[str, list[float]] = defaultdict(list)
    for arm, cells in buckets[setname].items():
        s, n = ordinal(cells)
        fam_scores[FAMILY.get(arm, "?")].append(s / n)
    print(f"\n--- {setname} ---")
    for fam in ("base", "chtd", "ctrl", "v4", "v5"):
        if fam in fam_scores:
            vals = fam_scores[fam]
            print(f"  {fam:5s} mean={sum(vals)/len(vals):.3f}  arms={[round(v,3) for v in vals]}")

# ---- paired task-level comparison within a holdout ------------------------------------
print()
print("=" * 78)
print("PAIRED TASK-LEVEL: family A vs family B, per holdout (delta = A - B in ordinal points)")
print("=" * 78)


def family_vec(setname: str, fam: str) -> dict[str, float] | None:
    arms = [a for a in buckets[setname] if FAMILY.get(a) == fam]
    if not arms:
        return None
    tasks = set.intersection(*[set(buckets[setname][a]) for a in arms])
    return {t: sum(SCORE.get(buckets[setname][a][t], 0) for a in arms) / len(arms) for t in tasks}


def compare(setname: str, a: str, b: str) -> None:
    va, vb = family_vec(setname, a), family_vec(setname, b)
    if va is None or vb is None:
        return
    common = sorted(set(va) & set(vb))
    if not common:
        return
    deltas = [va[t] - vb[t] for t in common]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    mean = sum(deltas) / len(deltas)
    # paired sign test (two-sided, exact)
    n_eff = wins + losses
    if n_eff:
        from math import comb
        k = min(wins, losses)
        p = min(1.0, 2 * sum(comb(n_eff, i) for i in range(k + 1)) / (2 ** n_eff))
    else:
        p = 1.0
    print(f"  {setname}: {a:5s} vs {b:5s}  n={len(common):2d}  mean_delta={mean:+.3f}  "
          f"W{wins}/L{losses}/T{ties}  sign-p={p:.3f}")


for setname in ("h1", "h2", "h3"):
    for a, b in (("chtd", "base"), ("ctrl", "base"), ("v5", "base"), ("chtd", "ctrl")):
        compare(setname, a, b)
    print()

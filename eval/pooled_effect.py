"""Pooled effect of the CHTD family against its base across all three held-out sets.

h1 (15 tasks), h2 (12), h3 (30) were run at different times with different judges for h1/h2
(fresh deepseek-flash) and h3 (the author's human read). The arms within a family differ too
(CHTD is 1 arm on h1, 1 on h2, 3 seeds on h3; base is 1, 3 and 1). Pooling is therefore reported
with both the per-set breakdown and the pooled number, and the sign test is the headline because
it needs no distributional assumption and is robust to the judge difference.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

CHTD = Path(r"E:\Coding\chtd")
SCORE = {"correct": 2, "partial": 1, "failed": 0, "no-answer": 0}

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
    "ctrl-h2a": "ctrl", "ctrl1-h2a": "ctrl", "v4-h2a": "v4",
    "v5-h2a": "v5", "v5b-h2a": "v5", "v5-h2b": "v5",
    "base": "base", "chtd-s0": "chtd", "chtd-s1": "chtd", "chtd-s2": "chtd",
    "ctrl-s0": "ctrl", "ctrl-s1": "ctrl", "v5": "v5",
}


def family_vec(setname: str, fam: str) -> dict[str, float]:
    arms = [a for a in buckets[setname] if FAMILY.get(a) == fam]
    tasks = set.intersection(*[set(buckets[setname][a]) for a in arms])
    return {t: mean(SCORE.get(buckets[setname][a][t], 0) for a in arms) for t in tasks}


def pooled(a: str, b: str) -> None:
    print("=" * 78)
    print(f"POOLED: {a} vs {b}")
    print("=" * 78)
    all_deltas: list[float] = []
    per_set: dict[str, list[float]] = {}
    for setname in ("h1", "h2", "h3"):
        va, vb = family_vec(setname, a), family_vec(setname, b)
        common = sorted(set(va) & set(vb))
        deltas = [va[t] - vb[t] for t in common]
        per_set[setname] = deltas
        all_deltas.extend(deltas)
        w = sum(1 for d in deltas if d > 0)
        l = sum(1 for d in deltas if d < 0)
        t_ = sum(1 for d in deltas if d == 0)
        print(f"  {setname}: n={len(deltas):2d}  mean={mean(deltas):+.3f}  W{w}/L{l}/T{t_}")

    n = len(all_deltas)
    m = mean(all_deltas)
    sd = stdev(all_deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    print(f"\n  POOLED n={n}  mean_delta={m:+.4f}  sd={sd:.3f}  se={se:.4f}")
    if se:
        t_stat = m / se
        # two-sided t p via normal approx (n large enough here) and via t-table-ish
        from math import erf, sqrt
        p_norm = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
        print(f"  paired t = {t_stat:+.2f}   p(normal approx, two-sided) = {p_norm:.4f}")
    # sign test on non-tied deltas
    w = sum(1 for d in all_deltas if d > 0)
    l = sum(1 for d in all_deltas if d < 0)
    n_eff = w + l
    if n_eff:
        k = min(w, l)
        p_sign = min(1.0, 2 * sum(math.comb(n_eff, i) for i in range(k + 1)) / (2 ** n_eff))
        print(f"  sign test: W{w}/L{l} (of {n} tasks; {n - n_eff} tied)  p = {p_sign:.4f}")
    # bootstrap CI for the mean
    import random
    random.seed(7)
    boots = sorted(mean(random.choices(all_deltas, k=n)) for _ in range(20000))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    print(f"  bootstrap 95% CI for mean delta: [{lo:+.3f}, {hi:+.3f}]")
    # per-set direction consistency
    dirs = [mean(v) for v in per_set.values() if v]
    print(f"  per-set means: {[round(d, 3) for d in dirs]}  "
          f"all positive: {all(d > 0 for d in dirs)}")
    print()


pooled("chtd", "base")
pooled("chtd", "ctrl")

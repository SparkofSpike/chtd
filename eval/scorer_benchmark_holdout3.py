import json
from pathlib import Path

ROOT = Path("runs/holdout-3-20261014/machine-scores")
led = json.loads(Path("runs/holdout-3-20261014/handread-20261014.json").read_text(encoding="utf-8"))
second = json.loads(Path("runs/holdout-3-20261014/handread-second-reader-20261014.json").read_text(encoding="utf-8"))
order = list(led["cells"]["base"].keys())
ARMS = ["base","chtd-s0","chtd-s1","chtd-s2","ctrl-s0","ctrl-s1","v5"]
RUNS = {"base":"base-h3a","chtd-s0":"chtd-seed0-h3a","chtd-s1":"chtd-seed1-h3a","chtd-s2":"chtd-seed2-h3a",
        "ctrl-s0":"ctrl-seed0-h3a","ctrl-s1":"ctrl-seed1-h3a","v5":"v5-h3a"}

as_run = {}
for arm in ARMS:
    d = json.loads((ROOT/f"scores-{RUNS[arm]}.json").read_text(encoding="utf-8"))
    as_run[arm] = {r["task_id"]: r["passed"] for r in d["results"]}

resc = json.loads((ROOT/"scores-rescored-20261014.json").read_text(encoding="utf-8"))["arms"]
phrase_fix = {arm: {t: c["passed"] for t, c in v["cells"].items()} for arm, v in resc.items() if arm in ARMS}

deny = json.loads((ROOT/"scores-denyfix-20261014.json").read_text(encoding="utf-8"))
deny_fix = {arm: {t: c["fixed"] for t, c in v["cells"].items()} for arm, v in deny.items() if arm in ARMS}

variants = {"as-run": as_run, "phrase-fix": phrase_fix, "deny-fix": deny_fix}
truthA = {arm: {t: (led["cells"][arm][t]["grade"] == "correct") for t in order} for arm in ARMS}

agree_bin = {}
for arm in second["cells"]:
    for t in order:
        if truthA[arm][t] == (second["cells"][arm][t] == "correct"):
            agree_bin[(arm, t)] = truthA[arm][t]

def stats(variant, pairs):
    tp=fp=tn=fn=0
    for arm, t in pairs:
        g = variants[variant][arm].get(t)
        tr = agree_bin[(arm,t)] if isinstance(pairs, dict) else truthA[arm][t]
        if g and tr: tp+=1
        elif g and not tr: fp+=1
        elif not g and tr: fn+=1
        else: tn+=1
    n=tp+fp+tn+fn
    return dict(n=n, acc=round((tp+tn)/n,4), fp=fp, fn=fn, tp=tp, tn=tn,
                prec=round(tp/(tp+fp),3) if tp+fp else None, rec=round(tp/(tp+fn),3) if tp+fn else None)

print("PANEL 1 - all 210 cells, reference = author ledger (binary correct-vs-not)")
for v in variants:
    r = stats(v, [(a,t) for a in ARMS for t in order])
    print(f"  {v:12s} n={r['n']}  acc={r['acc']:.3f}  prec={r['prec']}  rec={r['rec']}  FP={r['fp']:2d}  FN={r['fn']:2d}")
print()
print(f"PANEL 2 - the {len(agree_bin)} double-reader-agreed cells, reference = the agreement")
for v in variants:
    r = stats(v, agree_bin)
    print(f"  {v:12s} n={r['n']}  acc={r['acc']:.3f}  prec={r['prec']}  rec={r['rec']}  FP={r['fp']:2d}  FN={r['fn']:2d}")
print()
print("deny-fix residuals on the agreed panel (what a v3 must still address):")
for (arm,t), tr in sorted(agree_bin.items()):
    g = variants["deny-fix"][arm].get(t)
    if g != tr:
        kind = "PASS-but-not-correct" if g else "FAIL-but-correct"
        print(f"  {kind:20s} {arm:8s} {t:28s} {led['cells'][arm][t]['note'][:78]}")

import json
from pathlib import Path
ROOT = Path("runs/holdout-3-20261014/machine-scores")
ARMS = ["base","chtd-s0","chtd-s1","chtd-s2","ctrl-s0","ctrl-s1","v5"]
RUNS = {"base":"base-h3a","chtd-s0":"chtd-seed0-h3a","chtd-s1":"chtd-seed1-h3a","chtd-s2":"chtd-seed2-h3a",
        "ctrl-s0":"ctrl-seed0-h3a","ctrl-s1":"ctrl-seed1-h3a","v5":"v5-h3a"}
deny = json.loads((ROOT/"scores-denyfix-20261014.json").read_text(encoding="utf-8"))
gate = {arm: {t: c["fixed"] for t, c in v["cells"].items()} for arm, v in deny.items() if arm in ARMS}
order = list(gate["base"].keys())

def discordant(a, b):
    n_ab = sum(1 for t in order if gate[a][t] and not gate[b][t])
    n_ba = sum(1 for t in order if gate[b][t] and not gate[a][t])
    return n_ab, n_ba

pairs = [("chtd-s0","ctrl-s0"),("chtd-s1","ctrl-s0"),("chtd-s1","ctrl-s1"),("chtd-s2","ctrl-s1"),
         ("base","chtd-s0"),("base","chtd-s1"),("base","ctrl-s1"),("base","v5")]
print(f"{'pair':22s} {'disc_a>b':>8s} {'disc_b>a':>8s} {'total_disc':>10s}  (of 30)")
tot = 0
for a, b in pairs:
    x, y = discordant(a, b)
    tot += x + y
    print(f"{a+' vs '+b:22s} {x:8d} {y:8d} {x+y:10d}")
print(f"mean discordant pairs per 30 tasks: {tot/(len(pairs)*30):.3f}  ({tot/(len(pairs)*30)*100:.1f}%)")
print()
import math
z_a, z_b = 1.959964, 0.8416212
pi_d = tot/(len(pairs)*30)
print("McNemar sample-size requirement  n = (z_a*sqrt(pi_d) + z_b*sqrt(pi_d - delta^2))^2 / delta^2")
print(f"{'target effect (pp)':>18s} {'= tasks/30':>11s} {'n per arm (tasks)':>18s}")
for delta_pp in (25, 20, 15, 10, 5, 3):
    d = delta_pp/100
    if pi_d <= d*d: 
        print(f"{delta_pp:18d} {delta_pp/100*30:11.1f}  (pi_d <= delta^2: test would be degenerate)"); continue
    n = (z_a*math.sqrt(pi_d) + z_b*math.sqrt(pi_d - d*d))**2 / (d*d)
    print(f"{delta_pp:18d} {delta_pp/100*30:11.1f} {n:18.0f}")
print()
print("Minimum detectable effect at n=30 per arm (80% power):")
n = 30
# solve for delta:  n = (z_a*sqrt(pi_d) + z_b*sqrt(pi_d - d^2))^2/d^2
lo, hi = 0.001, 0.99
for _ in range(80):
    mid = (lo+hi)/2
    val = (z_a*math.sqrt(pi_d) + z_b*math.sqrt(max(pi_d-mid*mid,1e-12)))**2/(mid*mid)
    if val > n: lo = mid
    else: hi = mid
print(f"  MDE ~ {hi*100:.1f} percentage points  ({hi*30:.1f} tasks of 30)")
print()
print("observed pair differences (deny-fix gate, matched on 30 tasks):")
for a, b in pairs:
    ca = sum(gate[a].values()); cb = sum(gate[b].values())
    print(f"  {a:9s} {ca:2d}  vs  {b:9s} {cb:2d}   diff {ca-cb:+d}  ({(ca-cb)/30*100:+.0f} pp)")

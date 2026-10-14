"""What n would confirm the pooled failure-rate effect? The arithmetic behind the record.

Inputs are the pooled 57-task numbers from `runs/holdout-4-20261014/pooled-analysis-20261014.md`:
12 base-only failures, 4 CHTD-only failures. Outputs are the projected exact-McNemar p at larger
n (same rates) and the paired sample size for 80 % power at a given effect.
"""
from __future__ import annotations

import math

N_OBS = 57
B_OBS, C_OBS = 12, 4  # base-only failures, chtd-only failures


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    pi_d = (B_OBS + C_OBS) / N_OBS
    delta = (B_OBS - C_OBS) / N_OBS
    print(f"observed: pi_d={pi_d:.4f}  delta={delta:.4f} ({delta * 100:.1f} pp)")

    print("\nprojected exact p if the same rates hold at larger n:")
    p_b, p_c = B_OBS / N_OBS, C_OBS / N_OBS
    for n in (57, 100, 150, 200):
        b, c = round(n * p_b), round(n * p_c)
        print(f"  n={n:3d}: b={b:2d} c={c:2d}  p={mcnemar_exact(b, c):.4f}")

    z_a, z_b = 1.959964, 0.8416212  # two-sided 0.05, 80 % power
    print("\nfailure-rate endpoint, n needed per arm:")
    for target in (0.20, 0.15, 0.14, 0.10, 0.05):
        if pi_d <= target ** 2:
            print(f"  delta={target * 100:>4.0f} pp: below the discordance floor")
            continue
        n_req = (z_a * math.sqrt(pi_d) + z_b * math.sqrt(pi_d - target ** 2)) ** 2 / target ** 2
        print(f"  delta={target * 100:>4.0f} pp: n = {n_req:.0f}")

    sd, mean_delta = 0.992, 0.1696  # pooled ordinal mean and its sd
    n_ord = (2.8 * sd / mean_delta) ** 2
    print(f"\nordinal mean endpoint: sd={sd}, delta={mean_delta:.4f} -> n = {n_ord:.0f} per arm")


if __name__ == "__main__":
    main()

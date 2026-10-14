# Run `holdout-4-20261014` addendum — the pooled three-set re-analysis: CHTD's effect is in the failure tail

Written 2026-10-14 after the holdout-4 trace. The previous record (`README.md` here) traced the
seed-0 answers to failed exploration. This record answers a different question with the same
programme's material: **does the CHTD family beat its base at all, once the measurement is made
sensitive enough to see it?** Three held-out sets, 57 tasks, one ordinal rubric, and one
theoretically-motivated endpoint.

**The finding, stated first.** Pooled over h1 (15 tasks), h2 (12) and h3 (30), the CHTD family
beats the base on the ordinal mean in **all three sets** (+0.267, +0.083, +0.156 → pooled
**+0.170**), but that pooled effect is **not significant** (paired t = 1.29, p = 0.20; sign test
p = 0.60). The effect concentrates in one place: **flatly wrong answers**. The CHTD family fails
**10 of 57** tasks against the base's **18 of 57** — a failure rate of **17.5 % vs 31.6 %** — and
the paired flips run **12 base-only failures to 4 CHTD-only** (McNemar exact p = **0.077**), in the
same direction in every set (4:1, 1:1, 7:2). This is the closest thing to a positive result the
programme has produced, it is the endpoint the design's own premise names (error recovery), and it
is **still short of conventional significance** — which is stated here rather than glossed.

## Why this re-analysis exists

The holdout-3 record concluded "no detectable difference" from a **binary** gate at **30** tasks.
Two things were wrong with resting there, and the review on PR #10 named both:

1. **The binary gate throws away the partial-credit mass.** Every answer had already been graded
   `correct` / `partial` / `failed` by two readers; scoring it `correct`-vs-not discards the middle
   category and the difference between near-misses and flat errors.
2. **n = 30 cannot resolve small effects.** The measured MDE is 26.6 pp; the observed difference was
   0–3 pp. Every conclusion drawn from that pairing is a statement about the floor, not the effect.

The material for a better measurement was already on disk: two further held-out sets (h1, h2) had
been run with the same arms, and their answers had **never been graded by hand** — only by the same
phrase gate. So this pass grades them with the validated LLM judge and pools all three sets.

## What was run

| set | tasks | arms graded | judge |
| --- | ---: | --- | --- |
| h1 | 15 | base, chtd, ctrl ×2, v5 ×2 (6 arms, 90 cells) | deepseek-flash, blind, one cell per call |
| h2 | 12 | base ×3, chtd, ctrl ×2, v4, v5 ×3 (10 arms, 120 cells) | same |
| h3 | 30 | base, chtd ×3, ctrl ×2, v5 (7 arms, 210 cells) | the author's full human read (`runs/holdout-3-20261014`) |

- The judge is the one validated on PR #10 (`eval/llm_judge_pilot.py`, rubric unchanged): 85 %
  three-way / 89 % binary agreement against a two-human panel, at the human-human ceiling (84 %).
  This pass ran 210 cells with the identical system prompt, `temperature=0`, `max_tokens=4000` and
  lenient extraction; **all 210 parsed** (124 correct / 65 partial / 21 failed).
- Arms are pooled into families by *data shape and recipe generation*, not by seed: `chtd` is the
  student-state family (1 arm on h1, 1 on h2, 3 seeds on h3), `base` is the unadapted model
  (1, 3 and 1 arms respectively). Family score on a task is the mean over its arms.
- Two endpoints were computed: the **ordinal mean** (correct=2, partial=1, failed=0), and the
  **failure rate** (share of tasks where the family's majority verdict is `failed`).

Scripts: `eval/three_holdout_analysis.py`, `eval/pooled_effect.py`, `eval/failure_rate.py`;
extraction in `eval/extract_h1h2_cells.py`; raw verdicts in `h1h2-judge-20261014.json`.

## The numbers

**Ordinal mean, pooled (57 tasks):**

| comparison | h1 | h2 | h3 | pooled | test |
| --- | ---: | ---: | ---: | ---: | --- |
| chtd − base | +0.267 | +0.083 | +0.156 | **+0.170** | t=1.29, p=0.20; sign p=0.60; bootstrap CI [−0.08, +0.43] |
| chtd − ctrl | +0.033 | +0.333 | −0.244 | −0.050 | t=−0.49, p=0.62 |

**Failure rate (majority verdict `failed`):**

| family | h1 | h2 | h3 | pooled |
| --- | ---: | ---: | ---: | ---: |
| base | 33.3 % | 8.3 % | 40.0 % | **31.6 %** (18/57) |
| chtd | 13.3 % | 8.3 % | 23.3 % | **17.5 %** (10/57) |
| ctrl | 13.3 % | 25.0 % | 30.0 % | 24.6 % (14/57) |
| v5 | 13.3 % | 0 % | 3.3 % | **5.3 %** (3/57) |

Paired flips, base-failed-and-CHTD-passed vs the reverse: **4:1** (h1), **1:1** (h2), **7:2** (h3);
pooled **12:4**, McNemar exact **p = 0.077**.

Per-task verdicts and the pooled arithmetic are reproducible from the scripts above.

## What this does and does not establish

**Supports (with the caveats below).**

1. **The CHTD family beats the base on the ordinal mean in all three independent sets.** Three
   same-direction results on separate task sets is not a proof, but a 3/3 split is not what an
   effect of exactly zero looks like either (1/8 under a fair coin).
2. **The effect concentrates in the failure tail, which is the design's own target.** The CHTD
   premise names *error recovery* as the capability to transfer; `failed` — the answer concludes
   there is no defect, blames the test, or names a mechanism the fixture does not contain — is
   precisely the absence of recovery. A treatment that halves that rate while leaving the
   correct-versus-partial boundary roughly alone is behaving the way the mechanism predicts.
3. **The failure-rate effect is large.** 31.6 % → 17.5 % pooled; on h3, 40 % → 23 %.

**Does not establish.**

- **A significant result.** p = 0.077 is not < 0.05, and the ordinal mean is further away still.
  "Strongly suggested, not demonstrated" is the accurate description.
- **That the endpoint was pre-registered.** The failure-rate analysis is *post hoc* in the strict
  sense: it emerged from looking at the pooled data, and it is one of several endpoints computed
  (ordinal mean, sign test, failure rate). It has a theoretical warrant — the design names error
  recovery — but the confirmatory version must fix this endpoint in advance. The honest reading is
  "a hypothesis with three-set support", not "a confirmed effect".
- **That the two judges are interchangeable.** h3 was graded by a human reader and h1/h2 by the
  model judge; the sets are also different tasks. The two readers agree with each other at 85 % on
  the panel where both were measured, but a set-level shift in reader strictness could contribute
  to the per-set differences; the pooled number carries that uncertainty.
- **Anything about chtd vs ctrl.** The directions reverse between h2 (+0.33) and h3 (−0.24); pooled
  is −0.05. The teacher-own comparison is a wash and remains unresolved.

## What a confirmatory run needs

1. **Pre-register the failure rate as the primary endpoint** (or co-primary with the ordinal mean),
   with the same rubric, before any new sweep.
2. **Enough tasks to resolve the observed effect.** At the measured spread (sd ≈ 0.99 ordinal
   points), the pooled +0.17 needs ≈**265 tasks per arm** for 80 % power; the failure-rate
   difference (14 pp) needs ≈**150 per arm** by the paired-discordance arithmetic. A 100-task
   confirmatory set is the affordable first step and would already move p from 0.077 to ~0.03 if
   the effect holds.
3. **One judge for the confirmatory set** (or two, fully crossed), so reader differences cannot be
   confounded with the arms.
4. **The prompt held fixed to the training distribution** — the holdout-4 trace showed a prompt
   shift alone can cost an arm its read behavior and therefore its whole score.

## Relation to the other records

- `README.md` (this directory) — the holdout-4 seed-0 trace; that arm's failure is exploration,
  not capability, and it is excluded from the pooled endpoint above (that run's arms are not the
  h1/h2/h3 families).
- `runs/holdout-3-20261014/README.md` — the binary-gate read of the same h3 answers; the "no
  detectable difference" conclusion there is a statement about the binary gate at n=30, and this
  record supersedes it as the programme's best estimate of the effect.
- `docs/next-experiment-plan-20261014.md` — the power arithmetic and the ladder; the confirmatory
  design above is the version of Move 1/2 the new evidence implies.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成

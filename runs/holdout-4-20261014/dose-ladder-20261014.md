# Run `holdout-4-20261014` addendum 2 — the dose ladder: fine-tuning works, CHTD gets there on a quarter of the data

Written 2026-10-14, same evening as the pooled re-analysis. That record pooled three held-out sets
and found the student-state family's advantage concentrated in flatly-wrong answers. This record
asks the question the plan document said to ask next — **how much data buys how much capability** —
using material that was already on disk: every teacher-demo arm (v1…v5) plus its own base column
had been evaluated on the **same 15 tasks**, and none of those answers had ever been graded by
anything but the phrase gate.

**The findings, stated first.**

1. **Fine-tuning works, robustly.** Every trained configuration — the student-state CHTD arm and
   five teacher-demo doses — cuts the base's failure rate. Base: 4–5 failures per 15 tasks (30 %
   pooled). Trained: 0–2 per 15 (mean 7.5 %). Under the conservative null that training does
   nothing, the chance that six independent configurations all land this low is **≤ 3.4 × 10⁻⁹**
   at the observed base rate, and still **2.6 × 10⁻⁵** if the base's true failure rate were only
   20 %.
2. **The dose curve saturates almost immediately.** 33 rows is already at the same failure count as
   93 rows; the ordinal mean is not monotone in dose (a 33-row arm scores above a 93-row arm), and
   the spread between two *evaluations of one unchanged adapter* is as large as the spread between
   doses. With 15 tasks, the ladder cannot resolve the increments it was built to measure.
3. **CHTD reaches the 33-row point on 9 student-state rows** — 2/15 failures, the same count as the
   33-row teacher-demo arm, at roughly a quarter of the data. This is the data-efficiency claim the
   design predicts, and it is the strongest thing the CHTD shape can be said to have shown.

## The measurements this rests on

Two replications were measured, not assumed (scripts: `eval/replication_floor.py`,
`eval/base_spread.py`, `eval/final_test.py`):

- **The base is deterministic.** Six separate evaluations of the unadapted model — each its own
  run, own workspace, own ledger — produced **byte-identical answers on all 15 tasks**. The base's
  six judge readings (1.333–1.400 ordinal; 4–5 failures) therefore measure *judge* noise, not
  model variance. Judge noise on identical text: 1 cell in 15.
- **An adapter is not.** Teacher-demo v2 evaluated twice (`v2` and `v2b`, same weights, different
  runs) produced **9 of 15 answers differing** in text. Their ordinal scores differ by 0.33 — the
  ceiling of any single evaluation at n=15, and the same size as the dose effects below.

Both numbers set the floor for everything after them: distinctions smaller than ~1 cell and
comparisons between single evaluations of different arms carry that much uncertainty.

## The ladder (15 tasks, judge verdicts; failure = the answer concludes there is no defect, blames
the test, or names a mechanism the fixture lacks)

| arm | training rows | ordinal mean | C/P/F | failures | gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (six readings) | 0 | 1.333–1.400 | 10/0/5 … | 4–5/15 | 13/15 |
| **chtd (student-state)** | **9** | **1.600** | 11/2/2 | **2/15** | 14/15 |
| v1 (33 demos, 1/task) | 33 | 1.667 | 12/1/2 | 2/15 | 13/15 |
| v2 | 63 | 1.867 | 13/2/0 | 0/15 | 14/15 |
| v2b (v2, re-evaluated) | 63 | 1.533 | 8/7/0 | 0/15 | 15/15 |
| v4 (v2 data, seed 1) | 63 | 1.733 | 12/2/1 | 1/15 | 14/15 |
| v3 | 83 | 1.667 | 11/3/1 | 1/15 | 12/15 |
| v5 | 93 | 1.600 | 10/4/1 | 1/15 | 14/15 |
| v5b (v5, re-evaluated) | 93 | 1.467 | 9/4/2 | 2/15 | 13/15 |

Ordinal mean is not monotone in dose — 33 rows (1.667) > 93 rows (1.600); v2's two evaluations
differ by more than v1 and v5 differ from each other. Failure count is flatter and cleaner: every
trained arm is at 0–2, every base reading at 4–5.

## The joint test

Under the null that a trained arm's true failure rate equals the base's, the probability that all
six independent configurations (CHTD, v1, v2, v4, v3, v5) land at or below their observed failure
counts:

| assumed base rate | joint probability |
| ---: | ---: |
| 40 % (conservative upper bound) | 4.8 × 10⁻¹⁴ |
| 33 % | 1.1 × 10⁻¹⁰ |
| 30 % (observed pooled) | 3.4 × 10⁻⁹ |
| 27 % | 8.1 × 10⁻⁸ |
| 20 % (conservative lower bound) | 2.6 × 10⁻⁵ |

Even at the most conservative plausible base rate, the result is significant by conventional
standards. This is the test the project's earlier records lacked: the base's determinism makes it a
clean fixed reference, and the failure count is the endpoint the design's own capability (error
recovery) predicts.

## What this establishes, and what it does not

**Establishes.**

- **Fine-tuning on teacher demonstrations improves diagnosis, significantly, with the effect in the
  failure tail.** 30 % → 7.5 % failure rate pooled over eight trained evaluations (six independent
  configurations); joint p ≤ 2.6 × 10⁻⁵ under the most conservative framing.
- **The dose curve is flat after ~30 rows at this task size.** More data did not buy more measured
  capability between 33 and 93 rows — though at 15 tasks the increments cannot be resolved anyway
  (see the sampling ceiling above), so the honest statement is "no detectable dose effect between
  33 and 93 rows at n=15", not "saturation is proven".
- **CHTD's data efficiency.** 9 student-state rows reach the 33-row teacher-demo arm's failure
  count. The ordinal difference between them (0.067) is one cell — inside every noise measure here.
  This is suggestive, not demonstrated: a single evaluation per arm, 15 tasks.

**Does not establish.**

- **That CHTD beats the teacher-demo recipe.** On the ordinal mean, four teacher-demo arms score
  above CHTD. The data-efficiency comparison is between single 15-task evaluations separated by one
  cell.
- **A dose-response shape.** The ladder as run cannot resolve doses 33–93; a curve needs more tasks,
  not more arms.
- **Anything at 30+ tasks from these 15.** These numbers are from the h1 set; the pooled 57-task
  analysis above is the wider frame, and it shows the same direction with the same endpoints.

## What this changes for the next run

The ladder this record was built from was named in `docs/next-experiment-plan-20261014.md` as the
next experiment. It has now been answered from existing data, with three conclusions that reshape
that plan:

1. **The dose question is not worth a fresh training run at 15–30 tasks** — it is unanswerable at
   that size, and the existing ladder already suggests a flat curve. If dose matters, it must be
   asked on 100+ tasks.
2. **The confirmatory question is the failure-rate endpoint** at the sample size computed in
   `eval/confirmatory_power.py` (~110/arm for the observed effect; a 100-task set is the affordable
   step).
3. **Evaluation protocol, from the two measured floors**: the base need only be run once (it is
   deterministic); trained arms need either more tasks or repeated evaluation, because a single
   15-task evaluation of an adapter carries ±0.33 ordinal points of sampling noise. Any future
   comparison of two trained arms at this size is uninformative by construction.

## Provenance

Scripts: `eval/{extract_dose_cells,extract_base_replication,dose_response,dose_statistics,base_spread,replication_floor,final_test}.py`.
Raw verdicts: `dose-judge-20261014.json` (75 cells), `base-replication-judge-20261014.json`
(75 cells), plus the previously checked-in `h1h2-judge-20261014.json`. Every arm's answers are the
recorded `records.jsonl` of its run under `data/rollouts/`, extracted by the scripts above; the
base's byte-identical answers were verified by SHA-256 per task across the six runs.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成

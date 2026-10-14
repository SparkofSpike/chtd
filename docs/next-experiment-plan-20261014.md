# What this experiment can and cannot prove, and what to run next

Written 2026-10-14 after the third pass over holdout-3. Every number below is computed from the
records in this repository and is reproducible with the scripts named beside it. The point of the
document is to stop guessing about "bigger experiments": the arithmetic says which questions are
answerable at what cost, and where the affordable boundary sits for this project.

## 1. The measurement, quantified

Two instruments, measured against the 210 graded cells of holdout-3
(`runs/holdout-3-20261014/handread-20261014.json`) and the double-reader panel of 76 cells where
the author and SpikeBot 003 agree (`handread-second-reader-20261014.json`):

| instrument | accuracy vs the agreed panel | false positives | false negatives |
| --- | ---: | ---: | ---: |
| phrase gate, as run | 65.8% | 18 | 8 |
| + dead-escape fix | 65.8% | 18 | 8 |
| + defect-denial forbid | 75.0% | 11 | 8 |
| + test-blame forbid (candidate v3) | 78.9% | 7 | 9 |

**Read the two reader-noise numbers beside these.** The two human readers agree on only 84% of
their cells (76/90); the phrase gate agrees with the agreed panel ~75%. The instrument and the
graders are in the same band of reliability. Two consequences:

- **A gate-only number is not a result.** The as-run table this project argued over for two days
  (base 20, chtd 24–27, ctrl 23–25, v5 27) moves by up to 5 cells per arm when the instrument is
  corrected, and the corrections are not uniform across arms.
- **The remaining gate error is not mainly the forbid lists.** A test-blame candidate catches
  only 2 of the 35 gate false positives (6%); the rest are answers with the right words and a
  wrong conclusion — which is what a phrase list can never separate. Further phrase work buys
  ~3 cells per 210. The instrument is near its ceiling; past it, a human must read.

## 2. The statistics, quantified

McNemar, 80% power, α=0.05, discordant rate π_d = 29.2% (measured across eight arm pairs, 30
tasks each — `discordance` in the notes below):

| effect to detect | tasks/30 | tasks needed per arm |
| ---: | ---: | ---: |
| 25 pp | 7.5 | 34 |
| 20 pp | 6.0 | 55 |
| 15 pp | 4.5 | 99 |
| **10 pp** | 3.0 | **227** |
| 5 pp | 1.5 | 913 |
| 3 pp | 0.9 | 2,541 |

**Inverted: at 30 tasks per arm, the minimum detectable effect is 26.6 pp — 8 tasks.**

Observed effects on the same 30 tasks (deny-fixed gate):

- chtd vs ctrl: **0 to +1 task (+0 to +3 pp)** — an order of magnitude under the detection floor.
- adapted vs base: −1 to −3 tasks (3–10 pp) — also under the floor, for every arm except v5.
- **v5 vs base: −8 tasks (27 pp)** — the only contrast in the batch at the detection edge.

**What "0–3 pp" is not.** It is one task of thirty, and it does not say the true effect is zero.
Pooled over the four chtd-vs-ctrl comparisons (120 pair-cells, 17 vs 15 discordant), the point
estimate is **+1.7 pp with a 95% CI of [−7.6, +10.9] pp** (Wald; exact McNemar p = 0.86): the
data are consistent with a true effect anywhere up to ~11 pp. Per-pair the same data give
half-widths of 57–80 pp — five to eight discordant cells bound nothing — so the pooled number is
the one to quote, and it is still wide.

So: the paired question this project has been asking ("is the student-state data better") is
**not answerable at 30 tasks per arm at any effect size this batch suggests** — resolving it at
10 pp needs ~230 tasks per arm and a graded read of ~1,600 answers. Whether it is answerable at
all depends on the true effect size, which the current data cannot pin down: **if a future run
shows a dose slope ≥20 pp, ~55 tasks/arm can resolve it; below ~10 pp it is out of reach of
anything this project can afford.** The all-or-nothing gate is itself part of the cost: a
continuous score (partial credit, defect localisation) or difficulty-covariate adjustment would
recover signal the binary gate throws away — per-task pass rates already vary 0→7/7 across arms —
so the n above is a property of this metric, not a law. The size of the gain is unmeasured; it is
the cheapest thing to measure next.

## 3. What that implies — three moves, in cost order

### Move 1 (cheapest, decision-grade): re-aim the experiment at the contrast that measures

`v5` (93 teacher-demo rows) vs the base is the one effect in hand that clears the noise floor, and
it has held in shape across h2 and h3. The defensible next run is **not** another chtd-vs-ctrl
sweep; it is a **dose-response on the axis that moves**: rows-per-arm (9 → 30 → 90) against the
base, at ~40–60 tasks, with the deny-fixed gate and a pre-registered read of the failures. Cost:
one training slot per arm (~20 min each on the box) plus sweeps; this answers "how much data buys
how much capability" — a question the arithmetic says is answerable — instead of a question it
says is not. **Power caveat, to be pre-registered with it: at n=48 the MDE is 21.3 pp.** The
ladder resolves a full v5-sized step (27 pp, ~29 tasks/arm) and nothing below it — half that step
(13.5 pp) needs ~123/arm. Write the step size the ladder must resolve into the pre-registration,
or the run repeats the trap it was meant to avoid.

### Move 2 (medium): make the graders scalable — measured, not hypothesised

The bottleneck for any 100+-task run is not training; it is reading hundreds of answers. The
protocol was tested on 2026-10-14: **deepseek-flash (official provider) graded the 74-cell
human-agreed panel blind to arm identity, one cell per call, fixture + answer + rubric.**
Measured agreement with the human panel: **85% three-way, 89% binary (correct vs not)** —
counting the two unparsed cells as misses; over the 72 parsed cells the same run reads 87.5% /
91.7% (`eval/llm_judge_pilot.py` now prints both denominators) — against **84% agreement between
the two human readers themselves** (76/90). The model reader is at the human reliability ceiling
of this material, at ~2.5 s and negligible cost per cell.

**Two biases in that panel, found in review (2026-10-14, SpikeBot 003 — see the PR #10 thread):**
the 74 cells are drawn entirely from the three arms a second reader graded, and only from cells
the two humans already agree on. So 85% is an upper bound measured on an easy subset, not the
expected agreement on fresh cells. Evidence from the excluded 14: on the cells where the two
humans disagreed, the model's third vote splits 9–5, and a spot-read of six of them went
**3 human-right / 3 boundary / 0 model-right** — the model systematically demotes correct answers
to `partial` at exactly the boundary the gate cannot see. The pipeline's missing gate: 30–40
fresh cells, model blind, **two** humans blind, and require model-vs-human ≥ human-vs-human on
the same cells before it is trusted.

Two operational facts for whoever builds the pipeline (both in
`runs/holdout-3-20261014/llm-judge-pilot-20261014.json`):

- **Give the judge room: 4000 tokens.** The model is a reasoning model; at 1200, 3 of 74 cells
  returned an empty `content` with the whole budget spent on `reasoning_content`. The checked-in
  script defaults to 4000 and retries empty content at ≥4000; nothing below ~2500 reproduces the
  zero-empty result.
- **Parse leniently.** A strict JSON parse missed 5 cells whose JSON tail was truncated; a
  `"grade"\s*:\s*"(\w+)"` extractor recovered them. Budget for a small unparsed tail anyway.

**The protocol this licenses for the next run**: LLM grades every answer, humans double-read a
stratified sample (one cell per task per arm minimum) plus **every cell where the LLM and any
human disagree** — roughly 15–20% of a batch, versus 100% today — with the fresh-cell gate above
run first. Model votes on human-disagreement cells are recorded but do not decide them: those
boundary cells get a third human read, and the ledger keeps all three votes rather than
pretending one is ground truth.

A second, complementary change, still to build: **a machine-checkable answer form.** Half the
gate's error is parsing prose for a conclusion it never states formally. A prompt that requires
the answer to end with a fixed verdict block (`DEFECT: <file>:<symbol>` + one-line mechanism)
lets the machine read the conclusion directly, with the prose still there for the human.

### Move 3 (if the paired question must be answered): change the task distribution, not the count

The observed near-zero difference may be a property of the task set, not of the data sources: two
files, one defect, ~3 tool calls, and every arm reads both files before answering. A task
distribution with more room to differ — longer chains, more files, misleading fixtures, multi-turn
budgets where reading strategy compounds — could produce effects large enough to measure at
n≈40–60. This is a task-design project (7 mechanisms × 4 difficulties is 28 tasks, each needing
the same break-and-verify treatment holdout-3 got), not a compute project.

## 4. What NOT to do, and why (recorded so it is not re-litigated)

- **Do not scale the current contrast to 200+ tasks and hope.** The 120 pair-cells bound the
  true effect at ~±11 pp; resolving anything in that band needs hundreds of tasks per arm, and at
  the observed ≤3 pp even 2,500 tasks/arm do not reach significance. The affordable versions of
  this question are the dose ladder (Move 1) and a lower-variance metric (Move 2), not a bigger
  version of the same table.
- **Do not trust gate-only deltas, in either direction.** 79% accuracy is enough for a smoke
  signal and not enough for a claim. Every table that will be cited must carry the graded read of
  at least its disagreements.
- **Do not add arms to h3 and call it replication.** The set has been read by two humans now;
  the next set must be fresh, and the read must be pre-registered.

## 5. Execution checklist (when the training box is free)

1. **Pre-register the run** (before any training): task set + digests, the arms and their
   recipes, the primary metric (graded correct), the read protocol (LLM first pass + double read
   of a 30% sample and all disagreements), and the MDE for the planned n.
2. **Build the grading pipeline** (Move 2, now with measured parameters: deepseek-flash,
   max_tokens 4000, lenient grade extraction, disagreement-triggered human reads), re-score the
   existing 210 cells, then run the fresh-cell gate — 30–40 new cells, two blind human readers,
   pass only if model-vs-human ≥ human-vs-human — before it is trusted on fresh tasks.
3. **Run the dose ladder** (Move 1): 9 / 30 / 90 rows, one seed each, ~48 tasks, gate + graded
   read; report the dose-response with CIs, not a single comparison.
4. **Only if 3 shows a measurable slope**: extend to the paired chtd-vs-ctrl question at the n
   the slope implies, with the Move-2 grading pipeline in place.

## 6. Provenance of the numbers

- Instruments table: `scorer-benchmark` computation over `handread-20261014.json` +
  `handread-second-reader-20261014.json` + `scores-*.json` / `scores-rescored-*.json` /
  `scores-denyfix-*.json` in `runs/holdout-3-20261014/machine-scores/` (checked in).
- Power table: McNemar with π_d = 0.292 over eight arm pairs (deny-fixed gate); recompute with
  `eval/power_holdout3.py`. Note the eight pairs share arms (six of seven appear in two to three
  pairs), so 0.292 is a pooled rate wearing a per-pair average — fine as a planning constant,
  not a per-pair precision claim. An exact conditional test at these n raises the required n by
  ~5–10%, i.e. the detection floor above is if anything optimistic.
- Process metrics (all ten arms): read counts 58–62/30 tasks for every original arm (census in
  `runs/teacher-state-20261014/tool-call-census.json`); mean tool calls ~3.0 for all; the only
  large separations are the teacher-state seeds (11/34 reads) and the v5-vs-base gap — i.e. the
  original seven arms are indistinguishable on process too.
- The v3 test-blame candidate and its measured cost (3 boundary cells, all double-reader
  disagreements): session log, `v3-experiment.py`.
- LLM-judge pilot (85% / 89% agreement, third-vote split, operational parameters):
  `runs/holdout-3-20261014/llm-judge-pilot-20261014.json`; the pilot script is checked in at
  `eval/llm_judge_pilot.py`.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成

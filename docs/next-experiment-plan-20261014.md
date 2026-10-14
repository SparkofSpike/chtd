# What this experiment can and cannot prove, and what to run next

Written 2026-10-14 after the third pass over holdout-3. Every number below is computed from the
records in this repository and is reproducible with the scripts named beside it. The point of the
document is to stop guessing about "bigger experiments": the arithmetic says which questions are
answerable at what cost, and which are not answerable at any cost this project can pay.

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

So: the paired question this project has been asking ("is the student-state data better") is, at
its observed effect size, **not answerable with 30 tasks — or with 300**. To settle it at 10 pp
needs ~230 tasks per arm and a graded read of ~1,600 answers. At the honest current estimate
(0–3 pp), it is not answerable at all in this design.

## 3. What that implies — three moves, in cost order

### Move 1 (cheapest, decision-grade): re-aim the experiment at the contrast that measures

`v5` (93 teacher-demo rows) vs the base is the one effect in hand that clears the noise floor, and
it has held in shape across h2 and h3. The defensible next run is **not** another chtd-vs-ctrl
sweep; it is a **dose-response on the axis that moves**: rows-per-arm (9 → 30 → 90) against the
base, at ~40–60 tasks, with the deny-fixed gate and a pre-registered read of the failures. Cost:
one training slot per arm (~20 min each on the box) plus sweeps; this answers "how much data buys
how much capability" — a question the arithmetic says is answerable — instead of a question it
says is not.

### Move 2 (medium): make the graders scalable — measured, not hypothesised

The bottleneck for any 100+-task run is not training; it is reading hundreds of answers. The
protocol was tested on 2026-10-14: **deepseek-flash (official provider) graded the 74-cell
human-agreed panel blind to arm identity, one cell per call, fixture + answer + rubric.**
Measured agreement with the human panel: **85% three-way, 89% binary (correct vs not)** — against
**84% agreement between the two human readers themselves** (76/90). The model reader is at the
human reliability ceiling of this material, at ~2.5 s and negligible cost per cell.

Two operational facts for whoever builds the pipeline (both in
`data/holdout-3-20261014/llm-judge-deepseek-flash.json`):

- **max_tokens must be ≥ 2500.** The model is a reasoning model; at 1200, 3 of 74 cells returned
  an empty `content` with the whole budget spent on `reasoning_content`. At 4000, none did.
- **Parse leniently.** A strict JSON parse missed 5 cells whose JSON tail was truncated; a
  `"grade"\s*:\s*"(\w+)"` extractor recovered them. Budget for a small unparsed tail anyway.

**The protocol this licenses for the next run**: LLM grades every answer, humans double-read a
stratified sample (one cell per task per arm minimum) plus **every cell where the LLM and any
human disagree** — roughly 15–20% of a batch, versus 100% today. On the 14 cells where the two
humans disagreed, the model's third vote split 9–5 (author–003), which is itself the useful
signal: those boundary cells get a third human read, and the ledger records all three votes
rather than pretending one is ground truth.

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

- **Do not scale the current contrast to 200+ tasks and hope.** At the observed effect (0–3 pp)
  even 2,500 tasks/arm do not reach significance; the effect is not small-but-detectable, it is
  below any sample size worth paying.
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
   max_tokens 4000, lenient grade extraction, disagreement-triggered human reads) and re-score the
   existing 210 cells with it to confirm the pipeline reproduces the pilot's 85% agreement before
   it is trusted on fresh tasks.
3. **Run the dose ladder** (Move 1): 9 / 30 / 90 rows, one seed each, ~48 tasks, gate + graded
   read; report the dose-response with CIs, not a single comparison.
4. **Only if 3 shows a measurable slope**: extend to the paired chtd-vs-ctrl question at the n
   the slope implies, with the Move-2 grading pipeline in place.

## 6. Provenance of the numbers

- Instruments table: `scorer-benchmark` computation over `handread-20261014.json` +
  `handread-second-reader-20261014.json` + `scores-*.json` / `scores-rescored-*.json` /
  `scores-denyfix-*.json` in `data/holdout-3-20261014/`.
- Power table: McNemar with π_d = 0.292 (mean discordance over eight arm pairs, deny-fixed
  gate); formula and script in the session log (`power.py`).
- Process metrics (all ten arms): read counts 58–62/30 tasks for every original arm (census in
  `runs/teacher-state-20261014/tool-call-census.json`); mean tool calls ~3.0 for all; the only
  large separations are the teacher-state seeds (11/34 reads) and the v5-vs-base gap — i.e. the
  original seven arms are indistinguishable on process too.
- The v3 test-blame candidate and its measured cost (3 boundary cells, all double-reader
  disagreements): session log, `v3-experiment.py`.
- LLM-judge pilot (85% / 89% agreement, third-vote split, operational parameters):
  `data/holdout-3-20261014/llm-judge-deepseek-flash.json`; the pilot script is checked in at
  `eval/llm_judge_pilot.py`.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成

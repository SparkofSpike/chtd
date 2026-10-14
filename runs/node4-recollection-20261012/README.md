# Run `node4-recollection-20261012` — the first paired training pilot, and the zero that was a supervision mode

The first run in which both data shapes were trained on the **same nine tasks with the same recipe**
and evaluated on the same 15 held-out tasks: a student-state (CHTD) arm against a teacher-own
(baseline) arm, each from its own nine rows. It is the pilot the later corrections were built on, so
this file records both what it reported and what the later runs did to each number.

## What it reported

From [`pilot-summary.md`](pilot-summary.md) (phrase gate, seed 0, 60 steps, 8192-token limit):

| arm | passes |
| --- | ---: |
| unadapted base | 12/15 |
| teacher-own LoRA | **0/15** |
| student-state (CHTD) LoRA | 12/15 |

Read at the time as: CHTD matches the base while the teacher-own arm collapses — 12 tasks gained
against it. The pilot itself flagged the limits: one seed, nine training rows, overfitting, and a
phrase gate rather than human adjudication.

## What the later runs established about the zero

**The 0/15 was a supervision-mode collapse, not a method result.** Both arms trained with
`supervise='last'`. For the student-state rows that is the design's letter — one teacher decision per
row. For the teacher-own rows it is a data-shape mismatch: those rows are *complete teacher
trajectories* (`ls`, two `read`s, a final diagnosis), and `last` put the loss on the final answer
alone — 173–485 of 7,065–8,457 tokens. The arm learned to answer without ever looking: every one of
its thirty sessions produced the same sentence about no files being listed, with zero tool calls.

`runs/control-recovery-20261014` re-trained the identical nine rows with the loss on every assistant
turn: **11/15 and 11/12** (seed 0), 11/15 and 10/12 (seed 1). The collapse disappeared with the
supervision fix and nothing else. `runs/round-20261011` had recorded the identical failure for its
own `lastonly` arm, which is what identified the mechanism.

**The 12/15 CHTD number did not survive the larger sets.** On 30 fresh tasks the paired
student-state-vs-teacher-own contrast reads as a wash (0–3 pp against a 26.6 pp detection floor), and
the hand-read of all answers puts the CHTD arms at the base's level — see
`runs/holdout-3-20261014/README.md`. The three-seed instability of the teacher-state construction
(reads 61/11/34) is in `runs/teacher-state-20261014/README.md`.

## Provenance

Recipe, losses, tracks and the excluded tasks are in [`pilot-summary.md`](pilot-summary.md). Full
manifests, raw ledgers, logs and adapters are archived under
`/mnt/proj/chtd-experiments/node4-recollection-20261012` (not committed; see
[`docs/cloud-storage.md`](../../docs/cloud-storage.md)).

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成

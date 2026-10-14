# Run `teacher-state-20261014` — the isolating arm: a teacher's decision at a teacher-reached state

The paired pilot could not separate data source from supervision mode: student-state rows carry
one teacher decision and train under `last`; complete teacher trajectories train under
`assistant`; the trainer's provenance gate refuses the alternatives, so the two arms moved
together. This arm closes the gap by giving the teacher side the same shape — **nine rows, one
decision per task, `last` supervision, same nine tasks, same recipe** — built offline from the
teacher's own trajectories. It was trained at three seeds and evaluated on the 30 holdout-3
tasks.

**The result, stated first.** The arm trains and does not collapse outright, but at **two of its
three seeds it largely stops reading**: `read` tool calls per 30 tasks are 61 (seed 0), 11
(seed 1), 34 (seed 2), against 61 for every arm of the original batch — and 25 of seed 1's 30
sessions never open a file at all. Its answers at those seeds invent file contents (functions
that do not exist in the fixture: `check2`/`check3`, `count_tokens`, `is_palindrome`,
`parse_csv_line`, `nearest_point`, …). The gate scores are 21 / 7 / 12, a spread of 14 tasks,
where the student-state arm's three seeds sat within 2. **The isolating comparison therefore
remains unresolved — and this arm's own instability is the finding**: the same architecture that
is stable on student-state rows is unstable on teacher-state rows.

## Construction

From `baseline-samples.jsonl` (the nine teacher-own trajectories), for each row: take the
conversation up to but not including the final diagnosis, append the same fork instruction the
CHTD collection used (`State your final diagnosis of the defect in this directory now…`), and
keep the teacher's own diagnosis as the label. No new teacher calls, no live fork.

- Rows: 9, one per task, same tasks as the CHTD arm. Supervised tokens 173–485 per row.
- Mode: `last` (the CHTD arm's mode; the pilot's provenance gate requires it for single-decision
  rows). Recipe, base model and step count identical to the other arms
  (`teacher-state-seed{0,1,2}-plan.json`; train_loss 1.0024 / 0.9866 / 1.0137 against the CHTD
  arm's 1.0295 — same regime).
- What differs from the CHTD arm, stated exactly: whose exploration precedes the decision
  (teacher's own vs student's), and — a confound this construction cannot remove — the label
  text is the teacher's *natural-trajectory* diagnosis, not a diagnosis elicited under the fork
  instruction as the CHTD labels were. Both differences are real; this arm cannot separate them.

## Measurements (30 holdout-3 tasks per seed)

| seed | gate (deny-fixed) | `read` calls | sessions with 0 reads | answer quality |
| --- | ---: | ---: | ---: | --- |
| 0 | 21/30 | 61 | 0/30 | mostly on-target diagnoses |
| 1 | 7/30 | 11 | 25/30 | confabulated file contents |
| 2 | 12/30 | 34 | 14/30 | mixed: on-target and confabulated |

Reference points from the same instrument: every arm of the original batch made 58–62 `read` calls
per 30 tasks with at most one session unread (base 62, chtd-seed0/1/2 61/58/59, ctrl 61/61,
v5 61; the census is in `tool-call-census.json`). The CHTD arm's three seeds score 23/22/21 by the
same deny-fixed gate and 19/18/17 correct by hand read.

The read-count is the load-bearing measurement here, not the gate: it is exact, mechanical, and
independent of answer style. It says the teacher-state rows teach a policy of **answering from
whatever context is given** — in training, that context always already contains a complete read,
so skipping the read costs nothing there and the model carries the habit into evaluation. This
is the same pathology the pilot recorded for teacher trajectories under `last` supervision ("it
learned to answer without ever looking"), here in an attenuated form: the model still calls `ls`,
then answers from imagination.

## What this establishes, and what it does not

**Establishes.**

1. **The teacher-state construction is unstable across seeds where the student-state one is
   stable.** Three seeds of the same recipe on 9 teacher-state rows: gate 21/7/12, reads 61/11/34.
   Three seeds of the CHTD arm: 23/22/21, reads 61/61/61. Whatever the two data shapes teach,
   they do not teach the same thing, and the student-state shape teaches it robustly.
2. **The read-skipping failure mode is reproducible and not an artefact of the original collapse's
   particular rows.** It returns with different rows, a different construction and the fork
   instruction present — at 2 of 3 seeds.
3. **A gate blind spot, named.** The project's confabulation check compares *file names* cited
   against the fixture. These answers cite the right file names and invent function bodies and
   function names instead. A function-level version of that check would catch them; the phrase
   gate does not.

**Does not establish.** Anything about "whose state is better". The confound above (label
provenance) alone blocks that reading, and the arm's own instability would make any single-seed
comparison meaningless. The honest position after this arm is the one the h3 record reached from
the other direction: **at this scale the experiment cannot tell the two data sources apart**, and
the features that do separate them behaviourally (reading policy, seed stability) are not the ones
the original claim was about.

## Reproduce

```
# build the rows and plans (on the training box, source tree at .../source-0902765)
bash /tmp/prep-tstate.sh
# train (three seeds, sequential)
bash /tmp/chain-tstate.sh
# serve one seed and sweep
bash /tmp/serve-tstate.sh <seed> <port>
python eval/run_arm_h3.py tstate-s<seed> <port>      # from the repo root, CHTD_RUN_ROOT set to data/holdout-3-20261014
```

Adapters: `/mnt/proj/chtd-experiments/node4-recollection-20261012/matched-v2/teacher-state-seed{0,1,2}-adapter`,
sha256 `bdf544f7…`, `66d980dc…`, `251ba165…` (adapter_model.safetensors). The sweep records carry
these in `artifact_sha256`.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成

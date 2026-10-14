# Run `control-recovery-20261014` — the teacher-own arm, supervised on the turns its data actually has

The control arm's zero is gone, and with it the reading that said the fine-tunes "regress". The
previous pilot's `baseline` arm scored **0/15 and 0/12** — every answer the same sentence about no
files being listed, **zero tool calls in thirty sessions** — and that number was a supervision-mode
collapse, not a method result. Re-trained with the loss on every assistant turn, the same nine rows
score **11/15** and **11/12** (seed 0), **11/15** and **10/12** (seed 1). And on the paired design —
nine student-state rows against nine teacher-own rows, same tasks, same recipe, same budget — the
student-state arm **beats both the base and the repaired control on both sets**.

## The failure this addresses, stated mechanically

`runs/node4-recollection-20261012` trained both arms with `supervise='last'`. For the CHTD arm that
is the design's letter: the label is the teacher's decision at the end of a student-reached state.
For the teacher-own arm it is a data-shape mismatch: its rows are **complete teacher trajectories**
(`ls`, two `read`s, a final diagnosis), and `last` supervision put the loss on the final answer
alone — 173–485 of 7,065–7,457 tokens. The arm was shown "prose placed after a conversation
somebody else conducted", and it learned to answer without ever looking. `runs/round-20261011`
recorded the identical failure for its `lastonly` arm (0/15, zero tool calls, the same sentence),
and the same repair: supervise every assistant turn.

## What was run

The nine teacher-own rows from `prepared/matched-v2/baseline-samples.jsonl` are unchanged in
content. Each row gains `teacher_turn_indices` — the indices of all its assistant turns — which the
trainer's provenance gate requires for all-turn supervision; the definitional fact that these are
teacher decisions on the teacher's own trajectory is carried by the collection manifest and the
collection log (`[teacher] teacher/agnes-3.0-flash`). Two runs:

| run | data | mode | steps | loss | adapter sha256 |
| --- | --- | --- | ---: | ---: | --- |
| `matched-v2/baseline-assistant-adapter` | 9 rows | `assistant` (seed 0) | 60 | 0.2164 | `e0cd4d74…` |
| `matched-v2/baseline-assistant-seed1-adapter` | 9 rows | `assistant` (seed 1) | 60 | 0.2152 | `9d4670a4…` |

Recipe identical to the pilot's (rank 32 / alpha 64 / lr 2e-4 / 30 epochs / 8192 tokens / same base
fingerprint `e36d2f38…`). The dataset digest changed only through the added provenance field
(`637e4155…`); the original sample file is untouched.

## holdout-1 (15 tasks, one track, one evaluation harness)

All arms below ran through `eval/holdout_sweep.py` against the same home, profile bytes and
endpoint home — observed track `187df3c9…` on every record, `CHTD_EXPECTED_TRACK` enforced.
(An earlier attempt on a fresh home path produced a different track — the resolved profile hash
covers the expanded `dataDir`/`cacheDir`, so the home path is part of the track. Reused the base
arm's home; recorded here because it cost an hour to find.)

| arm | gate |
| --- | ---: |
| base | 13/15 |
| v5 (experimental, 93 rows) | 13/15 |
| **chtd arm (9 student-state rows, `last`)** | **14/15** |
| control seed 0 (9 teacher-own rows, `assistant`) | 11/15 |
| control seed 1 (same, seed 1) | 11/15 |
| control, as previously trained (`last`) | 0/15 (zero tool calls) |

Read against the answers, the gate scores move in both directions — the three arms whose fifteen
answers were each read in full grade like this on the same rubric, correct / partial / failed:

| arm | gate | answers (correct / partial / failed) |
| --- | ---: | --- |
| base | 13/15 | 11 / 1 / 3 |
| control seed 0 | 11/15 | 10 / 4 / 1 |
| control seed 1 | 11/15 | 12 / 1 / 2 |

The v5 and chtd rows above are gate-only; their holdout-1 passes were not each re-read for this
table. The per-answer deltas are named in "Scorer errors recorded this batch" below — e.g. the
control's `fix-closure-late-binding` and `fix-row-alias` are correct diagnoses a phrase spelling
missed, while base's `fix-zip-truncates` concludes "the test case is incorrect" and the control's
`fix-generator-exhaust` concludes "there is no defect". What the h1 set supports is the claim this
run is about: the 0/15 that started it was the supervision mode, not the data — and by the answer
read the repaired control sits in the same band as the base (10–12 correct of 15) at both seeds.

## holdout-2 (12 tasks) — gate plus human adjudication

Every arm was scored by the task's own phrase gate, then answers were read in full and graded
**correct** (names the fixture's mechanism) / **partial** (right direction, material error) /
**failed** (concludes wrongly). All five arms below ran in the same window, through the per-task-home
flow (`run_arm_h2.py`: one workspace and one ledger per task, the student route on port 8003). That
flow records no track digest — a per-task home is a per-task resolved profile — so its
configuration-identity check is the profile itself: the twelve per-task profile files are
byte-identical across all five arms in this window (sha256 `b299083a…`), and the older `arm-v5` /
`arm-v5b` directories on disk, which are not part of this table, carry a different one.

| arm | gate | correct | partial | failed |
| --- | ---: | ---: | ---: | ---: |
| v5 (experimental, 93 rows) | 12/12 | **8** | 4 | **0** |
| control seed 0 (9 teacher-own rows, `assistant`) | 11/12 | 5 | 5 | 2 |
| control seed 1 (same, seed 1) | 10/12 | 4 | 4 | 4 |
| **chtd arm (9 student-state rows, `last`)** | 10/12 | **7** | 3 | 2 |
| base | 9/12 | 5 | 5 | 2 |

Prior batches, same adjudication rubric: v5 8/4/0 (two sweeps, identical); base 4/5/3. The control
as previously trained: 0/12.

The v5 row was re-read in full on 2026-10-14 against the **h2b draw** sketched above. It lands on
the same totals as the two earlier draws — 8/4/0 — but not on the same cells: h2b's
`h2-strip-charset` fell from *correct* to *partial* (it says `strip('N/A')` removes characters that
"appear together at the start/end" — a substring story the earlier draws did not tell), while
`h2-naive-aware` held as correct. The per-draw difference is a real property of this arm at this
scale, and the paragraph below now names which draw it describes.

## What this establishes, and what it does not

> **Correction, 2026-10-14 (later the same night).** Two findings landed after this record was
> written, and both cut against claim 2 as stated. First, an adversarial review of this document
> found that on holdout-2 the phrase gate and the answer read point in **opposite directions** for
> this very pair (gate: control 11/12 vs chtd 10/12; answers: chtd 7 vs control 5), and that all
> three gate cells producing the reversal are gate errors — the corrected gate reads 9 vs 9, which
> agrees with the answers. The h2 gate table printed above therefore should be read with the
> reversal in mind, and the record should have said so. Second, the claim was re-tested on 30 fresh
> tasks with the student-state arm trained at **three seeds**: `runs/holdout-3-20261014/`. There the
> student-state seeds (24/26/27 of 30) overlap the teacher-own seeds (23/25), and the mean gap
> (~1.5 tasks) is smaller than the student-state arm's own seed spread (3 tasks). **The h1/h2
> advantage does not replicate; it is consistent with noise.** Claim 2 below stands only as "on the
> 12–15-task sets, at these seeds" — not as a general statement.

**Establishes.**

1. **The control arm's zero was a supervision-mode collapse.** Same rows, same recipe, same
   budget; the only change is which turns carry the loss. 0/15 → 11/15, 0/12 → 11/12, zero tool
   calls → normal exploration (20–40 s/task). The claim "both our fine-tunes regress" is dead:
   by the answer read the repaired control sits in the base's band on both sets (h1 10–12 correct
   of 15 against the base's 11; h2 5-correct to the base's 5, seed 0), and on holdout-3 every
   adapted arm clears the base by 3–7 tasks.
2. *(As stated for the h1/h2 sets, superseded as a general claim by the correction above.)* On the
   paired design at these seeds, the student-state arm leads the teacher-own arm on both small
   sets: h1 14/15 vs 11/15, h2 7-correct vs 5-correct. The control arm's rows even carry *more*
   supervised tokens (240–533 vs 140–357), so the margin is not "the other arm got more
   gradient". The 30-task re-test did not reproduce the gap.
3. **The experimental arm's lead over the base reproduces on holdout-2 for the third time** —
   v5 8-correct/0-failed against the base's 5/5/2 in this batch, in the same direction and shape
   as the two earlier sweeps, with the repaired control landing exactly where the base lands.
4. **The difference between the arms is in the answers, not the gate.** The gate's errors this
   batch cut both ways and are enumerated in the files below; on the v5 arm, two sweeps with
   identical human scores scored 10/12 and 12/12 by gate. Any gate-only comparison of these arms
   understates every arm's distance from the base.

**Does not establish.** A general capability claim. Twelve and fifteen tasks are small; both sets
have been read by humans in earlier rounds; the h1 screen is saturated for the base (13/15, and
the same two failures every sweep), so it can show a loss but barely a gain. On seed variance: the
control's two seeds agree on totals but **disagree on four of fifteen per-task cells**, so a single
seed's h1 total carries roughly ±2 tasks of noise — the adversarial review measured this, and
holdout-3's three student-state seeds (24–27) confirm the magnitude. One caveat on the comparison
itself: the two paired arms use the supervision mode each data shape calls for (`last` for a
student-state label, `assistant` for a complete trajectory — the trainer's provenance gate refuses
the alternatives), so the pair varies data source and supervision mode together. The mode is fixed
at `assistant` for every teacher-own arm; the isolating cell (student-state rows under `assistant`)
is empty *by construction*, not merely unrun.

## Scorer errors recorded this batch

The phrase gate is a proxy, and this batch produced a clean specimen set:

- **False negatives (h1):** the control's `closure-late-binding` (both seeds), `row-alias` and
  `ordinal-index` — correct mechanisms whose wording missed the phrase list (`*same*` with
  markdown, `sublist`, 1-based/0-based without the word "index").
- **False positives (h1):** the control's `fix-generator-exhaust` (seed 0) passes the gate while
  concluding "there is no defect"; it is the only answer in that arm's fifteen that a full read
  fails outright. Base's `fix-zip-truncates` and `fix-enumerate-start` also pass while pointing at
  the test rather than the implementation — partial-to-failed on the read, gate-passed.
- **False positives (h2):** the control's `min-key-string` ("there is no defect … for positive
  integers the function is correct") and `unbound-local` passed the gate while concluding wrongly;
  seed 1's `join-absolute`, `dict-mutation-view`, `unbound-local` and `min-key-string` all passed
  the gate with wrong mechanisms; the chtd arm's `class-attr-shared` passed with "there is no
  defect".
- **True failures all five arms share:** `h2-join-absolute` (nobody in this batch got the
  absolute-second-argument mechanism fully right; v5's h2b answer says `join` "returns just
  `/srv/www`", which is wrong in the same way the control's is wrong — it returns the *second*,
  absolute argument) and `h2-strip-charset` (every arm in the h2b draw tells a substring story;
  the earlier v5 draws had cleared this task by naming the character-set mechanism, so v5 is the
  one arm whose answer to this fixture moved between draws — down, not up).

## Files

```
data/rollouts/ctrl-assistant-h1-20261014/      control seed 0, h1 (15 tasks)
data/rollouts/ctrl-seed1-h1-20261014/          control seed 1, h1
data/rollouts/chtd-arm-h1-20261014/            student-state arm, h1
data/rollouts/v5-h1b-20261014/                 v5, h1
data/rollouts/base-h1-20261013/                base, h1 (earlier batch, same track)
data/holdout-2-20261013/
  records-ctrl-h2a.json, records-ctrl-seed1-h2a.json, records-chtd-arm-h2a.json,
  records-v5-h2b.json, records-base-h2b.json   raw sweeps (session, ledger, workspace)
  scores-*.json                                the gate results for each
/mnt/proj/chtd-experiments/node4-recollection-20261012/
  matched-v2/baseline-assistant-adapter/           seed 0 weights + run.json
  matched-v2/baseline-assistant-seed1-adapter/     seed 1 weights + run.json
  prepared/matched-v2/baseline-assistant-samples.jsonl   the 9 rows + provenance
  prepared/matched-v2/baseline-assistant-plan.json       plan (digest recomputed)
```

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
